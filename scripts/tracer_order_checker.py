#!/usr/bin/env python3
"""Order-checking subscriber (hard-coded config, no argparse).

Subscribes to the context store and verifies that the metadata.counter field
in llm_trace events increments by exactly +1 for every subsequent event.

Assumes a single producer (e.g., scripts/tracer_load_generator.py). Press
Ctrl-C to stop. On mismatch, prints an error and exits non-zero.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

from episodic import ContextStore, ContextSubscriber


# -----------------------------
# Hard-coded configuration
# -----------------------------
ENDPOINT = "http://localhost:8000"  # Context store endpoint
API_KEY = "your-api-key-here"        # Context store API key
PROJECT = "rllm-tracer-demo"        # Namespace/project


async def main_async() -> int:
    store = ContextStore(endpoint=ENDPOINT, api_key=API_KEY)
    subscriber = ContextSubscriber(context_store=store)

    # Internal state
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    expected = -1
    processed = 0
    stop = asyncio.Event()

    @subscriber.on_context_update(namespaces=[PROJECT])
    async def on_update(update):  # type: ignore[no-redef]
        nonlocal expected, processed

        c = update.context
        # Ignore batch-end signals and non-llm traces
        if c.type == "trace_batch_end":
            return

        if c.type != "llm_trace":
            return

        # Extract counter from data.metadata.counter
        data = c.data or {}
        md = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not isinstance(md, dict) or "counter" not in md:
            print(f"Skipping event without counter: id={c.id}")
            return

        counter = md.get("counter")
        try:
            counter = int(counter)
        except Exception:
            print(f"Invalid counter value for id={c.id}: {counter}", file=sys.stderr)
            return  # skip rather than fail hard

        # Order check: must be last + 1
        if counter != expected + 1:
            print(f"ORDER VIOLATION: got={counter}, expected={expected + 1}, id={c.id}", file=sys.stderr)
            # Push the offending event then stop
            await queue.put({"id": c.id, "counter": counter, "expected": expected + 1})
            stop.set()
            return

        expected = counter
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} events. Last counter={expected}")
        await queue.put({"id": c.id, "counter": counter})

    await subscriber.start()

    # Graceful shutdown on Ctrl-C
    loop = asyncio.get_running_loop()

    def _sig_handler():
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _sig_handler)
        loop.add_signal_handler(signal.SIGTERM, _sig_handler)
    except NotImplementedError:
        # Windows: ignore signal registration
        pass

    # Run until stop
    await stop.wait()
    await subscriber.stop()

    # If we stopped due to order violation, return non-zero
    # Peek at last item if it contains expected
    try:
        last = queue.get_nowait()
        if "expected" in last:
            return 1
    except Exception:
        pass
    return 0


def main() -> None:
    exit_code = 0
    try:
        exit_code = asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

