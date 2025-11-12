#!/usr/bin/env python3
"""Simple tracer load generator (hard-coded config, no argparse).

Continuously emits traces using rllm.sdk.tracing.LLMTracer, counting from 0
to infinity. Randomly alternates between small and large payloads to help test
ordering, batching, and backpressure behavior of the context store.

Press Ctrl-C to stop; the script will flush the tracer queue before exiting.
"""

from __future__ import annotations

import random
import signal
import string
import sys
import threading
import time
from typing import Any

from rllm.sdk.tracing import get_tracer, LLMTracer

# -----------------------------
# Hard-coded configuration
# -----------------------------
ENDPOINT = "http://localhost:8000"  # Context store endpoint
API_KEY = "your-api-key-here"        # Context store API key
PROJECT = "rllm-tracer-demo"        # Namespace/project

# Emission behavior
RATE = 20.0                 # traces per second (approximate)
LARGE_PROB = 0.1            # probability of sending a large payload
MIN_LARGE_BYTES = 50_000    # min size for large payload
MAX_LARGE_BYTES = 150_000   # max size for large payload
SEED: int | None = None     # set to an int for reproducible randomness


def _rand_text(n: int) -> str:
    # Generate a pseudo-random text blob deterministically-per-process
    # (this is not cryptographically strong and is just for load testing).
    alphabet = string.ascii_letters + string.digits + " \n"
    return "".join(random.choice(alphabet) for _ in range(n))

def main() -> None:
    if SEED is not None:
        random.seed(SEED)

    if not ENDPOINT or not API_KEY:
        print("Missing ENDPOINT or API_KEY. Edit scripts/tracer_load_generator.py to set them.", file=sys.stderr)
        sys.exit(2)

    tracer: LLMTracer = get_tracer(project=PROJECT, endpoint=ENDPOINT, api_key=API_KEY)  # type: ignore

    stop_event = threading.Event()

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
        print("\nStopping... Flushing pending traces (best-effort).", file=sys.stderr)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"Starting tracer load: endpoint={ENDPOINT} project={PROJECT} rate={RATE}/s large_prob={LARGE_PROB}")
    i = 0
    delay = 1.0 / max(RATE, 0.0001)

    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            # Alternate payload size randomly
            is_large = random.random() < LARGE_PROB
            if is_large:
                n = random.randint(max(1, MIN_LARGE_BYTES), max(MIN_LARGE_BYTES, MAX_LARGE_BYTES))
                output = {"assistant": _rand_text(n)}
                name = "load.large"
            else:
                output = {"assistant": f"ok-{i}"}
                name = "load.small"

            # Minimal fake token summary and model
            tokens = {"prompt": 1, "completion": 1, "total": 2}
            model = "synthetic/model"
            messages = [{"role": "user", "content": f"count={i}"}]

            tracer.log_llm_call(
                name=name,
                model=model,
                input={"messages": messages},
                output=output,
                latency_ms=0.0,
                tokens=tokens,
                metadata={"counter": i, "large": is_large},
                tags=["synthetic", "load-test"],
            )

            i += 1

            # Pacing to approximate the requested rate
            elapsed = time.perf_counter() - t0
            sleep_for = max(0.0, delay - elapsed)
            if sleep_for:
                time.sleep(sleep_for)

    finally:
        # Best-effort flush (Python thread->async boundary handled by tracer)
        try:
            import asyncio

            asyncio.run(tracer.flush(timeout=15.0))  # type: ignore[attr-defined]
        except Exception:
            pass
        tracer.close_sync()
        print("Done.")


if __name__ == "__main__":
    main()
