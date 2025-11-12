from episodic import ContextStore
import json
import asyncio
import random
import time
import numpy as np
from rllm.sdk.tracing import LLMTracer

store = ContextStore(
    endpoint="http://localhost:8000",
    api_key="your-api-key-here",
)

tracer = LLMTracer(
    context_store=store,
    project="rllm-agent-omni-engine",
    max_queue_size=10_000,
    max_concurrent_stores=100,
)


async def enqueue_trace(payload: dict, sema: asyncio.Semaphore):
    """Queue a trace via LLMTracer to mimic LiteLLM callback behavior."""
    async with sema:
        start_time = time.time()
        tracer.log_llm_call(
            name=payload.get("name", "unknown"),
            input=payload.get("input", {}),
            output=payload.get("output", {}),
            model=payload.get("model", "unknown"),
            latency_ms=payload.get("latency_ms", 0.0),
            tokens=payload.get("tokens", {}),
            contexts=payload.get("contexts", []),
            metadata=payload.get("metadata"),
            session_id=payload.get("session_id"),
            trace_id=payload.get("trace_id"),
        )
        end_time = time.time()
        return end_time - start_time


async def main(max_concurrent: int = 512, num_repeat: int = 2):
    datas = []
    with open("/home/tianhao/rllm/result.json", "r") as f:
        for line in f:
            datas.append(json.loads(line))

    datas = datas * num_repeat
    random.shuffle(datas)

    sema = asyncio.Semaphore(max_concurrent)
    tasks = []
    enqueue_start = time.time()
    for data in datas:
        tasks.append(enqueue_trace(data, sema))
    queue_durations = await asyncio.gather(*tasks)
    enqueue_end = time.time()

    print(f"Average enqueue time: {sum(queue_durations) / len(queue_durations)} seconds")
    print(f"Max enqueue time: {max(queue_durations)} seconds")
    print(f"95th percentile enqueue time: {np.percentile(queue_durations, 95)} seconds")
    print(f"99th percentile enqueue time: {np.percentile(queue_durations, 99)} seconds")
    print(f"50th percentile enqueue time: {np.percentile(queue_durations, 50)} seconds")
    print(f"Total enqueue wall time: {enqueue_end - enqueue_start} seconds")

    flush_start = time.time()
    await tracer.flush(timeout=240)
    flush_end = time.time()
    print(f"Tracer flush time: {flush_end - flush_start} seconds")

    # await store.close()


asyncio.run(main())
