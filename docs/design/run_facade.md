# Run/RunAsync Facade Design

## Overview
- Unify agent execution behind a single entry: `run(args)` and `run_async(args)`.
- The facade owns engine initialization (execution/workflow), trajectory generation, and result normalization.
- Callers choose execution mode (inline/thread/process/async) without knowing engine internals.
- Works the same for REST and in‑process usage.

## Goals
- Single, stable API for offline batch, interactive, and job/REST scenarios.
- Clean separation of orchestration (executors) from agent logic (engines).
- Safe defaults for long‑running/GPU workloads (process isolation via spawn).
- Easy observability: consistent metadata/tracing across all modes.

## API
- Synchronous: `run(args: dict) -> RunResult`
- Asynchronous: `run_async(args: dict) -> Awaitable[RunResult]`
- Optional helpers:
  - `validate_args(args) -> None | raises`
  - `estimate_cost(args) -> CostEstimate`

## Args Schema (suggested)
- `engine`: `"execution" | "workflow"`
- `tasks`: `list[dict] | dict` (normalize to list)
- `engine_args`: dict (e.g., `max_steps`, `tokenizer`, `rollout_engine_args`)
- `agent_args/env_args` or `workflow_args`: dict (engine‑specific)
- `parallelism`: `{ n_parallel: int }` (maps to thread/workflow pools)
- `executor`: `"inline" | "async" | "thread" | "process"`
- `timeouts`: `{ total_s: float | None }`
- `retries`: `{ retry_limit: int }` (task‑level where applicable)
- `tracing`: `{ session_id: str | None, metadata: dict | None }`
- `idempotency_key`: `str | None` (for REST idempotency)

Notes:
- For process executor, args must be picklable; avoid passing live instances; prefer importable class paths + plain dicts.

## Result Schema (RunResult)
- `episodes | trajectories`: list (normalized)
- `metrics`: `{ tokens, latency_ms, success_rate, errors }`
- `artifacts`: optional URIs/ids for persisted outputs (e.g., store keys)
- `debug`: optional minimal diagnostics (e.g., engine used, executor used)

## Executors
- **InlineExecutor** (default for short runs): calls engine directly in current process.
- **AsyncExecutor**: uses current event loop; requires `run_async()`.
- **ThreadExecutor**: runs async work in a worker thread when caller needs sync semantics inside an event loop.
- **ProcessExecutor (spawn)**: isolates CUDA/memory; ideal for long or GPU‑heavy jobs; returns on child completion. Child imports engine classes and constructs instances from args, runs, returns serialized `RunResult`.
  - ⚠️ **Context Variable Propagation**: Python's `contextvars` (used for session tracking) do NOT automatically propagate to spawned child processes. The executor must explicitly capture context in parent and restore in child (see Context Propagation section below).

## Engine Adapters
- **ExecutionRunner**:
  - Instantiates `AgentExecutionEngine` from `engine_args/agent_args/env_args`.
  - Builds tasks, runs execute‑style loop, normalizes to `RunResult`.
- **WorkflowRunner**:
  - Instantiates `AgentWorkflowEngine` from `workflow_args`/rollout args.
  - Uses `execute_tasks(...)` and returns `RunResult`.

## Lifecycle & Flow
1. `run` / `run_async` validates args.
2. Choose executor (inline/async/thread/process).
3. Choose adapter (execution/workflow) and instantiate engine.
4. Run tasks, collect results, normalize to `RunResult`.
5. Trace/log with provided `session_id`/metadata.

## REST Integration
- Minimal REST:
  - `POST /runs?mode=sync` → `run(args)` (optional SSE streaming).
  - `POST /runs` → enqueue with `ProcessExecutor`; return `job_id`.
  - `GET /runs/:id` → status/result; `GET /runs/:id/stream` → SSE.
- REST layer simply wraps `run`/`run_async`; no engine coupling in handlers.

## Error Handling & Cancellation
- Normalize engine exceptions into `RunResult.metrics.errors`; raise only on programmer errors (invalid args).
- Cancellation:
  - Process executor: kill/terminate by `job_id`.
  - Async/thread: cooperative cancellation via timeouts.
- Timeouts: enforce `total_s` at executor level; map engine timeouts to `trajectory_timeout` where applicable.

## Telemetry
- Standardize tracing metadata keys: `session_id`, `job`, `route` (optional).
- Log: engine, executor, tokens, latency_ms, task count; avoid noisy provider internals.
- If using the LiteLLM proxy, retain callback‑based logging; this layer just enriches metadata.

## Security & Isolation
- Process executor for untrusted or large jobs (isolation, memory reclaim).
- In‑process/thread for trusted, short tasks.
- REST edge can enforce auth, quotas, idempotency keys.

## MVP Scope
- Implement `RunService` with:
  - `InlineExecutor` and `ProcessExecutor`.
  - `ExecutionRunner` adapter (add `WorkflowRunner` next).
  - Minimal args validation + normalization.
  - Basic tracing hook passthrough.
- Files:
  - `rllm/runner/run_service.py` (service, executors, adapters)
  - `rllm/runner/types.py` (RunArgs, RunResult dataclasses)

## Context Propagation (Critical for Session Tracking)

**Problem**: Python's `contextvars` (used by `RLLMClient.session()` for automatic session tracking) do NOT propagate to child processes when using `multiprocessing` with `spawn` method.

**Impact**: Without explicit handling, all traces in ProcessExecutor will have `session_id=None`, breaking session grouping.

**✅ Solution: Use `@client.entrypoint` Decorator**

The RLLM SDK provides the `@client.entrypoint` decorator that solves this problem elegantly:

```python
from rllm.sdk import RLLMClient

client = RLLMClient()

# User wraps their agent function with entrypoint (no arguments)
@client.entrypoint
def my_agent_function(task):
    # All LLM calls here get the metadata from _metadata kwarg
    llm = client.get_chat_client(provider="openai")
    response = llm.chat.completions.create(...)
    return response

# User calls normally (auto-generated session, no metadata):
my_agent_function(task)

# Run Facade calls with dynamic metadata:
my_agent_function(task, _metadata={"session_id": "run-123", "experiment": "v1"})
```

**How it works**:
1. User decorates function with `@client.entrypoint` (no arguments)
2. Run Facade passes `_metadata` kwarg when calling the function
3. The decorator extracts `_metadata` and creates a session context
4. When ProcessExecutor spawns child process:
   - Child process starts with NO context (spawn method)
   - Entrypoint decorator creates NEW session context from `_metadata`
   - All LLM calls inside get correct session_id and metadata
5. No manual context passing needed!

**Key Design Decisions**:
- Decorator takes NO arguments (no metadata merge complexity)
- ALL metadata comes from `_metadata` kwarg at call time
- Run Facade has full control over session_id and metadata
- Functions remain picklable for multiprocessing

**Why spawn?** The design requires `spawn` for GPU isolation (CUDA safety). Alternative methods:
- `fork`: Inherits context but **unsafe with CUDA** (GPU state corruption)
- `forkserver`: Still doesn't inherit context set after server starts

**User Experience**: Users simply wrap their agent function with `@client.entrypoint` and the Run Facade ProcessExecutor passes `_metadata` transparently.

See `docs/design/context_propagation_multiprocess.md` for detailed analysis and test results.

## Open Questions
- How much of agent/environment construction should the facade own vs. caller factories?
- Do we allow per-task overrides for sampling/model within a batch?
- Default policy for switching to `ProcessExecutor` (threshold by runtime/tokens/GPU flag)?
- Where to persist large results (token IDs/logprobs) — episodic store only or also a blob store?

## Decisions & Clarifications

- Environment ownership
  - Decision: The caller (user) is responsible for environment setup within `run`/`run_async`. The facade does not create environments; it passes through `env_args` and any factories supplied by the user.

- Per‑task overrides (what it means and default)
  - Meaning: In a single `run`, each task can specify its own `model` and `sampling` parameters that override run‑level defaults (e.g., temperature, max_tokens, top_p).
  - Default: Supported. Precedence: task‑level > run‑level defaults > engine defaults. Note: mixed models/sampling in the same batch may reduce batching efficiency or require per‑task routing; engines should handle tasks independently.

- Executor auto‑selection policy (what it means and default)
  - Meaning: If `executor="auto"`, the facade selects Inline/Async/Process based on workload heuristics so callers don’t have to choose.
  - Default policy:
    - Choose `ProcessExecutor (spawn)` when any of:
      - `engine_args` or tasks indicate GPU usage, or
      - `timeouts.total_s` > 30s, or
      - `len(tasks)` ≥ parallelism and estimated tokens/work > threshold.
    - Otherwise: use Inline for `run()` or Async for `run_async()`.
  - Callers can always override `executor` explicitly.

- Persistence of large results
  - Decision: Persist large results (e.g., token IDs, logprobs) in the Episodic Context Store under the project namespace. Optionally compress or summarize when arrays are very large; store references/IDs in `RunResult.artifacts` as needed.

- Context propagation for ProcessExecutor
  - Decision: Users wrap their agent functions with `@client.entrypoint` decorator (no arguments). Run Facade passes `_metadata` kwarg when calling the function. The decorator automatically creates session context from `_metadata` in child processes.
  - Implementation: The `@client.entrypoint` decorator is implemented in `rllm/sdk/client.py` and tested in `rllm/sdk/test_entrypoint.py`.
  - Technical details: See `docs/design/context_propagation_multiprocess.md` for analysis and test results.
