# RLLM SDK Design
> Scope: This document is reserved for SDK API design. Exclude implementation mechanics, storage layouts, and control-flow internals so the surface stays concise.
> See `RLLM_PIPELINE_AND_TRAINER_APIS.md` for rollout processing and trainer API details.

### `RLLMClient`
Top-level orchestrator that wires together tracing, inference, and training services.
```python
from rllm_sdk import RLLMClient

client = RLLMClient(
    storage_endpoint="https://storage",
    training_endpoint="https://training",
    inference_endpoint="https://inference"
)
```
Key capabilities:
- `session(...)`: context manager for grouping traces.
- `entrypoint(...)`: decorator for wrapping existing handlers with automatic session + metadata propagation.
- `get_chat_client(provider, model=...)`: drop-in LiteLLM-backed chat surface (OpenAI, Anthropic, Azure OpenAI, etc.) that routes between production providers and training backend, with per-call provider/model selection.
- `get_pipeline()`: returns a trace pipeline for assembling traces into rewarded episodes.
- `get_policy_trainer()`: consumes prepared episodes, performs policy updates, and surfaces checkpoint/resume APIs.
- `get_agent_trainer()`: drives an agent endpoint on mock data, assembles episodes, and trains end-to-end.

### Concept Map
```
RLLMClient
├─ session(...), entrypoint(...), get_chat_client(...) ──┐
│    └─ emit Trace objects (persisted via Context Store) │
│                                                       │
├─ get_pipeline() ──────────────┐                       │
│   └─ Pipeline                 │                       │
│        • assemble_and_score(traces, assembler, reward)│
│        • emits EpisodeStream                          │
│                                                       │
└─ get_agent_trainer(...) ──────────────────────────────┘
     AgentTrainer
     ├─ pipeline (Pipeline instance)
     ├─ policy_trainer (PolicyTrainer)
     ├─ rollout(endpoint, batches, …) → RolloutQueue → Trace stream
     ├─ fit_agent(...)
     │    └─ rollout(...)
     │    └─ pipeline.assemble_and_score(...)
     │    └─ policy_trainer.step(...)
     └─ validate(...), save_state(...), load_state(...)

RolloutQueue
└─ delivers Trace objects confirmed in the context store

Assembler
└─ consume Trace → buffer/group → emit Episode/EpisodeGroup

Reward adapter
└─ score EpisodeArtifact → attach reward signals

EpisodeStream
└─ consumed by PolicyTrainer.step(...)
```

### Session + Metadata Helpers
```python
with client.session(session_id="eval-run-1", split="validation"):
    ...
```
```python
@client.entrypoint(metadata={"mode": "production"})
def handle_request(payload):
    ...
```
Sessions tag every downstream LLM call; decorators make it easy to retrofit production endpoints without touching handler signatures.

### Retrofit Example: Multi-Step Retrieval
```python
client = RLLMClient(...)
llm = client.get_chat_client(provider="openai", model="gpt-4o-mini")
retriever = DenseSearchClient(...)

@client.entrypoint(metadata={"service": "research-assistant"})
def handle_ticket(ticket):
    with client.session(session_id=f"ticket-{ticket.id}", channel=ticket.source):
        docs = retriever.search(ticket.question, top_k=5)

        summary = llm.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize the evidence."},
                {"role": "user", "content": ticket.question},
                {"role": "assistant", "content": format_docs(docs)},
                {"role": "user", "content": "Give me a concise summary."},
            ]
        )

        answer = llm.chat.completions.create(
            messages=[
                {"role": "system", "content": "Answer with references."},
                {"role": "user", "content": ticket.question},
                {"role": "assistant", "content": summary.choices[0].message.content},
                {"role": "user", "content": "Provide the final response."},
            ]
        )

        return {
            "summary": summary.choices[0].message.content,
            "answer": answer.choices[0].message.content,
            "references": [doc.id for doc in docs],
        }
```
Swap your provider SDK for `get_chat_client`, wrap the handler with `@entrypoint`, and drop a `client.session(...)` in the workflow. Every LLM call lands in the context store with session metadata, ready for rollout or training.

### Chat Client
```python
llm = client.get_chat_client(provider="openai", model="gpt-4o-mini")
response = llm.chat.completions.create(
    messages=[{"role": "user", "content": "Hi"}],
    metadata={"user_id": "123"}
)
```
- `get_chat_client` returns a thin adapter that routes OpenAI-compatible calls through a LiteLLM proxy. The helper exposes the proxy base URL (already scoped to the active session/attempt), required headers carrying serialized metadata, and default request options (`return_token_ids=True`, `logprobs=True`, configurable `top_logprobs`) so downstream engines emit token telemetry consistently.
- In production sessions the proxy forwards calls to the configured provider, then extracts `prompt_token_ids`, per-choice `token_ids`, and logprob payloads from the response and records them via `tracer.log_llm_call`.
- In training/evaluation sessions the same proxy routes traffic to VERL (or another training backend) while preserving the trace schema so episode assemblers see identical token/logprob structures regardless of source.

#### Tracing Implementation
- `session()` allocates rollout/attempt identifiers and exposes them through SDK contextvars. The proxy middleware reuses those identifiers, assigns strictly increasing sequence ids, and attaches metadata to every request/response.
- `@entrypoint` propagates session context into production handlers so any OpenAI client pointed at the proxy automatically carries the correct rollout metadata.
- When the proxy completes an LLM call it invokes `tracer.log_llm_call`, packaging prompt/response text, token ids, logprobs, latency, and metadata so rewards and episode builders receive a complete record without extra instrumentation.
- Rewards emitted via `client.add_reward(...)` or downstream adapters are logged with the same rollout/attempt identifiers, keeping reward events aligned with token-level traces.

### Trace Pipeline + Trainer
```python
pipeline = client.get_pipeline()
episodes = pipeline.process(filters={"session_id": "nightly"}, assembler=my_assembler, reward=qa_reward)

policy_trainer = client.get_policy_trainer()
policy_trainer.step(episodes)
```
`pipeline.process` converts raw traces into rewarded RL episodes; `policy_trainer.step` submits those episodes to the configured backend and updates the model registry.

## Trainer Interfaces

- **Policy trainer (episode-in, offline RL)**: Operates on pre-built `Episode` objects—ideal for benchmark runs, curated datasets, or any workflow where traces are prepared ahead of time. Matches the behavior of `tinker_policy_trainer` today. API surface:
  - `policy_trainer.step(episodes, *, resume_from=None, run_id=None)`: runs an offline training step, optionally resuming from a saved trainer state and tagging the active run.
  - `policy_trainer.save_state(run_id, *, include_model=True)`: checkpoints trainer progress under a stable identifier; toggles whether the current policy weights are captured.
  - `policy_trainer.load_state(run_id)`: restores a previously saved trainer state to continue training without repetition.
  - `policy_trainer.save_model(model_id, *, source_run=None)`: exports the current policy weights to the model registry with optional linkage back to a trainer run.
- **Agent trainer (endpoint-in, on-policy loop)**: Drives an annotated endpoint against provided datasets or simulators, collects fresh trajectories via the SDK wrapper, and then performs training. Mirrors the responsibilities of `tinker_agent_trainer`.

Keep the distinction in mind when choosing APIs: policy trainers never call the agent/environment themselves, while agent trainers orchestrate both rollout and policy update.

## Supported Workflows

### 1. Production Trace Fine-Tuning
Log real traffic and periodically train (policy trainer flow).
```python
chat = client.get_chat_client(provider="openai", model="gpt-4o-mini")
with client.session("nightly", source="prod"):
    response = chat.chat.completions.create(...)
    client.add_metadata(reward=user_feedback_score)

pipeline = client.get_pipeline()
episodes = pipeline.process(filters={"source": "prod"}, assembler=prod_assembler, reward=prod_reward)
client.get_policy_trainer().step(episodes)
```
Ideal when you want staging/production observability plus scheduled RL updates.

### 2. Endpoint-Orchestrated RL (Minimal Integration)
Wrap an existing production handler; reuse it during training without refactoring (agent trainer flow).
```python
@client.entrypoint(metadata={"mode": "production"})
def handle_chat(request):
    llm = client.get_chat_client(provider="openai", model="gpt-4o-mini")
    return llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=request.messages,
        metadata={"user_id": request.user_id}
    )

agent_trainer = client.get_agent_trainer()
# Endpoint-orchestrated training loop
# Step 1: stream rollouts (traces land in the context store and feed the queue)
rollout = agent_trainer.rollout(
    endpoint=handle_chat,
    batches=dataset.iter_batches(batch_size=32),
    session_tags={"job": "nightly-train"},
)

# Step 2: assemble + score traces into episodes using the trainer-owned pipeline
episode_stream = agent_trainer.pipeline.assemble_and_score(
    traces=rollout,
    assembler=endpoint_assembler,
    reward=compute_reward,
)

# Step 3: feed minibatches of episodes into the policy trainer
buffer = []
async for episode in episode_stream:
    buffer.append(episode)
    if len(buffer) == 8:
        await agent_trainer.policy_trainer.step(buffer, run_id="nightly-train")
        buffer.clear()

if buffer:
    await agent_trainer.policy_trainer.step(buffer, run_id="nightly-train")

# Step 4: wait for rollout completion and checkpoint trainer state
await rollout.wait()
agent_trainer.policy_trainer.save_state("nightly-train")
```
Production traffic continues to use the provider API; when invoked inside a training session the same handler transparently talks to VERL.

## API Snapshot
| Capability | Entry Point |
|------------|-------------|
| Start a trace session | `client.session(session_id, **metadata)` |
| Decorate production handler | `@client.entrypoint(metadata=...)` |
| Chat client with tracing | `client.get_chat_client(provider, model=...)` |
| Convert traces to episodes | `client.get_pipeline().process(...)` |
| Offline policy update (episodes) | `client.get_policy_trainer().step(...)` |
| Save trainer checkpoint | `client.get_policy_trainer().save_state(run_id, include_model=True)` |
| Load trainer checkpoint | `client.get_policy_trainer().load_state(run_id)` |
| Publish trained model | `client.get_policy_trainer().save_model(model_id, source_run=None)` |
| On-policy endpoint training | `client.get_agent_trainer().fit_agent(...)` |

## Open Questions / Decisions Pending
- **Chat client scope**: Fixed — it records prompt IDs, completion IDs, and optional logprobs only. Streaming and tool-call handling stay with the underlying provider SDK.
- **Training job control plane**: Fixed — SDK returns a non-blocking handle (future/URL) when submitting jobs so callers can poll externally.
- **Metadata schema**: Still TBD — need a portable convention for production vs training sessions to keep assemblers interoperable.
