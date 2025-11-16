# Fast Testing Guide for RLLM Training Pipeline

Before deploying expensive vLLM infrastructure, validate your setup incrementally with these automated tests.

## Test 1: SqliteTracer (< 1 min, Free)

Test the tracer component independently:

```bash
python examples/sdk/test_tracer.py --db-path /tmp/test_tracer.db
```

**Tests:** SQLite storage, queue processing, background worker, flush mechanism

**No infrastructure needed!**

---

## Test 2: Proxy Integration (1-2 min, ~$0.02) - **RECOMMENDED**

**Fully automated** - starts proxy, tests, cleans up:

```bash
export OPENAI_API_KEY="sk-..."
python examples/sdk/test_proxy_tracer_standalone.py --db-path /tmp/test_proxy.db
```

**What it does:**
1. ✅ Starts LiteLLM proxy server in subprocess
2. ✅ Configures it with OpenAI backend (gpt-3.5-turbo)
3. ✅ Tests chat completions through proxy
4. ✅ Tests trace collection via TracingCallback
5. ✅ Tests flush endpoint (`/admin/tracer/flush`)
6. ✅ Validates SQLite persistence
7. ✅ Automatic cleanup

**Tests the REAL proxy integration** (not mocked!)

---

## Test 3: Full Training (2-5 min, ~$0.10)

Test complete training pipeline:

```bash
# Start proxy manually for full training
export OPENAI_API_KEY="sk-..."
cd examples/sdk
./start_proxy_openai.sh

# In another terminal
python -m examples.sdk.simple_math.train_hendrycks_math_openai \
    data.train_batch_size=4 \
    trainer.total_epochs=1
```

**Tests:** Complete AgentOmniEngine, VerlProxyManager, training loop

---

## Recommended Workflow

1. **Test Tracer** (< 1 min) → Catches storage issues
2. **Test Proxy** (1-2 min) → **Validates full proxy integration**
3. **Test Training** (2-5 min) → End-to-end validation
4. **Deploy vLLM** → Production with confidence!

See [TESTING_OPTIONS.md](TESTING_OPTIONS.md) for detailed documentation.
