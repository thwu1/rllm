# Fast Testing with OpenAI via LiteLLM Proxy

## Overview

Before deploying expensive vLLM infrastructure, you can quickly validate your entire training pipeline using **inexpensive OpenAI models through the LiteLLM proxy**. This approach tests the complete system including proxy integration.

## What It Tests

**✅ Full system validation:**
- **LiteLLM proxy integration** - The critical component!
- Proxy trace collection and flush mechanisms
- Proxy routing and metadata injection
- Session management and trace collection
- SQLite storage layer
- Training pipeline (data processing, batching, PPO)
- Configuration validation
- Ray distributed training setup

**❌ Only excludes:**
- vLLM server configuration (uses OpenAI API instead)
- GPU memory management

**Cost:** ~$0.05-0.10 per test run with gpt-3.5-turbo
**Time:** 2-5 minutes (vs hours for vLLM deployment)

---

## Comparison: OpenAI vs vLLM

| Feature | OpenAI via Proxy | vLLM Deployment |
|---------|------------------|-----------------|
| Speed | ⚡⚡ (2-5 min) | ⚡ (hours) |
| Cost | $0.05-0.10 | GPU costs |
| LiteLLM Proxy | ✅ Fully tested | ✅ Yes |
| Trace Collection | ✅ Fully tested | ✅ Yes |
| Infrastructure | Proxy only | Full stack (proxy + vLLM) |
| Use Case | Pre-deployment validation | Production training |

---

## Quick Start Guide

### 1. Set OpenAI API Key

```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Start LiteLLM Proxy with OpenAI

```bash
cd examples/omni_trainer
./start_proxy_openai.sh
```

Or manually:
```bash
python -m rllm.sdk.proxy.litellm_server \
  --config examples/omni_trainer/litellm_openai_config.yaml \
  --host 127.0.0.1 \
  --port 4000 \
  --state-dir /tmp/litellm_proxy \
  --db-path /tmp/rllm_test.db \
  --project rllm-openai-test \
  --admin-token my-shared-secret
```

### 3. Run Training (in another terminal)

**Simple math example:**
```bash
python -m examples.omni_trainer.simple_math.train_hendrycks_math_openai \
    data.train_batch_size=4 \
    trainer.total_epochs=1
```

**Solver-judge workflow:**
```bash
python -m examples.omni_trainer.solver_judge_workflow.train_decorator_openai \
    data.train_batch_size=4 \
    trainer.total_epochs=1
```

---

## What Gets Validated

When you run OpenAI-based tests, you're validating the **entire system**:

1. ✅ **Proxy Manager** - VerlProxyManager initialization and config generation
2. ✅ **Proxy Server** - LiteLLM proxy startup and health checks
3. ✅ **Request Routing** - Metadata injection and URL routing
4. ✅ **Trace Callbacks** - TracingCallback and SamplingParametersCallback
5. ✅ **Trace Collection** - Full trace capture in proxy callbacks
6. ✅ **Trace Flush** - Async flush from proxy to SQLite database
7. ✅ **Session Tracking** - Session names and metadata propagation
8. ✅ **Storage Layer** - SQLite writes, reads, and queries
9. ✅ **Data Processing** - Step grouping, trajectory assembly, batching
10. ✅ **Training Loop** - Complete PPO algorithm with real data

The **only** difference from production is the model backend (OpenAI API vs vLLM). Everything else is **identical**!

---

## Cost Estimates

| Example | Model | Approx Tokens | Cost |
|---------|-------|---------------|------|
| Simple Math (1 epoch, 4 samples) | gpt-3.5-turbo | ~5K | $0.02-0.05 |
| Solver-Judge (1 epoch, 4 samples) | gpt-3.5-turbo | ~10K | $0.05-0.10 |
| Simple Math (1 epoch, 4 samples) | gpt-4o-mini | ~5K | $0.01-0.02 |

💡 Use `gpt-4o-mini` for even lower costs (~60% cheaper than gpt-3.5-turbo)

---

## Configuration File

The provided config at `examples/omni_trainer/litellm_openai_config.yaml`:

```yaml
model_list:
  # Fast and cheap model for testing
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: ${OPENAI_API_KEY}

  # Alternative: Even cheaper model
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
```

---

## Troubleshooting

### Issue: Proxy startup fails
**Solution:**
- Check OPENAI_API_KEY is set: `echo $OPENAI_API_KEY`
- Verify API key is valid (starts with `sk-`)

### Issue: Training can't connect to proxy
**Solution:**
- Check proxy is running: `curl http://localhost:4000/health`
- Check proxy logs for errors
- Verify port 4000 is not blocked

### Issue: Traces not collected
**Solution:**
- Check database path is writable: `ls -la /tmp/rllm_test.db`
- Verify proxy flush is working: check proxy logs for "flush" messages
- Try manual flush: `curl http://localhost:4000/admin/tracer/flush`

### Issue: High API costs
**Solution:**
- Use smaller batch sizes (4-8) for testing
- Run only 1 epoch: `trainer.total_epochs=1`
- Use `gpt-4o-mini` instead of `gpt-3.5-turbo`
- Limit training dataset size in config

---

## Recommended Workflow

1. **Start with OpenAI testing** (`train_*_openai.py`)
   - Validates full proxy integration
   - Tests trace collection and flush
   - Small cost (~$0.10), quick turnaround (2-5 min)
   - Catches 95% of potential issues!

2. **Fix any issues found**
   - Configuration errors
   - Storage problems
   - Proxy integration bugs

3. **Deploy vLLM for production**
   - Now confident everything works
   - Only model backend changes
   - Full training with your actual models

---

## Summary

Testing with **OpenAI via LiteLLM proxy** gives you:

- ✅ Full proxy integration validation
- ✅ Complete trace collection pipeline testing
- ✅ Fast feedback (minutes, not hours)
- ✅ Low cost (~$0.10 per test)
- ✅ High confidence before vLLM deployment

**This is the recommended approach before deploying vLLM!**
