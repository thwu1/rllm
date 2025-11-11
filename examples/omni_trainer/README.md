# Omni Trainer

This example demonstrates how to use the Omni Trainer for reinforcement learning with language models.

## Prerequisites

1. Download and install [episodic](https://github.com/agentica-org/episodic):
   ```bash
   git clone https://github.com/agentica-org/episodic
   cd episodic/episodic-sdk
   pip install -e .
   ```

## Setup

### 1. Launch the Context Store

Start the episodic context store server:

```bash
episodic serve --db-path /tmp/episodic.db  # choose a local path for better performance
```

### 2. Deploy the LiteLLM Proxy

In a separate terminal, start the LiteLLM proxy server:

```bash
#!/bin/bash

# Set ulimit first
ulimit -n 65536

# Set aiohttp connection limits
export AIOHTTP_CONNECTOR_LIMIT=4096
export AIOHTTP_KEEPALIVE_TIMEOUT=60

# Verify the limits are set
echo "Current ulimit -n: $(ulimit -n)"
echo "AIOHTTP_CONNECTOR_LIMIT: $AIOHTTP_CONNECTOR_LIMIT"
echo "AIOHTTP_KEEPALIVE_TIMEOUT: $AIOHTTP_KEEPALIVE_TIMEOUT"
echo "Starting LiteLLM proxy..."

# Start the proxy
python scripts/litellm_proxy_server.py \
  --config litellm_proxy_config_autogen.yaml \
  --host 127.0.0.1 \
  --port 4000 \
  --state-dir /tmp/litellm_proxy \
  --cs-endpoint http://localhost:8000 \
  --cs-api-key "your-api-key-here" \
  --project rllm-agent-omni-engine \
  --admin-token my-shared-secret
```


## Running the Examples

Once both the context store and LiteLLM proxy are running, you can execute one of the training examples:

### Solver-Judge Flow Training

```bash
./train_solver_judge_flow.sh
```

### Hendrycks Math Training

```bash
./train_hendrycks_math.sh
```