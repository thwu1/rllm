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
  --db-path ~/.rllm/research-common-27.db \
  --project rllm-agent-omni-engine \
  --admin-token my-shared-secret
