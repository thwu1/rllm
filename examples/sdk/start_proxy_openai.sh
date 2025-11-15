#!/bin/bash
# Start LiteLLM proxy with OpenAI configuration for fast testing
#
# This script starts the LiteLLM proxy using cheap OpenAI models,
# allowing you to test the full proxy integration including:
# - Proxy routing and metadata injection
# - Trace collection through proxy
# - Proxy flush mechanisms
# - Session tracking
#
# Prerequisites:
#   export OPENAI_API_KEY="sk-..."
#
# Usage:
#   ./start_proxy_openai.sh

set -e

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "Starting LiteLLM proxy with OpenAI configuration..."
echo "API Key: ${OPENAI_API_KEY:0:10}... (truncated)"

# Set ulimit for high concurrency
ulimit -n 65536

# Set aiohttp connection limits
export AIOHTTP_CONNECTOR_LIMIT=4096
export AIOHTTP_KEEPALIVE_TIMEOUT=60

echo "Current ulimit -n: $(ulimit -n)"
echo "AIOHTTP_CONNECTOR_LIMIT: $AIOHTTP_CONNECTOR_LIMIT"
echo "AIOHTTP_KEEPALIVE_TIMEOUT: $AIOHTTP_KEEPALIVE_TIMEOUT"

# Create temp directories
mkdir -p /tmp/litellm_proxy
mkdir -p /tmp/rllm_test

# Start proxy
python -m rllm.sdk.proxy.litellm_server \
  --config examples/omni_trainer/litellm_openai_config.yaml \
  --host 127.0.0.1 \
  --port 4000 \
  --state-dir /tmp/litellm_proxy \
  --db-path /tmp/rllm_test.db \
  --project rllm-openai-test \
  --admin-token my-shared-secret

echo "LiteLLM proxy stopped"
