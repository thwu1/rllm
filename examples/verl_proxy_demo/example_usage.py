"""Example: Using VERL with LiteLLM Proxy for Load-Balanced Inference

This example demonstrates:
1. Setting up VERL engine with multiple vLLM replicas
2. Configuring LiteLLM proxy for load balancing
3. Using OpenAI client with session tracking
4. Automatic metadata routing and tracing
"""

import asyncio
from openai import AsyncOpenAI
from rllm.sdk import RLLMClient
from rllm.engine.proxy_manager import VerlProxyManager


async def example_basic_setup():
    """Example 1: Basic proxy setup with VERL engine."""
    print("=" * 80)
    print("Example 1: Basic Proxy Setup")
    print("=" * 80)
    
    # NOTE: This assumes you have a VERL engine already initialized
    # In practice, you would initialize VERL with your config
    # verl_engine = VerlEngine(config, rollout_manager, tokenizer)
    
    # For demonstration, we'll show the API
    print("""
    # Initialize VERL engine
    from rllm.engine.rollout.verl_engine import VerlEngine
    verl_engine = VerlEngine(config, rollout_manager, tokenizer)
    
    # Create proxy manager
    from rllm.engine.proxy_manager import VerlProxyManager
    proxy_mgr = VerlProxyManager(
        rollout_engine=verl_engine,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        proxy_host="127.0.0.1",
        proxy_port=4000,
    )
    
    # Get server addresses
    servers = proxy_mgr.get_server_addresses()
    print(f"vLLM replicas: {servers}")
    # Output: ['192.168.1.100:8000', '192.168.1.101:8001', '192.168.1.102:8002']
    
    # Get proxy endpoint
    endpoint = proxy_mgr.get_proxy_url()
    print(f"Proxy endpoint: {endpoint}")
    # Output: http://127.0.0.1:4000/v1
    
    # View generated config
    config = proxy_mgr.get_litellm_config()
    print(f"Number of replicas in config: {len(config['model_list'])}")
    """)


async def example_with_agent_omni_engine():
    """Example 2: Using AgentOmniEngine with auto-configured proxy."""
    print("=" * 80)
    print("Example 2: AgentOmniEngine with Auto-Configured Proxy")
    print("=" * 80)
    
    print("""
    from rllm.engine.agent_omni_engine import AgentOmniEngine
    from rllm.sdk import RLLMClient
    
    # Initialize RLLM client
    client = RLLMClient(
        context_store_endpoint="http://localhost:8000",
        project="verl-demo"
    )
    
    # Define agent function
    @client.entrypoint
    async def my_agent(task: str):
        from openai import AsyncOpenAI
        
        # Get endpoint from engine
        openai_client = AsyncOpenAI(
            base_url=engine.get_openai_endpoint(),
            api_key="EMPTY"
        )
        
        response = await openai_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": task}],
        )
        return response.choices[0].message.content
    
    # Create engine with proxy auto-start
    engine = AgentOmniEngine(
        agent_run_func=my_agent,
        rollout_engine=verl_engine,
        proxy_config={
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "proxy_port": 4000,
            "auto_start": True,  # Automatically start proxy
        },
        tracer=client.tracer,
    )
    
    # Proxy is now running at engine.get_openai_endpoint()
    print(f"Proxy running at: {engine.get_openai_endpoint()}")
    
    # Run agent with session context
    result = await my_agent(
        "What is 2+2?",
        _metadata={
            "session_id": "demo-session",
            "experiment": "math-eval"
        }
    )
    print(f"Result: {result}")
    """)


async def example_manual_proxy_lifecycle():
    """Example 3: Manual proxy lifecycle management."""
    print("=" * 80)
    print("Example 3: Manual Proxy Lifecycle Management")
    print("=" * 80)
    
    print("""
    from rllm.engine.proxy_manager import VerlProxyManager
    
    # Create proxy manager
    proxy_mgr = VerlProxyManager(
        rollout_engine=verl_engine,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        proxy_host="0.0.0.0",  # Bind to all interfaces
        proxy_port=4000,
    )
    
    # Write config to file (for inspection or external use)
    config_path = proxy_mgr.write_config_file("./litellm_verl.yaml")
    print(f"Config written to: {config_path}")
    
    # Start proxy server
    proxy_mgr.start_proxy_server()
    print(f"Proxy started at: {proxy_mgr.get_proxy_url()}")
    
    # Check if running
    if proxy_mgr.is_running():
        print("Proxy is running!")
    
    # Use the proxy...
    # (your application code here)
    
    # Stop proxy when done
    proxy_mgr.stop_proxy_server()
    print("Proxy stopped")
    """)


async def example_config_inspection():
    """Example 4: Inspecting generated LiteLLM config."""
    print("=" * 80)
    print("Example 4: Inspecting Generated Config")
    print("=" * 80)
    
    print("""
    import yaml
    from rllm.engine.proxy_manager import VerlProxyManager
    
    proxy_mgr = VerlProxyManager(
        rollout_engine=verl_engine,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        proxy_port=4000,
    )
    
    # Get the config
    config = proxy_mgr.get_litellm_config()
    
    # Print it nicely
    print(yaml.dump(config, default_flow_style=False))
    
    # Output:
    # model_list:
    # - model_name: Qwen/Qwen2.5-7B-Instruct
    #   litellm_params:
    #     model: hosted_vllm/Qwen/Qwen2.5-7B-Instruct
    #     api_base: http://192.168.1.100:8000/v1
    #     drop_params: true
    #   model_info:
    #     id: verl-replica-0
    #     replica_rank: 0
    # - model_name: Qwen/Qwen2.5-7B-Instruct
    #   litellm_params:
    #     model: hosted_vllm/Qwen/Qwen2.5-7B-Instruct
    #     api_base: http://192.168.1.101:8001/v1
    #     drop_params: true
    #   model_info:
    #     id: verl-replica-1
    #     replica_rank: 1
    # ...
    # litellm_settings:
    #   drop_params: true
    #   num_retries: 3
    #   routing_strategy: simple-shuffle
    
    # Inspect individual replicas
    for idx, model_entry in enumerate(config['model_list']):
        api_base = model_entry['litellm_params']['api_base']
        replica_rank = model_entry['model_info']['replica_rank']
        print(f"Replica {replica_rank}: {api_base}")
    """)


async def example_metadata_routing():
    """Example 5: Understanding metadata routing."""
    print("=" * 80)
    print("Example 5: Metadata Routing")
    print("=" * 80)
    
    print("""
    from rllm.sdk.proxy.metadata_slug import (
        encode_metadata_slug,
        build_proxied_base_url,
        assemble_routing_metadata
    )
    
    # Example metadata
    metadata = {
        "session_id": "run-123",
        "experiment": "math-eval",
        "split": "test",
        "model_version": "v1.0"
    }
    
    # Encode metadata to slug
    slug = encode_metadata_slug(metadata)
    print(f"Encoded slug: {slug}")
    # Output: rllm1:eyJleHBlcmltZW50IjoibWF0aC1ldmFsIiwibW9kZWxfdmVyc2lvbiI6InYxLjAiLCJzZXNzaW9uX2lkIjoicnVuLTEyMyIsInNwbGl0IjoidGVzdCJ9
    
    # Build proxied URL
    base_url = "http://localhost:4000/v1"
    proxied_url = build_proxied_base_url(base_url, metadata)
    print(f"Proxied URL: {proxied_url}")
    # Output: http://localhost:4000/meta/rllm1:eyJ...}/v1
    
    # This is what the ProxyTrackedChatClient does automatically:
    from rllm.sdk.chat.proxy_chat_client import ProxyTrackedAsyncChatClient
    from rllm.sdk.context import set_current_session, set_current_metadata
    
    # Set session context
    set_current_session("run-123")
    set_current_metadata({"experiment": "math-eval", "split": "test"})
    
    # Create client
    client = ProxyTrackedAsyncChatClient(
        base_url="http://localhost:4000/v1",
        api_key="EMPTY",
        default_model="Qwen/Qwen2.5-7B-Instruct"
    )
    
    # When you call chat.completions.create(), it automatically:
    # 1. Reads current session context
    # 2. Encodes it into a slug
    # 3. Builds the proxied URL
    # 4. Makes the request
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
    # Request goes to: http://localhost:4000/meta/rllm1:{slug}/v1/chat/completions
    """)


async def example_production_deployment():
    """Example 6: Production deployment pattern."""
    print("=" * 80)
    print("Example 6: Production Deployment")
    print("=" * 80)
    
    print("""
    # Step 1: Generate config file
    from rllm.engine.proxy_manager import VerlProxyManager
    
    proxy_mgr = VerlProxyManager(
        rollout_engine=verl_engine,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        proxy_host="0.0.0.0",
        proxy_port=4000,
    )
    
    config_path = proxy_mgr.write_config_file("/etc/litellm/verl_config.yaml")
    
    # Step 2: Create production proxy app (proxy_app.py)
    # See examples/proxy_demo/proxy_app.py for full implementation
    
    # Step 3: Run with uvicorn (in separate process/container)
    # $ uvicorn proxy_app:app --host 0.0.0.0 --port 4000 --workers 4
    
    # Step 4: Use in your application
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(
        base_url="http://proxy-service:4000/v1",
        api_key="EMPTY"
    )
    
    # All requests are load-balanced across vLLM replicas
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Hello"}]
    )
    """)


async def main():
    """Run all examples."""
    await example_basic_setup()
    print("\n")
    
    await example_with_agent_omni_engine()
    print("\n")
    
    await example_manual_proxy_lifecycle()
    print("\n")
    
    await example_config_inspection()
    print("\n")
    
    await example_metadata_routing()
    print("\n")
    
    await example_production_deployment()
    print("\n")
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

