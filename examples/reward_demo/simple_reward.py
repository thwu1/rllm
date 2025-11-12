"""Simple example demonstrating reward assignment for LLM responses."""

import asyncio

from rllm.sdk import get_chat_client, session, set_reward, set_reward_async


def sync_example():
    """Synchronous example of setting rewards."""
    print("=== Synchronous Reward Example ===\n")
    
    # Create a chat client pointing to the proxy
    chat = get_chat_client(
        api_key="token-123",
        base_url="http://localhost:4000/v1",
    )
    
    # Make an LLM call within a session
    with session(job="reward-demo"):
        response = chat.chat.completions.create(
            model="mock-gpt-4",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        
        print(f"Response ID: {response.id}")
        print(f"Response: {response.choices[0].message.content}\n")
        
        # Evaluate the response (in real use, this would be your evaluation logic)
        is_correct = "4" in response.choices[0].message.content
        reward_value = 1.0 if is_correct else 0.0
        
        # Set reward using the response object
        print(f"Setting reward: {reward_value}")
        result = set_reward(
            response,
            reward=reward_value,
            metadata={"is_correct": is_correct, "question": "2+2"},
            admin_token="my-shared-secret"  # Use env var LITELLM_PROXY_ADMIN_TOKEN instead
        )
        print(f"Result: {result}\n")
        
        # Alternative: Set reward using just the ID
        result2 = set_reward(
            response.id,
            reward=reward_value,
            metadata={"method": "using_id"},
            admin_token="my-shared-secret"
        )
        print(f"Result (using ID): {result2}")


async def async_example():
    """Asynchronous example of setting rewards."""
    print("\n=== Asynchronous Reward Example ===\n")
    
    # Create a chat client pointing to the proxy
    chat = get_chat_client(
        api_key="token-123",
        base_url="http://localhost:4000/v1",
    )
    
    # Make an LLM call within a session
    with session(job="reward-demo-async"):
        response = chat.chat.completions.create(
            model="mock-gpt-4",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        
        print(f"Response ID: {response.id}")
        print(f"Response: {response.choices[0].message.content}\n")
        
        # Evaluate the response
        is_correct = "Paris" in response.choices[0].message.content
        reward_value = 1.0 if is_correct else 0.0
        
        # Set reward asynchronously
        print(f"Setting reward asynchronously: {reward_value}")
        result = await set_reward_async(
            response,
            reward=reward_value,
            metadata={"is_correct": is_correct, "question": "capital of France"},
            admin_token="my-shared-secret"
        )
        print(f"Result: {result}")


def batch_example():
    """Example of setting rewards for multiple responses."""
    print("\n=== Batch Reward Example ===\n")
    
    chat = get_chat_client(
        api_key="token-123",
        base_url="http://localhost:4000/v1",
    )
    
    questions = [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 5+5?", "10"),
    ]
    
    with session(job="batch-reward-demo"):
        for question, expected_answer in questions:
            response = chat.chat.completions.create(
                model="mock-gpt-4",
                messages=[{"role": "user", "content": question}],
            )
            
            # Evaluate
            is_correct = expected_answer in response.choices[0].message.content
            reward_value = 1.0 if is_correct else 0.0
            
            # Set reward
            result = set_reward(
                response,
                reward=reward_value,
                metadata={"question": question, "expected": expected_answer},
                admin_token="my-shared-secret"
            )
            
            print(f"Question: {question}")
            print(f"  Response: {response.choices[0].message.content}")
            print(f"  Reward: {reward_value}")
            print(f"  Result: {result['status']}\n")


if __name__ == "__main__":
    # Note: Make sure the LiteLLM proxy server is running before running this example
    # python scripts/litellm_proxy_server.py --config <config> --admin-token my-shared-secret ...
    
    print("Make sure the LiteLLM proxy server is running on http://localhost:4000")
    print("with --admin-token my-shared-secret\n")
    
    try:
        # Run synchronous example
        sync_example()
        
        # Run asynchronous example
        asyncio.run(async_example())
        
        # Run batch example
        batch_example()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. LiteLLM proxy server is running")
        print("2. Admin token is set correctly")
        print("3. Context store is configured")

