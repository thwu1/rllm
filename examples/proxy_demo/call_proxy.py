from rllm.sdk import get_chat_client, session, set_reward

# client = RLLMClient(
#     api_key="token-123",
#     base_url="http://localhost:4000/v1",  # proxy base URL
#     cs_endpoint="http://localhost:8000",
#     cs_api_key="your-api-key-here",
#     project="proxy-demo",
# )

chat = get_chat_client(
    api_key="token-123",
    base_url="http://localhost:4000/v1",  # proxy base URL
)
with session(job="nightly"):
    response = chat.chat.completions.create(
        model="mock-gpt-4",
        messages=[{"role": "user", "content": "hello"}],
    )
    print(response)

set_reward(response, 1.0)
