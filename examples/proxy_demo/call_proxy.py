from rllm.sdk import RLLMClient

client = RLLMClient(
    api_key="openai-key",
    base_url="http://localhost:4000/v1",  # proxy base URL
    cs_endpoint="http://localhost:8000",
    cs_api_key="your-api-key-here",
    project="proxy-demo",
)

with client.session(session_id="demo-session", job="nightly"):
    chat = client.get_chat_client()
    response = chat.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    print(response.choices[0].message.content)
