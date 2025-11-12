import asyncio
from episodic import ContextStore, ContextSubscriber

context_store = ContextStore(
    endpoint="http://localhost:8000",
    api_key="your-api-key-here",
)

async def main():

    
    await context_store.store(
        context_id="abc",
        data={"test": "test"},
        namespace="test",
    )

asyncio.run(main())
