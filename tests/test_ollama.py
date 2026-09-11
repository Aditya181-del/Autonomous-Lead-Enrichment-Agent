import asyncio

from ollama import AsyncClient


async def main():
    client = AsyncClient(
        host="http://localhost:11434"
    )

    response = await client.chat(
        model="nemotron-3-super:cloud",
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly one sentence "
                    "saying that the system is working."
                ),
            }
        ],
    )

    print(
        response.message.content
    )


if __name__ == "__main__":
    asyncio.run(main())