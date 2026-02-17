import asyncio
from dotenv import load_dotenv
import os


load_dotenv()


async def main():
    print("Hello from mcp-server!")


if __name__ == "__main__":
    asyncio.run(main())
