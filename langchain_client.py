from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

async def main():
    print("Hello langchain-mcp")

if __name__ == "__main__":
    asyncio.run(main())