from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
load_dotenv()


#tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    tool that searches over internet
    Args:
        query: the query to search for
    Returns:
        The search result
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOllama(model="llama3.2")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain course from Ankit")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Delhi NCR area on linkedin and list their details")})
    print(result["messages"][-1].content)   


if __name__ == "__main__":
    main()
