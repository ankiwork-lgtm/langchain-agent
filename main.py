from dotenv import load_dotenv
from langchain.tools import tool
#from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from callback import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    print(f"get_text_length enter with {text=}")
    return len(text)


def find_tool_by_name(tools: list, tool_name: str):
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    print("Hello, bind_tools Langchain!")
    tools = [get_text_length]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    llm_with_tools = llm.bind_tools(tools)

    question = "What is the length in characters of text DOG"
    messages = [HumanMessage(content=question)]

    while True:
        response: AIMessage = llm_with_tools.invoke(messages)
        print(response)

        if not response.tool_calls:
            print("Final answer:", response.content)
            break

        messages.append(response)
        for tool_call in response.tool_calls:
            tool_to_use = find_tool_by_name(tools, tool_call["name"])
            result = tool_to_use.invoke(tool_call["args"])
            print(f"Tool result: {result}")
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        print(response.content)
        break