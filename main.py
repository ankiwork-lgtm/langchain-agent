from dotenv import load_dotenv

load_dotenv()

# from langchain import hub
# import langchainhub as hub
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse


tools = [TavilySearch()]
llm = ChatOllama(model="llama3.2")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_inst = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions=output_parser.get_format_instructions())


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_inst,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))


chain = agent_executor | extract_output | parse_output


def main():

    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the Noida area on linkedin and list their details"
        }
    )
    print("result")


if __name__ == "__main__":
    main()
