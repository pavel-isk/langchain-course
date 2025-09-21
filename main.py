"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.chat_models import ChatYandexGPT

# from langchain import hub
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()


tools = [
    TavilySearch(
        include_domains=["hh.ru", "superjob.ru"],
    )
]
llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0.3, model_name="yandexgpt-5-lite")
# react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "format_instructions"],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    print("Starting Hello world project...")

    result = chain.invoke(
        input={
            "input": """
            search for 3 Python Developer jobs in Moscow and list their titles and links.
            """,
        }
    )

    print("Final result:", result)


if __name__ == "__main__":
    main()
