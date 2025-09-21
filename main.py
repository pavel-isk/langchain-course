"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os
from dotenv import load_dotenv

from langchain import hub

# from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.chat_models import ChatYandexGPT
from langchain_tavily import TavilySearch

# from langchain_core.prompts import PromptTemplate


load_dotenv()


tools = [
    TavilySearch(
        include_domains=["hh.ru", "superjob.ru"],
    )
]
llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0.3, model_name="yandexgpt-5-lite")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    print("Starting Hello world project...")

    result = chain.invoke(
        input={
            "input": """
            search for 5 jobs for DevOps for AI products in Saint Petersburg where I can start working without significant ML and AI experience, 
            list their details and provide links to them
            """,
        }
    )

    print("Final result:", result)


if __name__ == "__main__":
    main()
