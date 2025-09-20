"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os
from dotenv import load_dotenv

from langchain import hub
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.chat_models import ChatYandexGPT
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate


load_dotenv()


tools = [TavilySearch()]
llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0.3, model_name="yandexgpt-5-lite")
react_prompt = hub.pull("hwchase17/react")


def main():
    print("Starting Hello world project...")


    task = "2 and 8"
    template = "Multiple the following numbers using the multiply tool: {task}"
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["task"],
    )


    resp = llm.invoke([prompt_template.format(task=task)])
    print("LLM output:", resp.content)


if __name__ == "__main__":
    main()
