#!/usr/bin/env python3
"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os

# import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import GET_RATIONAL_THOUGHTS_PROMPT
from schemas import AgentResponse

load_dotenv()


def main():  # Make use of Pydantic schemas in your agent logic
    print("Starting Hello world project...")

    llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0.5, model_name="yandexgpt-5-pro")
    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

    information = load_app_data()
    extract_output = RunnableLambda(lambda x: output_parser.parse(x.content))

    template = GET_RATIONAL_THOUGHTS_PROMPT
    prompt_template = PromptTemplate(
        input_variables=["information", "user_case"],
        template=template,
    ).partial(
        information=information,
        format_instructions=output_parser.get_format_instructions(),
    )

    chain = prompt_template | llm | extract_output

    user_case = {
        # "Ситуация": input("Опиши ситуацию: "),
        # "Мысли": input("Опиши мысли: "),
        # "Эмоции": input("Опиши эмоции и их интенсивность по 10-бальной шкале: "),
        # "Телесная реакция": input("Опиши телесную реакцию: "),
        # "Поведение": input("Опиши поведение: "),
        "Ситуация": "Я опоздал на работу, потому что проспал.",
        "Мысли": "Я всегда опаздываю, я неорганизованный человек и меня уволят. Важно чтобы никто не узнал, что я опоздал.",
        "Эмоции": "Тревога 8, стыд 7",
        "Телесная реакция": "Сердцебиение, потливость, напряжение в плечах",
        "Поведение": "Я быстро оделся и помчался на работу, не позвонив никому и не предупредив.",
    }

    user_case_str = "\n".join([f"{k}: {v}" for k, v in user_case.items()])
    response = chain.invoke(input={"user_case": user_case_str})

    print_response(response)


def load_app_data():
    information = ""
    with open("data/app_data.json", "r", encoding="utf-8") as f:
        information = f.read()
    return information


# Function that prints response in human readable format with colors
def print_response(response: AgentResponse):
    print("Response:")

    print(f"\033[1;97m{response.summary}\033[0m")

    sorted_mistakes = sorted(response.mistakes, key=lambda x: x.probability, reverse=True)
    print(f"\033[38;5;208m Ошибки мышления\033[0m")
    for idx, mistake in enumerate(sorted_mistakes, 1):
        print(f"\033[38;5;208m{idx}. {mistake.name}: {mistake.explanation}\033[0m")

    print(f"\033[91m Установки\033[0m")
    for idx, attitude in enumerate(response.attitudes, 1):
        print(f"\033[91m{idx}. {attitude.content}\033[0m")

    print(f"\033[94m Рациональные мысли\033[0m")
    for idx, thought in enumerate(response.rationalization, 1):
        print(f"\033[94m{idx}. {thought.content}\033[0m")


if __name__ == "__main__":
    main()
