#!/usr/bin/env python3
"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os
# import json
from dotenv import load_dotenv
from prompt import GET_RATIONAL_THOUGHTS_PROMPT
from langchain_core.prompts import PromptTemplate

# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatYandexGPT

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


def main():
    print("Starting Hello world project...")

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     max_retries=2,
    # )
    llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0.3, model_name="yandexgpt-5-pro")
    # llm = ChatOllama(
    #     temperature=0.3,
    #     model="gemma3:270m"
    # )

    information = ""
    with open("data/app_data.json", "r", encoding="utf-8") as f:
        information = f.read()

    template = GET_RATIONAL_THOUGHTS_PROMPT
    prompt_template = PromptTemplate(
        input_variables=["information", "user_case"],
        template=template,
    ).partial(information=information)

    chain = prompt_template | llm
    user_case = {
        "Ситуация": input("Опиши ситуацию: "),
        "Мысли": input("Опиши мысли: "),
        "Эмоции": input("Опиши эмоции и их интенсивность по 10-бальной шкале: "),
        "Телесная реакция": input("Опиши телесную реакцию: "),
        "Поведение": input("Опиши поведение: "),
    }

    user_case_str = "\n".join([f"{k}: {v}" for k, v in user_case.items()])
    response = chain.invoke(input={"user_case": user_case_str})
    print("Response:")
    print(f"\033[92m{response.content}\033[0m")


if __name__ == "__main__":
    main()
