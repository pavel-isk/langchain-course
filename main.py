"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os
from dotenv import load_dotenv
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

    information = """
    Вот список распространённых ошибок мышления, характерных для депрессивного состояния:
    1. Катастрофизация — ожидание худшего исхода событий.
    2. Чёрно-белое мышление — видение ситуации только в крайностях, без нюансов.
    3. Обесценивание позитивного — игнорирование или преуменьшение своих достижений и положительных событий.
    4. Персонализация — принятие на себя чрезмерной ответственности за негативные события.
    5. Обобщение — вывод глобальных негативных выводов на основе единичных случаев.
    6. Фильтрация — сосредоточение только на негативных деталях, игнорируя позитивные.
    7. Эмоциональное рассуждение — выводы на основе эмоций, а не фактов.
    8. Предсказание будущего — уверенность, что всё будет плохо, без объективных оснований.
    9. Чтение мыслей — предположение, что окружающие думают о вас негативно.
    10. Навешивание ярлыков — использование негативных определений для себя или других.
    """

    template = """
        Ты бот помощник, который помогает людям распознавать ошибки эмоционального мышления и предлагает рациональные альтернативы.
        Используя информацию об ошибках эмоционального мышления человека: {information},
        предложи, какие ошибки совершает человек в следующей ситуации, мыслях и реакции: {situation}.
        Предложи рациональные альтернативные мысли и реакции.
        Ответ должен быть на русском языке.
    """
    prompt_template = PromptTemplate(
        input_variables=["information", "situation"],
        template=template,
    ).partial(information=information)

    # prompt = prompt_template.format(information=information, task=task)
    # response = llm.invoke([
    #     SystemMessage(content="You are a helpful assistant that helps people find information."),
    #     HumanMessage(content=prompt)
    # ])

    chain = prompt_template | llm


    # situation = "Коллега по работе не поздоровался со мной утром. Наверное, он меня не любит и хочет уволить. Я зажимаюсь в плечах и чувствую себя подавленным."
    situation = input("Напиши что у тебя случилось: ")

    while situation != "пока":
        response = chain.invoke(input={"situation": situation})
        print("Response:")
        print(f"\033[92m{response.content}\033[0m")

        situation = input("Напиши что думаешь или пока: ")


if __name__ == "__main__":
    main()
