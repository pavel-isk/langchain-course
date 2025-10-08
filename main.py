"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os
import json

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage,  SystemMessage
from langchain_gigachat import GigaChat
# from langchain_community.chat_models import ChatYandexGPT
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Starting Hello world project...")
    cognitive_distortions = None
    with open("data/app_data.json", "r", encoding="utf-8") as f:
        t = json.loads(f.read())
        cognitive_distortions = "\n".join([
            f"- {k}: {v}" for k, v in t.get("Искажения").items()
        ])

    system_message = SystemMessage(content="""
Ты психологический ассистент, который помогает людям работать по методу Когнитивно-поведенческой терапии (КПТ).
Твоя задача - собрать информацию от пользователя по следующим важным пунктам: 
- автоматические мысли
- эмоции с их инетенсивностью по шкале от 0 до 5
- как он поступил в этой ситуации
- его физические ощущения.
Собирай данные постепенно, задавая уточняющие вопросы. Веди разговор как его друг. 
Не спрашивай все сразу, общение должно казаться живым и естественным.
Как соберёшь данные, сделай небольшое саммари по ситуации пользователя.
Затем, используя эту информацию, сообщить пользователю, какие у него когнитивные искажения с объяснением почему они применимы.
Затем предложить рациональные, более адаптивные мысли, который скорректируют эти искажения.    
Список когнитивных искажений на который ты должен ориентироваться при анализе ситуации пользователя:
{cognitive_distortions}.
""".format(cognitive_distortions=cognitive_distortions.strip()))

    # llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0, model_name="yandexgpt-5-pro")
    # llm = ChatOllama(model="llama3.1:8b", temperature=0)
    llm = GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), temperature=0, model="GigaChat-2-Pro", verify_ssl_certs=False)
    llm_with_tools = llm.bind_tools([ask_user])
    messages = [
        system_message,
        HumanMessage(content="Начинай сбор информации от пользователя."),
    ]

    ai_message = llm_with_tools.invoke(messages)
    print("AI response:", ai_message, len(ai_message.tool_calls))
    messages.append(ai_message)
    
    while ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "ask_user":
                tool_result = ask_user.invoke(tool_args)
                messages.append(ToolMessage(
                    tool_result,
                    tool_call_id=tool_call["id"],
                ))
                ai_message = llm_with_tools.invoke(messages)
                messages.append(ai_message)

    print(f"\033[94mFinal result:\n{ai_message.content}\033[0m")


@tool
def ask_user(message: str ="") -> int:
    """
    Отправляет сообщение от модели message пользователю и возвращает ответ пользователя в виде строки.
    Этот инструмент должен использоваться LLM всякий раз, когда необходимо пообщаться с пользователем, задать вопросы или получить ввод.
    """
    return input(f"\033[92m{message}\033[0m: ")

if __name__ == "__main__":
    main()
