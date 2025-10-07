"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage,  AIMessage
from langchain_gigachat import GigaChat
# from langchain_community.chat_models import ChatYandexGPT
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Starting Hello world project...")

    # llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0, model_name="yandexgpt-5-pro")
    # llm = ChatOllama(model="llama3.1:8b", temperature=0)
    llm = GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), temperature=0, model_name="GigaChat-2", verify_ssl_certs=False)
    llm_with_tools = llm.bind_tools([count_characters])
    messages = [
        HumanMessage("How many characters are in the word 'Hello world'?")
    ]

    ai_message = llm_with_tools.invoke(messages)
    print("AI response:", ai_message.content)
    messages.append(ai_message)
    
    while ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(tool_call)
            
            if tool_name == "count_characters":
                tool_result = count_characters.invoke(tool_args)
                messages.append(ToolMessage(
                    tool_result,
                    tool_call_id=tool_call["id"],
                ))
                ai_message = llm_with_tools.invoke(messages)
                print("AI response:", ai_message.content)

    print("Final result:", ai_message.content)


@tool
def count_characters(text: str ="") -> int:
    """Count the number of characters in a given text."""
    print(f"Counting characters in: {text}")
    return len(text)


if __name__ == "__main__":
    main()
