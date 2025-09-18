import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


def main():
    print("Starting DeepSeek example...")

    llm = ChatYandexGPT(
        folder_id=os.getenv("YC_FOLDER"),
        temperature=0.3,
        model_name="yandexgpt-5-lite"
    )

    information = """
    Jeff Bezos is an American entrepreneur and investor best known as the founder and former CEO of Amazon, the world's largest online retailer.
    He also owns The Washington Post and founded Blue Origin, a private aerospace manufacturer and sub-orbital spaceflight services company.
    Bezos is recognized for revolutionizing e-commerce and cloud computing, and is one of the wealthiest individuals globally.
    """

    task = "Describe the person in a short sentence. List two piculiar facts about him."

    template = "Given the information {information} about a person, implement the task: {task}"
    prompt_template = PromptTemplate(
        input_variables=["information", "task"],
        template=template,
    )

    # prompt = prompt_template.format(information=information, task=task)
    # response = llm.invoke([
    #     SystemMessage(content="You are a helpful assistant that helps people find information."),
    #     HumanMessage(content=prompt)
    # ])

    chain = prompt_template | llm

    response = chain.invoke(input={
        "information": information,
        "task": task
    })

    print("Response:")
    print(response.content)

if __name__ == "__main__":
    main()
