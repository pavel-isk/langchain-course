import datetime
import os
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser

from schemas import AnswerQuestion, ReviseAnswer


load_dotenv()


yc_folder = os.getenv("YC_FOLDER")
yc_api_key = os.getenv("YC_API_KEY")
model_name = os.getenv("YC_MODEL_NAME", "yandexgpt")
yc_base_url = os.getenv("YC_BASE_URL", "https://llm.api.cloud.yandex.net/v1")

llm = ChatOpenAI(
    api_key=yc_api_key,
    base_url=yc_base_url,
    model=f"gpt://{yc_folder}/{model_name}/latest",
    temperature=0,
)
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты являешься профессиональным исследователем.
Текущее время: {time}.

1. {first_instruction}.
2. Обдумай и предоставь критический отзыв на свой ответ. Будь строгим, чтобы максимизировать улучшения.
3. Порекомендуй запросы на поиск информации, чтобы улучшить свой ответ.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda: datetime.datetime.now(tz=ZoneInfo("Europe/Moscow")).isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Напиши ответ длинной примерно 250 слов."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"  # set a tool to be always used by llm
)

revise_instructions = """Обнови свой предыдущий ответ используя новую информацию.
- Ты обязан использовать рецензию предыдущего ответа и добавить важную информацию в ответ.
  - Ты должен добавить пронумерованные цитаты в свой обновлённый ответ, чтобы он мог быть верифицирован.
  - Добавь секцию "Ссылки" в конце ответа (эта секция не включается в лимит слов). Например:
    - [1] https://sourcelink.com
    - [2] https://anothersource.net
  - Используй рецензию к предыдущему ответу, чтоб удалить лишнюю информацию и УБЕДИСЬ что ты добавил пропущенную информацию."""

revisor = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Напиши научную статью о AI интрументах для медицины."
        "Перечисли стартапы которые работают в этой сфере и уже получили финансирование."
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
