import os

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


load_dotenv()


reflection_prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    "Ты являешь популярным блоггером на ВКонтакте (VK) и очень много знаешь про ведение контента в социальных сетях."
    " Ты должен просмотреть текст поста в социальную сеть и выдать рекомендации по его улучшению, чтобы он был наиболее интересен аудитории."
    " Всегда предоставляй детальные рекомендации, включая такая параметры как длинна поста, возможноть завируситься, стиль подачи материала и так далее.",
  ),
  MessagesPlaceholder(variable_name="messages")
])

generation_prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    "Ты популярный блоггер про путешествия по городам России у которого стоит задача сделать качественный пост для размещения на своей странице в соц сети."
    " Сделай лучший пост который только можно в соответствии с запросом пользователя."
    " Если пользователь даёт какую то критику через обратную связь, обнови предыдущую версию поста в соответствии с рекомендациями.",
  ),
  MessagesPlaceholder(variable_name="messages")
])


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


generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
