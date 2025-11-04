import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()

@tool
def triple_tool(number: float) -> float:
    """Умножает число на три.
    Args:
        number (float): Число для умножения.
    Returns:
        float: Результат умножения на три.
    """
    return float(number) * 3


tools = [TavilySearch(max_results=1), triple_tool]

yc_folder = os.getenv("YC_FOLDER")
yc_api_key = os.getenv("YC_API_KEY")
model_name = os.getenv("YC_MODEL_NAME", "yandexgpt-lite")
yc_base_url = os.getenv("YC_BASE_URL", "https://llm.api.cloud.yandex.net/v1")

print(f"Using model: {model_name}")

llm = ChatOpenAI(
    api_key=yc_api_key,
    base_url=yc_base_url,
    model=f"gpt://{yc_folder}/{model_name}/latest",
    temperature=0,
).bind_tools(tools)

