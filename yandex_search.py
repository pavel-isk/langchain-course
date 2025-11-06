from langchain_core.tools import tool

import requests
import json
import dotenv
import os

dotenv.load_dotenv()

SEARCH_API_URL = "https://searchapi.api.cloud.yandex.net/v2/gen/search"
YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER = os.getenv("YC_FOLDER")


@tool
def yandex_gensearch_tool(query: str) -> dict:
    """
    Используется когда нужно найти какую-то информацию в интернете по запросу.
    В качестве ответа возвращает саммари по информации с наиболее релеватных сайтов.
    Args:
      - query (str): запрос на поиск в интернете.
    Returns:
      - dict: словарь с "message" и "sources" ключами.
    """
    headers = {"Authorization": f"Api-Key {YC_API_KEY}"}

    body = json.dumps(
        {
            "folderId": YC_FOLDER,
            "searchType": "SEARCH_TYPE_COM",
            "messages": [
                {
                    "role": "ROLE_ASSISTANT",  # ROLE_USER or ROLE_UNSPECIFIED
                    "content": query,
                }
            ],
            "searchFilters": [{"lang": "ru"}],
        }
    )

    r = requests.post(SEARCH_API_URL, headers=headers, data=body)
    r = json.loads(r.content)
    return r[0]
