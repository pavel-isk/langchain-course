from dotenv import load_dotenv
from yandex_search import yandex_gensearch_tool
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()


def run_queries(search_queries: list[str], **kwargs):
    """Запускает запросы на поиск в интернете."""
    return yandex_gensearch_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
