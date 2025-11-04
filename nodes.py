from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools


SYSTEM_MESSAGE = """Ты - ассистент, который помогает пользователю с разными задачами, используя инструменты."""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Запускает агента с обоснованиеями в мышлении."""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_MESSAGE},
        *state["messages"],
    ])

    return {"messages": [response]}

tool_node = ToolNode(tools)