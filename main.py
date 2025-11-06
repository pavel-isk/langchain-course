#!/usr/bin/env python3


from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages

from chains import revisor, first_responder
from tool_executor import execute_tools


load_dotenv()

MAX_ITERATIONS = 2
DRAFT = "draft"
EXECUTE_TOOLS = "execute_tools"
REVISE = "revise"
LAST = -1


def event_loop(state: list[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS


builder = StateGraph(MessagesState)


builder.add_node(DRAFT, first_responder)
builder.add_node(EXECUTE_TOOLS, execute_tools)
builder.add_node(REVISE, revisor)


builder.set_entry_point(DRAFT)
builder.add_edge(DRAFT, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISE)
builder.add_conditional_edges(REVISE, event_loop, path_map={END:END, EXECUTE_TOOLS:EXECUTE_TOOLS})

graph = builder.compile()
# graph.get_graph().draw_mermaid_png(output_file_path="data/reflexion_agent.png")


print("Hello from LangGraph Reflexion Agent!")

inputs = "Напиши научную статью о AI интрументах для медицины. Перечисли стартапы которые работают в этой сфере и уже получили финансирование."
response = graph.invoke(HumanMessage(content=inputs))
print(response[LAST])
