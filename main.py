#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue, # возвращает END или ACT, то есть решает что делать после вызова агента
    {END:END, ACT:ACT}) # в зависимости от полученной строки переходит к той или иной ноде
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
# app.get_graph().draw_mermaid_png(output_file_path="data/flow.png")


if __name__ == "__main__":
    print("Hello from LangGraph React Agent!")
    user_city = input("Введи город: ")
    user_input = f"Скажи какая погода в {user_city} и умножь численные показатели на 3."
    res = app.invoke(
        {"messages": [
            HumanMessage(content=user_input)
        ]}
    )

    print(res["messages"][LAST].content)