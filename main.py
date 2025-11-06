#!/usr/bin/env python3


from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from yandex_search import yandex_gensearch_tool


load_dotenv()

GENERATE = "generate"
REFLECT = "reflect"


# class MessageGraph(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]


# def generation_node(state: MessageGraph):
#     print("Generation node works")
#     return {"messages": [
#         generate_chain.invoke({
#             "messages": state["messages"]
#         })
#     ]}


# def reflection_node(state: MessageGraph):
#     print("Reflection node works")
#     res = reflect_chain.invoke({
#         "messages": state["messages"]
#     })
#     return {"messages": [HumanMessage(content=res.content)]}


# def should_continue(state: MessageGraph):
#     print("Conditional node works")
#     if len(state["messages"]) > 3:
#         return END
#     return REFLECT


# builder = StateGraph(state_schema=MessageGraph)
# builder.add_node(GENERATE, generation_node)
# builder.add_node(REFLECT, reflection_node)

# builder.set_entry_point(GENERATE)
# builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
# builder.add_edge(REFLECT, GENERATE)

# graph = builder.compile()
# # print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Hello from LangGraph Reflexion Agent!")

#     inputs = HumanMessage(content="""
# Инструкция: Сделай текст для поста в социальную сеть с описанием города лучше и точнее, добавь детализации:
# Текст поста: "Москва — удивительный город, который сочетает в себе богатую историю и современные технологии.
# Здесь всегда кипит жизнь, а культурные события проходят каждый день.
# В Москве отличная инфраструктура, множество парков и уютных кафе.
# Город предлагает огромные возможности для образования и карьеры.
# Я считаю, что Москва — лучший город для жизни!"
# Твой ответ:
# """)

#     response = graph.invoke(inputs)

#     print(response["messages"][-1].content)
