#!/usr/bin/env python3
"""
This script demonstrates how to use LangChain with YandexGPT to process information
and generate responses based on a given prompt.
"""

import os

# import json
from dotenv import load_dotenv

from langchain import hub
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_community.vectorstores import Pinecone

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


def main():  # Make use of Pydantic schemas in your agent logic
    print("Starting simple RAG world project...")

    llm = GigaChat(credentials=os.getenv("GIGACHAT_CREDENTIALS"), temperature=0, model="GigaChat-2-Pro", verify_ssl_certs=False)
    embeddings = GigaChatEmbeddings(credentials=os.getenv("GIGACHAT_CREDENTIALS"), temperature=0, model="Embeddings", verify_ssl_certs=False)
    result = embeddings.embed_documents(texts=["Привет!"])
    print(result)

    # generate_vs_data("sample.pdf", embeddings)
    # generate_vs_data("sample2.pdf", embeddings)
    
    # vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("faiss_index_sample2")
    # vectorstore = FAISS.load_local(
    #     # "data/faiss_index_sample", embeddings, allow_dangerous_deserialization=True
    #     "data/faiss_index_sample2", embeddings, allow_dangerous_deserialization=True
    # )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(
    #     llm, retrieval_qa_chat_prompt
    # )
    # retrieval_chain = create_retrieval_chain(
    #     vectorstore.as_retriever(), combine_docs_chain
    # )
    
    # res = retrieval_chain.invoke(
    #     {"input": input("Введи свой тупой вопрос: ")}
    # )
    # print(f"Ответ: {res['answer']}")



if __name__ == "__main__":
    main()
