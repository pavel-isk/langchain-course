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
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


def main():  # Make use of Pydantic schemas in your agent logic
    print("Starting simple RAG world project...")
    # filename = "sample2.pdf"
    # pdf_path = f"data/{filename}"
    # documents = PyPDFLoader(pdf_path).load()
    # print(f"Loaded {len(documents)} documents from {pdf_path}")
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    # docs = text_splitter.split_documents(documents)
    # print(f"Split into {len(docs)} chunks of text (max. 1000 characters each)")

    llm = ChatYandexGPT(folder_id=os.getenv("YC_FOLDER"), temperature=0, model_name="yandexgpt-5-pro")
    embeddings = YandexGPTEmbeddings(folder_id=os.getenv("YC_FOLDER"), iam_token=os.getenv("YC_API_KEY"))
    
    # vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local("faiss_index_sample2")
    vectorstore = FAISS.load_local(
        # "data/faiss_index_sample", embeddings, allow_dangerous_deserialization=True
        "data/faiss_index_sample2", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(), combine_docs_chain
    )
    
    res = retrieval_chain.invoke(
        {"input": input("Введи свой тупой вопрос: ")}
    )
    print(f"Ответ: {res['answer']}")

if __name__ == "__main__":
    main()
