#!/usr/bin/env python3
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    print("Hello from LangGraph React Agent!")
    print(len(os.environ["YC_API_KEY"]))