# llm.py
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.0,
    api_key=OPENAI_API_KEY
)
