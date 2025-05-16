from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from vector_store import vector_store
import os
load_dotenv()

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke([HumanMessage(content="What is 2+2?")])
print(response.content)
print("[DEBUG] COHERE_API_KEY loaded:", os.environ.get("COHERE_API_KEY"))
query = "What was the revenue in 2008?"
hits = vector_store.similarity_search(query, k=5)
for hit in hits:
    print(hit.metadata['id'])
    print(hit.page_content)
