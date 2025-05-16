# state.py
from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(BaseModel):
    messages: List[HumanMessage | AIMessage] = []
    question: str = ""
    queries: List[str] = []
    documents: List[Document] = []
    reranked_documents: List[Document] = []
    context: str = ""
    context_table: str = ""          
    context_narrative: str = ""      
    sources: List[str] = []
    prompt: str = ""
    generation: str = ""
    answer: str = ""

