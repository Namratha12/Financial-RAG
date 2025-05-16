# vector_store.py

import faiss
import pickle
import pandas as pd
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = Path("data/faiss_index.bin")
METADATA_PATH = Path("data/faiss_metadata.pkl")
CSV_PATH = Path("data/parsed_convfinqa.csv")
MODEL_NAME = "intfloat/e5-base-v2" 

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(["query: " + query])[0].astype("float32")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        query_vec = self.embed_query(query).reshape(1, -1)
        _, indices = self.index.search(query_vec, k)
        return [self._to_document(self.metadata[i]) for i in indices[0]]

    def _to_document(self, item: dict) -> Document:
        content = f"passage: {item.get('table_markdown', '')}\n\n{item.get('context', '')}"
        return Document(page_content=content, metadata={"id": item.get("id", "")})


class RelevantDocumentRetriever:
    def __init__(self, csv_path: str = str(CSV_PATH)):
        self.df = pd.read_csv(csv_path)
        self.df["clean_question"] = self.df["question"].fillna("").apply(self._clean_text)

    def _clean_text(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", str(text)).strip().lower()

    def query(self, question: str) -> List[Document]:
        cleaned_question = self._clean_text(question)
        row = self.df[self.df["clean_question"] == cleaned_question].head(1)

        if row.empty:
            raise ValueError(f"No matching question found: {question}")

        row_data = row.iloc[0]
        content = f"{row_data['table_markdown']}\n\n{row_data['context']}"
        return [Document(page_content=content, metadata={"id": row_data["id"]})]


# Global instance for RAG pipeline use
vector_store = VectorStore()
