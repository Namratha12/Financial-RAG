# build_index.py (updated for E5/BGE models)

import os
import pickle
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Config ---
CSV_PATH = "data/parsed_convfinqa.csv"
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/faiss_metadata.pkl"
MODEL_NAME = "intfloat/e5-base-v2"  # Or "BAAI/bge-large-en"

# --- Document structure ---
class Document(BaseModel):
    id: str
    question: str
    answer: str
    table_markdown: str
    context: str

    def to_text(self) -> str:
        return f"{self.table_markdown}\n\n{self.context}"

# --- Embedding logic ---
def embed_documents(documents: List[Document], model) -> np.ndarray:
    texts = ["passage: " + doc.to_text() for doc in documents]
    return model.encode(texts, convert_to_numpy=True)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
def extract_table_rows(table_markdown: str) -> List[str]:
    rows = table_markdown.strip().split("\n")
    return [r for r in rows if "|" in r and not r.strip().startswith("| ---")]

def to_documents(row, table_rows):
    docs = []
    for tr in table_rows:
        docs.append(Document(
            id=row["id"] + "::" + tr[:20].strip(),  # unique row ID
            question=row["question"],
            answer=row["answer"],
            table_markdown=tr,
            context=row["context"]
        ))
    return docs

def main():
    print(f"[INFO] Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).fillna("")

    documents = []
    for _, row in df.iterrows():
        try:
            table_rows = extract_table_rows(row["table_markdown"])
            row_docs = to_documents(row, table_rows)
            documents.extend(row_docs)
        except Exception as e:
            print(f"[WARN] Skipping row due to validation error: {e}")


    print(f"[INFO] Loaded {len(documents)} valid documents.")

    print(f"[INFO] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Generating embeddings with prefix 'passage:'...")
    embeddings = embed_documents(documents, model)

    print("[INFO] Building FAISS index...")
    index = build_faiss_index(embeddings)

    Path("data").mkdir(exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved index to {INDEX_PATH}")

    metadata = [doc.dict() for doc in documents]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {METADATA_PATH}")

if __name__ == "__main__":
    main()
