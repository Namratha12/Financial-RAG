import os
import pickle
import faiss
import numpy as np
import cohere
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- Config ---
INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/faiss_metadata.pkl"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TOP_K_FAISS = 50
TOP_K_FINAL = 10

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not set in .env file.")

# Initialize models
co = cohere.Client(cohere_api_key)
model = SentenceTransformer(MODEL_NAME)

def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve_faiss(query: str, index, metadata, top_k: int = TOP_K_FAISS):
    embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(embedding, top_k)
    return [metadata[i] for i in indices[0]]

def rerank_with_cohere(query: str, candidates: list, top_k: int = TOP_K_FINAL):
    documents = [c["table_markdown"] + "\n\n" + c["context"] for c in candidates]
    results = co.rerank(query=query, documents=documents, top_n=top_k, model="rerank-english-v3.0").results
    return [candidates[result.index] for result in results]

def main():
    index, metadata = load_index_and_metadata()
    query = input("Enter your financial question:\n> ").strip()

    print("\n[INFO] Retrieving candidates from FAISS...")
    candidates = retrieve_faiss(query, index, metadata)

    print("[INFO] Reranking with Cohere...")
    top_docs = rerank_with_cohere(query, candidates)

    print("\n=== Top Reranked Results ===")
    for i, doc in enumerate(top_docs):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {doc['id']}")
        print(f"Question: {doc['question']}")
        print(f"Answer: {doc['answer']}")
        print("Table:\n" + doc["table_markdown"])
        print("Context (truncated):\n" + doc["context"][:300] + "...\n")

if __name__ == "__main__":
    main()
