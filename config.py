# config.py

from pydantic import BaseModel
from pathlib import Path

class Config(BaseModel):
    # File paths
    data_path: Path = Path("data/parsed_convfinqa.csv")
    faiss_index_path: Path = Path("data/faiss_index.bin")
    metadata_path: Path = Path("data/faiss_metadata.pkl")

    # Limits
    evaluation_sample_limit: int = 100
    database_build_limit: int = 10000

    # Retrieval behavior
    use_ground_truth_retrieval: bool = False

    # LLM generation control
    disable_llm_generation: bool = False

    # Embedding & reranker
    embedding_model_name: str = "intfloat/e5-base-v2"
    reranker_model_name: str = "rerank-english-v3.0"

    # Vector store settings
    top_k_retrieval: int = 30
    top_k_rerank: int = 10

    # LLM generation hyperparameters
    temperature: float = 0.0
    top_p: float = 0.95


# Global singleton config instance
config = Config()
