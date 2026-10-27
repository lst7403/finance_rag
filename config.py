"""Central application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


def _detect_rag_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class AppConfig:
    workspace_dir: str
    data_dir: str
    database_dir: str
    web_dir: str
    fundamentals_file: str
    fundamentals_db: str
    news_file: str
    chroma_dir: str
    chroma_collection: str
    embed_model_name: str
    ollama_base_url: str
    ollama_model: str
    rerank_model_name: str
    rag_device: str


WORKSPACE_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_DIR / "data"
DATABASE_DIR = DATA_DIR / "database"
WEB_DIR = WORKSPACE_DIR / "web"

CONFIG = AppConfig(
    workspace_dir=str(WORKSPACE_DIR),
    data_dir=str(DATA_DIR),
    database_dir=str(DATABASE_DIR),
    web_dir=str(WEB_DIR),
    fundamentals_file=str(DATA_DIR / "fundamentals.json"),
    fundamentals_db=str(DATABASE_DIR / "foundamental_sql.db"),
    news_file=str(DATA_DIR / "news.json"),
    chroma_dir=str(DATABASE_DIR / "news_vector"),
    chroma_collection="financial_rag_stocks",
    embed_model_name=os.environ.get(
        "EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
    ollama_base_url=os.environ.get(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ),
    ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
    rerank_model_name=os.environ.get(
        "RERANK_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
    rag_device=_detect_rag_device(),
)
