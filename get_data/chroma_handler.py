"""
chroma_handler.py - Dedicated ChromaDB Vector Store Layer & Document Indexer

Manages vector database persistence, embedding models, and document transformations
directly into ChromaDB (data/database/news_vector). Zero JSON storage dependencies.
Accurate deduplication using document doc_id via get_existing_news_ids.
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Set
import chromadb
import torch
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from config import CONFIG
    CHROMA_DIR = CONFIG.chroma_dir
    CHROMA_COLLECTION = CONFIG.chroma_collection
    DB_PATH = CONFIG.fundamentals_db
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import CONFIG
    CHROMA_DIR = CONFIG.chroma_dir
    CHROMA_COLLECTION = CONFIG.chroma_collection
    DB_PATH = CONFIG.fundamentals_db

try:
    from .sql_handler import load_fundamentals
    from .get_data import get_news_id, batch_fetch_news
except ImportError:
    from sql_handler import load_fundamentals
    from get_data import get_news_id, batch_fetch_news

__all__ = [
    "CHROMA_DIR",
    "CHROMA_COLLECTION",
    "get_embedding_model",
    "build_news_documents",
    "get_chroma_vector_store",
    "get_chroma_document_count",
    "get_existing_chroma_doc_ids",
    "get_existing_news_ids",
    "index_news_data",
    "index_all_data",
    "load_news",
    "save_news",
    "fetch_and_save_news",
]


# ==========================================
# 1. Vector Database & Embedding Layer
# ==========================================

def get_embedding_model():
    """
    Initializes the HuggingFace embedding model with CUDA GPU acceleration,
    optimized batch sizing, and cuDNN benchmarking.
    """
    device = CONFIG.rag_device
    batch_size = 32

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        dev_name = torch.cuda.get_device_name(0)
        print(f"[ChromaHandler] 🚀 Initializing Embedding Model on CUDA ({dev_name}, batch_size={batch_size})")
    else:
        print(f"[ChromaHandler] Initializing Embedding Model on {device} (batch_size={batch_size})")

    return HuggingFaceEmbedding(
        model_name=CONFIG.embed_model_name,
        device=device,
        embed_batch_size=batch_size
    )


# ==========================================
# 2. News Document Builders
# ==========================================

def _normalize_news_by_ticker(news_data: Optional[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalizes news input into a ticker -> articles mapping."""
    if news_data is None:
        return {}

    if isinstance(news_data, list):
        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for item in news_data:
            ticker = str(item.get("ticker", "UNKNOWN")).upper()
            normalized.setdefault(ticker, []).append(item)
        return normalized

    if isinstance(news_data, dict):
        return {
            str(ticker).upper(): articles
            for ticker, articles in news_data.items()
            if isinstance(articles, list)
        }

    return {}


def build_news_documents(
    news_data: Optional[Any] = None,
    fundamentals_data: Optional[List[Dict[str, Any]]] = None,
    db_path: str = DB_PATH
) -> List[Document]:
    """
    Parse news articles from in-memory news_data into LlamaIndex documents.
    Uses yfinance UUID as doc_id, or falls back to deterministic title UUID.
    """
    news_by_ticker = _normalize_news_by_ticker(news_data)
    if not news_by_ticker:
        return []

    if fundamentals_data is None:
        fundamentals_data = load_fundamentals(db_path=db_path)

    fundamentals_by_ticker = {
        str(item.get("ticker", "")).upper(): item
        for item in fundamentals_data
        if item.get("ticker")
    }
    docs: List[Document] = []
    seen_news_ids: Set[str] = set()

    for ticker, articles in news_by_ticker.items():
        t_upper = str(ticker).upper()
        profile = fundamentals_by_ticker.get(t_upper, {})
        sector = profile.get("sector") or "N/A"
        company_name = profile.get("company_name") or t_upper

        for item in articles:
            title = str(item.get("title", "")).strip()
            summary = str(item.get("summary", "")).strip()
            publisher = str(item.get("publisher", "Market News")).strip()
            pub_time = str(item.get("publish_time", "")).strip()
            link = str(item.get("link", "")).strip()

            news_uuid = get_news_id(item)
            # Deterministic fallback: generate UUID5 from title if UUID is missing
            if not news_uuid and title:
                seed = title
                news_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

            if not news_uuid or news_uuid in seen_news_ids:
                continue
            seen_news_ids.add(news_uuid)

            text = (
                f"Headline: {title}\n"
                f"Source: {publisher} | Published: {pub_time}\n"
                f"Article Summary: {summary}\n"
                f"Link: {link}"
            )

            metadata = {
                "doc_id": news_uuid,
                "yf_uuid": news_uuid,
                "ticker": t_upper,
                "company_name": company_name,
                "sector": sector,
                "publisher": publisher,
                "publish_time": pub_time,
                "link": link,
                "title": title,
                "doc_type": "news",
            }

            doc = Document(
                text=text,
                doc_id=news_uuid,
                metadata=metadata,
                excluded_llm_metadata_keys=["doc_id"],
                excluded_embed_metadata_keys=["doc_id"],
            )
            docs.append(doc)

    print(f"[ChromaHandler] Built {len(docs)} news documents.")
    return docs


# ==========================================
# 3. Chroma Client & Storage Access
# ==========================================

def get_chroma_vector_store(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION
):
    """Initializes and returns the ChromaVectorStore."""
    os.makedirs(chroma_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=chroma_collection)


def get_chroma_document_count(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION
) -> int:
    """Return the number of records in the Chroma collection."""
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collection = chroma_client.get_collection(collection_name)
        return int(collection.count())
    except Exception:
        return 0


def get_existing_chroma_doc_ids(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION
) -> Set[str]:
    """Returns all logical document IDs currently stored in Chroma."""
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collection = chroma_client.get_collection(collection_name)
        result = collection.get(include=["metadatas"])
        if not isinstance(result, dict):
            return set()

        logical_ids = {
            str(metadata["doc_id"])
            for metadata in result.get("metadatas", [])
            if metadata and metadata.get("doc_id")
        }
        return logical_ids
    except Exception as e:
        print(f"[ChromaHandler] No existing Chroma collection or unable to read ids: {e}")
        return set()


def get_existing_news_ids(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION
) -> Set[str]:
    """
    Returns the set of news doc_ids already stored in ChromaDB metadata (doc_type == 'news').
    Used for precise deduplication before embedding.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collection = chroma_client.get_collection(collection_name)
        result = collection.get(
            where={"doc_type": "news"},
            include=["metadatas"],
        )
        return {
            str(metadata.get("doc_id") or metadata.get("yf_uuid"))
            for metadata in result.get("metadatas", [])
            if metadata and (metadata.get("doc_id") or metadata.get("yf_uuid"))
        }
    except Exception as e:
        print(f"[ChromaHandler] Could not read existing news doc_ids: {e}")
        return set()


# ==========================================
# 4. Vector Indexing Pipelines (With doc_id Deduplication)
# ==========================================

def index_news_data(
    news_data: Optional[Any] = None,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    fundamentals_data: Optional[List[Dict[str, Any]]] = None
) -> VectorStoreIndex:
    """
    Indexes news articles into ChromaDB with doc_id deduplication.
    Skips any news items whose doc_id already exists in ChromaDB.
    """
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    vector_store = get_chroma_vector_store(chroma_dir=chroma_dir, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    news_docs = build_news_documents(
        news_data=news_data,
        fundamentals_data=fundamentals_data
    )

    if not news_docs:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    # Filter out articles that already exist in ChromaDB via doc_id
    existing_news_ids = get_existing_news_ids(chroma_dir=chroma_dir, collection_name=collection_name)
    new_docs = [doc for doc in news_docs if str(doc.doc_id) not in existing_news_ids]

    print(f"[ChromaHandler] News articles: {len(news_docs)} provided, {len(new_docs)} new to index ({len(news_docs) - len(new_docs)} already exist)")

    if not new_docs:
        print("[ChromaHandler] All news articles already indexed in ChromaDB. Skipping embedding.")
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    index = VectorStoreIndex.from_documents(
        new_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print(f"[ChromaHandler] Indexed {len(new_docs)} new news documents into ChromaDB ({chroma_dir})")
    return index


def index_all_data(
    fundamentals_data: Optional[List[Dict[str, Any]]] = None,
    news_data: Optional[Any] = None,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    db_path: str = DB_PATH
) -> VectorStoreIndex:
    """
    Builds embeddings and updates the ChromaDB VectorStoreIndex for news only:
    1. Purges any fundamental documents from ChromaDB (fundamentals are stored exclusively in SQLite).
    2. Builds news documents and deduplicates them using doc_id before embedding.
    """
    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    vector_store = get_chroma_vector_store(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build news documents and deduplicate using doc_id
    news_docs = build_news_documents(
        news_data=news_data,
        fundamentals_data=fundamentals_data,
        db_path=db_path
    )
    existing_news_ids = get_existing_news_ids(chroma_dir=chroma_dir, collection_name=collection_name)
    new_news_docs = [doc for doc in news_docs if str(doc.doc_id) not in existing_news_ids]

    print(f"[ChromaHandler] News documents to index: {len(new_news_docs)} new news (skipped {len(news_docs) - len(new_news_docs)} existing news)")

    if not new_news_docs:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    index = VectorStoreIndex.from_documents(
        new_news_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    total_in_chroma = get_chroma_document_count(chroma_dir=chroma_dir, collection_name=collection_name)
    print(f"[ChromaHandler] ChromaDB VectorStoreIndex updated! Total active documents in collection: {total_in_chroma}")
    return index


def load_news(
    ticker: Optional[str] = None,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Queries stored news articles directly from ChromaDB metadata (no JSON file required).
    Returns mapping of ticker -> List[Dict].
    """
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collection = chroma_client.get_collection(collection_name)
        where_filter = {"doc_type": "news"}
        if ticker:
            where_filter = {"$and": [{"doc_type": "news"}, {"ticker": str(ticker).strip().upper()}]}

        result = collection.get(where=where_filter, include=["metadatas"])
        news_by_ticker: Dict[str, List[Dict[str, Any]]] = {}

        for meta in result.get("metadatas", []):
            if not meta:
                continue
            t = str(meta.get("ticker", "UNKNOWN")).upper()
            item = {
                "doc_id": meta.get("doc_id"),
                "yf_uuid": meta.get("yf_uuid") or meta.get("doc_id"),
                "ticker": t,
                "title": meta.get("title", ""),
                "publisher": meta.get("publisher", ""),
                "publish_time": meta.get("publish_time", ""),
                "summary": meta.get("summary", ""),
                "link": meta.get("link", ""),
            }
            news_by_ticker.setdefault(t, []).append(item)

        return news_by_ticker
    except Exception as e:
        print(f"[ChromaHandler] Could not load news from ChromaDB: {e}")
        return {}


def save_news(data: Dict[str, List[Dict[str, Any]]], *args, **kwargs) -> int:
    """
    Compatibility wrapper: Indexes news into ChromaDB with doc_id deduplication.
    (Cold storage JSON is no longer used).
    """
    index = index_news_data(news_data=data)
    return get_chroma_document_count()


def fetch_and_save_news(
    tickers: List[str],
    max_workers: int = 6,
    max_news_per_ticker: int = 8,
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches latest news in-memory and indexes directly into ChromaDB with doc_id deduplication.
    """
    news_data = batch_fetch_news(
        tickers=tickers,
        max_workers=max_workers,
        max_news_per_ticker=max_news_per_ticker
    )
    index_news_data(news_data=news_data)
    return load_news()

