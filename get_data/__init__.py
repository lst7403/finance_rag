"""
get_data - Financial Data Acquisition, SQLite Storage & ChromaDB Vector Indexing

Cleanly modularized into three core engines:
1. get_data: In-memory network acquisition & financial calculations (tickers, fundamentals, news).
2. sql_handler: Local SQLite database persistence, schema management, and queries.
3. chroma_handler: Vector database indexing, HuggingFace embeddings, and doc_id deduplication.
"""

from .get_data import (
    FUNDAMENTAL_METRIC_KEYS,
    convert_nan_to_none,
    load_json,
    save_json,
    get_slickcharts_ticker,
    fundamental_analysis,
    batch_fetch_fundamentals,
    get_news_id,
    get_ticker_news,
    batch_fetch_news,
)
from .sql_handler import (
    DB_PATH as FUNDAMENTALS_DB,
    FUNDAMENTAL_METRIC_COLS,
    ALL_COLS,
    get_connection,
    init_db,
    upsert_fundamentals,
    save_fundamentals,
    delete_stock_by_ticker,
    delete_stocks,
    load_fundamentals,
    fetch_all_fundamentals,
    fetch_stock_by_ticker,
    count_fundamentals,
)
from .chroma_handler import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    get_embedding_model,
    build_news_documents,
    get_chroma_vector_store,
    get_chroma_document_count,
    get_existing_chroma_doc_ids,
    get_existing_news_ids,
    index_news_data,
    index_all_data,
    load_news,
    save_news,
    fetch_and_save_news,
)
from .data_pipeline import (
    WORKSPACE_DIR,
    DATA_DIR,
    FUNDAMENTALS_FILE,
    NEWS_FILE,
    run_pipeline,
)
from . import get_data
from . import sql_handler
from . import chroma_handler
from . import data_pipeline

__all__ = [
    # Submodules
    "get_data",
    "sql_handler",
    "chroma_handler",
    "data_pipeline",
    # Constants
    "WORKSPACE_DIR",
    "DATA_DIR",
    "FUNDAMENTALS_DB",
    "FUNDAMENTALS_FILE",
    "NEWS_FILE",
    "CHROMA_DIR",
    "CHROMA_COLLECTION",
    "FUNDAMENTAL_METRIC_KEYS",
    "FUNDAMENTAL_METRIC_COLS",
    "ALL_COLS",
    # In-memory Acquisition (get_data)
    "convert_nan_to_none",
    "load_json",
    "save_json",
    "get_slickcharts_ticker",
    "fundamental_analysis",
    "batch_fetch_fundamentals",
    "get_news_id",
    "get_ticker_news",
    "batch_fetch_news",
    # Relational Storage (sql_handler)
    "get_connection",
    "init_db",
    "upsert_fundamentals",
    "save_fundamentals",
    "delete_stock_by_ticker",
    "delete_stocks",
    "load_fundamentals",
    "fetch_all_fundamentals",
    "fetch_stock_by_ticker",
    "count_fundamentals",
    # Vector Indexing (chroma_handler)
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
    # Orchestration
    "run_pipeline",
]
