"""
rag - Hierarchical Multi-Agent Financial RAG Package

Exports the CIO Supervisor Agent, domain-specialized sub-agents (Fundamental & Market News),
vector indexing pipeline, and RAG query execution engine.
"""

from .rag_engine import (
    FinancialRAGEngine, 
    check_ollama_status, 
    get_llm,
    get_reranker
)
from get_data.chroma_handler import (
    index_all_data,
    get_chroma_vector_store,
    get_chroma_document_count,
    get_embedding_model,
    build_news_documents,
    get_existing_chroma_doc_ids,
)
from .tools import (
    # CIO Supervisor
    CIO_SYSTEM_PROMPT,
    create_cio_agent,
    execute_cio_query,
    get_supervisor_tools,
    # Fundamental Analyst & Tools
    create_fundamental_agent,
    create_fundamental_agent_tool,
    get_fundamental_tools,
    get_stock_fundamentals,
    screen_stocks_by_fundamentals,
    compare_stocks_fundamentals,
    list_available_tickers,
    # Market News Analyst & Tools
    create_news_agent,
    create_news_agent_tool,
    create_news_query_engine_tool,
    get_news_tools,
    get_latest_news_headlines,
)
from get_data import FUNDAMENTALS_FILE, NEWS_FILE
from config import CONFIG

CHROMA_DIR = CONFIG.chroma_dir
COLLECTION_NAME = CONFIG.chroma_collection

__all__ = [
    # Engine & Helpers
    "FinancialRAGEngine",
    "check_ollama_status",
    "get_llm",
    "get_reranker",
    # Indexer
    "index_all_data",
    "get_chroma_vector_store",
    "get_chroma_document_count",
    "get_embedding_model",
    "build_news_documents",
    "get_existing_chroma_doc_ids",
    # CIO Supervisor Agent
    "CIO_SYSTEM_PROMPT",
    "create_cio_agent",
    "execute_cio_query",
    "get_supervisor_tools",
    # Fundamental Tools & Agent
    "create_fundamental_agent",
    "create_fundamental_agent_tool",
    "get_fundamental_tools",
    "get_stock_fundamentals",
    "screen_stocks_by_fundamentals",
    "compare_stocks_fundamentals",
    "list_available_tickers",
    # News Tools & Agent
    "create_news_agent",
    "create_news_agent_tool",
    "create_news_query_engine_tool",
    "get_news_tools",
    "get_latest_news_headlines",
    # Config & Constants
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "FUNDAMENTALS_FILE",
    "NEWS_FILE",
]
