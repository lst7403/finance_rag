"""
data_pipeline.py - Unified Daily Automation Pipeline Orchestrator

Integrates the 3 core layers of the data subsystem:
1. Data Acquisition (get_data): Pure in-memory scraping of tickers, financial metrics, and news.
2. Relational Storage (sql_handler): Atomic upserts & queries in SQLite (data/database/foundamental_sql.db).
3. Vector Database (chroma_handler): Document building and ChromaDB vector indexing with doc_id deduplication.

Zero JSON storage, zero JSON migration, zero JSON fallbacks.
Runnable as a daily automation script via CLI, cron, or Web UI 1-Click Trigger.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG

try:
    from .get_data import (
        FUNDAMENTAL_METRIC_KEYS,
        convert_nan_to_none,
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
    )
except ImportError:
    from get_data import (
        FUNDAMENTAL_METRIC_KEYS,
        convert_nan_to_none,
        get_slickcharts_ticker,
        fundamental_analysis,
        batch_fetch_fundamentals,
        get_news_id,
        get_ticker_news,
        batch_fetch_news,
    )
    from sql_handler import (
        DB_PATH as FUNDAMENTALS_DB,
        FUNDAMENTAL_METRIC_COLS,
        ALL_COLS,
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
    from chroma_handler import (
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
    )

# Compatibility exports
WORKSPACE_DIR = CONFIG.workspace_dir
DATA_DIR = CONFIG.data_dir
FUNDAMENTALS_FILE = CONFIG.fundamentals_file
NEWS_FILE = CONFIG.news_file

__all__ = [
    "WORKSPACE_DIR",
    "DATA_DIR",
    "FUNDAMENTALS_DB",
    "CHROMA_DIR",
    "CHROMA_COLLECTION",
    "FUNDAMENTAL_METRIC_KEYS",
    "convert_nan_to_none",
    "get_slickcharts_ticker",
    "fundamental_analysis",
    "batch_fetch_fundamentals",
    "get_news_id",
    "get_ticker_news",
    "batch_fetch_news",
    "init_db",
    "upsert_fundamentals",
    "save_fundamentals",
    "load_fundamentals",
    "fetch_all_fundamentals",
    "fetch_stock_by_ticker",
    "count_fundamentals",
    "index_news_data",
    "index_all_data",
    "load_news",
    "run_pipeline",
]


# ==========================================
# Unified Pipeline Orchestration
# ==========================================

def run_pipeline(
    refresh_fundamentals: bool = True,
    refresh_news: bool = True,
    index_type: str = "nasdaq100",
    max_workers: int = 6,
    max_news_per_ticker: int = 10,
    db_path: str = FUNDAMENTALS_DB,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> Dict[str, Any]:
    """
    Executes the unified data pipeline end-to-end for daily automation:
    1. Scrapes constituent tickers from Slickcharts (with automatic fallback to SQLite)
    2. Concurrently extracts financial metrics & computes key ratios (get_data)
    3. Concurrently harvests latest news headlines & catalysts (get_data)
    4. Atomically upserts fundamentals into SQLite (sql_handler, safe without accidental deletions)
    5. Deduplicates news against ChromaDB and indexes vectors (chroma_handler)
    """
    start_time = time.time()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"🚀 [Daily Pipeline] Financial RAG Ingestion Started at {now_str}")
    print(f"   • Database : SQLite ({db_path})")
    print(f"   • VectorDB : ChromaDB ({chroma_dir})")
    print(f"   • Config   : Refresh Funds={refresh_fundamentals} | Refresh News={refresh_news} | Index={index_type}")
    print("=" * 70)

    existing_stock_count = count_fundamentals(db_path=db_path)
    tickers: List[str] = []

    # Step 1: Resilient Ticker Resolution
    print(f"\n[Step 1/3] 🌐 Resolving {index_type} constituent tickers...")
    try:
        tickers = get_slickcharts_ticker(index_type=index_type, limit=100)
        print(f"✓ Retrieved {len(tickers)} constituent tickers from Slickcharts.")
    except Exception as e:
        print(f"[Pipeline] ⚠️ Slickcharts scraping unavailable ({e}).")
        if existing_stock_count > 0:
            existing_stocks = load_fundamentals(db_path=db_path)
            tickers = [s["ticker"] for s in existing_stocks if s.get("ticker")]
            print(f"[Pipeline] 🔄 Fallback: Using {len(tickers)} existing tickers from SQLite database.")
        else:
            raise RuntimeError(f"Unable to scrape Slickcharts and SQLite database is empty: {e}")

    # Step 2: Fundamentals Ingestion & Safe Atomic Upsert
    fundamentals: List[Dict[str, Any]] = []
    if refresh_fundamentals or existing_stock_count == 0:
        print(f"\n[Step 2/3] 📊 Fetching & computing fundamentals for {len(tickers)} stocks via yfinance...")
        fetched_fundamentals = batch_fetch_fundamentals(
            tickers,
            max_workers=max_workers,
            batch_size=25,
            delay=0.8
        )

        # Use delete_missing=False to safely upsert and protect against accidental deletions during network hiccups
        saved_count = save_fundamentals(fetched_fundamentals, db_path=db_path, delete_missing=False)
        print(f"✓ Atomically upserted {saved_count}/{len(tickers)} stock records in SQLite.")

        # Reload complete synchronized fundamental records from SQLite
        fundamentals = load_fundamentals(db_path=db_path)
    else:
        print(f"\n[Step 2/3] 📂 Skipping fundamentals fetch. Loading {existing_stock_count} stocks from SQLite.")
        fundamentals = load_fundamentals(db_path=db_path)
        if not tickers:
            tickers = [s["ticker"] for s in fundamentals if s.get("ticker")]

    # Step 3: News Harvesting & ChromaDB Vector Indexing
    news_data: Dict[str, List[Dict[str, Any]]] = {}
    new_docs_indexed = 0
    chroma_before = get_chroma_document_count(chroma_dir=chroma_dir, collection_name=collection_name)

    if refresh_news and tickers:
        print(f"\n[Step 3/3] 📰 Harvesting latest news headlines & catalysts for {len(tickers)} tickers...")
        news_data = batch_fetch_news(
            tickers=tickers,
            max_workers=max_workers,
            max_news_per_ticker=max_news_per_ticker
        )
        total_fetched_news = sum(len(v) for v in news_data.values())
        print(f"✓ Harvested {total_fetched_news} articles across {len(news_data)} tickers.")

        print("\n[Indexing] 🔄 Indexing deduplicated news vectors into ChromaDB...")
        index_all_data(
            fundamentals_data=fundamentals,
            news_data=news_data,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            db_path=db_path
        )
        chroma_after = get_chroma_document_count(chroma_dir=chroma_dir, collection_name=collection_name)
        new_docs_indexed = max(0, chroma_after - chroma_before)
        print(f"✓ ChromaDB indexing complete: +{new_docs_indexed} new articles added (Total in DB: {chroma_after}).")
    else:
        print("\n[Step 3/3] ⏭️ Skipping news harvesting (refresh_news=False).")
        chroma_after = chroma_before

    # Clear cached collection handle if news_tools module is loaded in memory
    try:
        import rag.tools.news_tools as nt
        nt._CACHED_CHROMA_COLLECTION = None
    except Exception:
        pass

    elapsed = round(time.time() - start_time, 2)
    total_db_stocks = count_fundamentals(db_path=db_path)
    total_fetched_news = sum(len(v) for v in news_data.values())

    print("\n" + "=" * 70)
    print("✅ Financial RAG Daily Pipeline Run Finished Successfully!")
    print(f"   • Completed At            : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   • Total Elapsed Time      : {elapsed} seconds")
    print(f"   • Stocks in SQLite DB     : {total_db_stocks}")
    print(f"   • News Articles Fetched   : {total_fetched_news}")
    print(f"   • New Chroma Docs Indexed : +{new_docs_indexed}")
    print(f"   • Total Chroma Documents  : {chroma_after}")
    print("=" * 70)

    return {
        "status": "success",
        "timestamp": now_str,
        "elapsed_seconds": elapsed,
        "stocks_in_sqlite": total_db_stocks,
        "tickers_count": len(tickers),
        "news_articles_fetched": total_fetched_news,
        "new_articles_indexed": new_docs_indexed,
        "total_chroma_documents": chroma_after,
        "sqlite_db": db_path,
        "chroma_dir": chroma_dir,
    }


def main():
    """CLI entry point for daily cron jobs, Windows Task Scheduler, and automated scripts."""
    parser = argparse.ArgumentParser(
        description="Financial RAG Daily Automation Pipeline (Fundamentals & News Updater)"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="nasdaq100",
        help="Constituent stock index (default: 'nasdaq100', e.g. 'sp500')"
    )
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="Skip re-fetching fundamentals from yfinance"
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Skip news harvesting"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Max concurrent worker threads for network requests (default: 6)"
    )
    parser.add_argument(
        "--max-news",
        type=int,
        default=10,
        help="Maximum news items to harvest per ticker (default: 10)"
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            refresh_fundamentals=not args.no_fundamentals,
            refresh_news=not args.no_news,
            index_type=args.index,
            max_workers=args.workers,
            max_news_per_ticker=args.max_news,
        )
        if result.get("status") != "success":
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ [Daily Pipeline Error] Execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
