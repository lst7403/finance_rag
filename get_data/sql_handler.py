"""
sql_handler.py - Dedicated SQLite Database Layer for Stock Fundamentals

Provides high-performance local SQL persistence, indexing, atomic upserts,
and querying for financial fundamental metrics in SQLite (data/database/foundamental_sql.db).
Zero JSON storage or fallback dependencies.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator

try:
    from config import CONFIG
    DB_PATH = CONFIG.fundamentals_db
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import CONFIG
    DB_PATH = CONFIG.fundamentals_db


FUNDAMENTAL_METRIC_COLS = (
    "pe_ratio",
    "revenue_growth_1y",
    "revenue_cagr_3y",
    "eps_growth_1y",
    "eps_cagr_3y",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "operating_margin_trend",
    "roe",
    "roic",
    "fcf_to_net_income",
    "fcf_margin",
    "fcf_growth_1y",
    "net_debt_to_ebitda",
)

ALL_COLS = (
    "ticker",
    "company_name",
    "sector",
    "industry",
    *FUNDAMENTAL_METRIC_COLS,
    "last_updated",
)

FUNDAMENTAL_METRIC_KEYS = FUNDAMENTAL_METRIC_COLS

__all__ = [
    "DB_PATH",
    "FUNDAMENTAL_METRIC_COLS",
    "ALL_COLS",
    "FUNDAMENTAL_METRIC_KEYS",
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
]


@contextmanager
def get_connection(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Creates a connection context to the SQLite database with Row row_factory, automatically closing upon exit."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes SQLite schema and creates performance indexes."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_fundamentals (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                pe_ratio REAL,
                revenue_growth_1y REAL,
                revenue_cagr_3y REAL,
                eps_growth_1y REAL,
                eps_cagr_3y REAL,
                gross_margin REAL,
                operating_margin REAL,
                net_margin REAL,
                operating_margin_trend REAL,
                roe REAL,
                roic REAL,
                fcf_to_net_income REAL,
                fcf_margin REAL,
                fcf_growth_1y REAL,
                net_debt_to_ebitda REAL,
                last_updated TEXT
            )
        """)
        # Indexes for fast querying & screening
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_sector ON stock_fundamentals (sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_roic ON stock_fundamentals (roic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_pe ON stock_fundamentals (pe_ratio)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_rev_growth ON stock_fundamentals (revenue_growth_1y)")
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a SQLite row into the standardized nested dictionary format."""
    r = dict(row)
    funds = {col: r.get(col) for col in FUNDAMENTAL_METRIC_COLS}
    return {
        "ticker": r.get("ticker"),
        "company_name": r.get("company_name") or r.get("ticker"),
        "sector": r.get("sector") or "N/A",
        "industry": r.get("industry") or "N/A",
        "fundamentals": funds,
        "last_updated": r.get("last_updated"),
    }


def upsert_fundamentals(
    stocks_data: List[Dict[str, Any]], 
    db_path: str = DB_PATH,
    delete_missing: bool = True
) -> int:
    """
    Inserts or updates fundamental metrics for multiple stock records in SQLite.
    Supports both nested ('fundamentals': {...}) and flattened dictionary structures.
    
    If delete_missing is True and stocks_data contains valid records, any tickers
    currently in SQLite that are NOT present in stocks_data will be deleted from the database.
    """
    if not stocks_data:
        return 0

    init_db(db_path)
    records = []
    incoming_tickers = set()

    for item in stocks_data:
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        incoming_tickers.add(ticker)
        company_name = item.get("company_name") or ticker
        sector = item.get("sector") or "N/A"
        industry = item.get("industry") or "N/A"
        last_updated = item.get("last_updated") or ""

        raw_funds = item.get("fundamentals") if isinstance(item.get("fundamentals"), dict) else item
        
        row_vals = [
            ticker,
            company_name,
            sector,
            industry,
        ]

        for col in FUNDAMENTAL_METRIC_COLS:
            val = raw_funds.get(col)
            if val is not None:
                try:
                    import math
                    if isinstance(val, (float, int)) and math.isnan(val):
                        val = None
                    else:
                        val = float(val)
                except (TypeError, ValueError):
                    val = None
            row_vals.append(val)

        row_vals.append(last_updated)
        records.append(tuple(row_vals))

    if not records:
        return 0

    placeholders = ", ".join(["?"] * len(ALL_COLS))
    update_clause = ", ".join([f"{col} = excluded.{col}" for col in ALL_COLS if col != "ticker"])
    sql = f"""
        INSERT INTO stock_fundamentals ({", ".join(ALL_COLS)})
        VALUES ({placeholders})
        ON CONFLICT(ticker) DO UPDATE SET
        {update_clause}
    """

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # 1. Atomic upsert for all incoming records
        cursor.executemany(sql, records)

        # 2. Synchronize: Delete any tickers from DB that are not in the new dataset
        if delete_missing and incoming_tickers:
            cursor.execute("SELECT ticker FROM stock_fundamentals")
            existing_tickers = {row[0] for row in cursor.fetchall()}
            obsolete_tickers = existing_tickers - incoming_tickers
            if obsolete_tickers:
                cursor.executemany(
                    "DELETE FROM stock_fundamentals WHERE ticker = ?",
                    [(t,) for t in obsolete_tickers]
                )
                print(f"[SQLHandler] Deleted {len(obsolete_tickers)} obsolete ticker(s) not in current dataset: {sorted(obsolete_tickers)}")

        conn.commit()

    return len(records)


def delete_stock_by_ticker(ticker: str, db_path: str = DB_PATH) -> bool:
    """Deletes a single stock profile by its ticker from SQLite database."""
    if not ticker:
        return False
    init_db(db_path)
    clean_ticker = ticker.strip().upper()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM stock_fundamentals WHERE ticker = ?", (clean_ticker,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            print(f"[SQLHandler] Deleted stock '{clean_ticker}' from SQLite ({db_path})")
        return deleted


def delete_stocks(tickers: List[str], db_path: str = DB_PATH) -> int:
    """Deletes multiple stock profiles by their tickers from SQLite database."""
    if not tickers:
        return 0
    init_db(db_path)
    clean_tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not clean_tickers:
        return 0
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("DELETE FROM stock_fundamentals WHERE ticker = ?", [(t,) for t in clean_tickers])
        conn.commit()
        deleted_count = cursor.rowcount
        print(f"[SQLHandler] Deleted {deleted_count} stock(s) from SQLite ({db_path})")
        return deleted_count


def save_fundamentals(
    data: List[Dict[str, Any]], 
    db_path: str = DB_PATH,
    filepath: Optional[str] = None,
    delete_missing: bool = True,
    **kwargs
) -> int:
    """
    Persists stock fundamental metrics exclusively into the local SQLite database.
    If delete_missing is True, any tickers currently in SQLite but not in 'data' are deleted.
    (Any filepath argument is checked: if ending in .db it overrides db_path; JSON is ignored).
    """
    target_db = filepath if (filepath and filepath.endswith(".db")) else db_path
    saved_count = upsert_fundamentals(data, db_path=target_db, delete_missing=delete_missing)
    print(f"[SQLHandler] Saved fundamental data for {saved_count} stocks to SQLite database ({target_db})")
    return saved_count


def fetch_all_fundamentals(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all stock fundamental profiles from the SQLite database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stock_fundamentals ORDER BY ticker ASC")
        rows = cursor.fetchall()
        return [_row_to_dict(row) for row in rows]


def load_fundamentals(
    db_path: str = DB_PATH,
    filepath: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Loads stock fundamental data exclusively from SQLite database.
    """
    target_db = filepath if (filepath and filepath.endswith(".db")) else db_path
    return fetch_all_fundamentals(target_db)


def fetch_stock_by_ticker(ticker: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a single stock fundamental profile by its ticker from SQLite."""
    if not ticker:
        return None
    init_db(db_path)
    clean_ticker = ticker.strip().upper()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stock_fundamentals WHERE ticker = ? LIMIT 1", (clean_ticker,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None


def count_fundamentals(db_path: str = DB_PATH) -> int:
    """Returns the total number of stock records in the SQLite database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_fundamentals")
        return cursor.fetchone()[0]
