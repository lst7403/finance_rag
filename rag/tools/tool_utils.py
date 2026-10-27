"""
tool_utils.py - Shared Utilities for Multi-Agent Financial Tools

Provides:
1. run_async_safe: Centralized safe async runner for agent workflows (prevents event loop deadlocks).
2. resolve_ticker: Dynamically resolves company names or tickers to official ticker symbols via SQLite.
3. resolve_ticker_list: Normalizes list or comma-separated strings of tickers/company names.
4. normalize_ratio: Fault tolerance for LLM percentage values (e.g., converts 15 or 15% to 0.15).
"""

import asyncio
import concurrent.futures
import functools
import inspect
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union

from config import CONFIG


# =========================================================================
# 1. Safe Async Runner for Multi-Agent Workflows
# =========================================================================

_GLOBAL_AGENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
_GLOBAL_AGENT_THREAD: Optional[threading.Thread] = None
_GLOBAL_AGENT_LOCK = threading.Lock()


def get_agent_event_loop() -> asyncio.AbstractEventLoop:
    """
    Returns or lazily creates a persistent background event loop on a dedicated thread.
    This ensures that async clients (e.g. Ollama AsyncClient, HTTP connection pools)
    remain permanently valid and bound to an active loop across multi-turn conversations,
    completely preventing 'Event loop is closed' errors.
    """
    global _GLOBAL_AGENT_LOOP, _GLOBAL_AGENT_THREAD
    with _GLOBAL_AGENT_LOCK:
        if _GLOBAL_AGENT_LOOP is None or _GLOBAL_AGENT_LOOP.is_closed():
            _GLOBAL_AGENT_LOOP = asyncio.new_event_loop()

            def _loop_runner(loop: asyncio.AbstractEventLoop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            _GLOBAL_AGENT_THREAD = threading.Thread(
                target=_loop_runner,
                args=(_GLOBAL_AGENT_LOOP,),
                name="AgentEventLoopThread",
                daemon=True
            )
            _GLOBAL_AGENT_THREAD.start()
        return _GLOBAL_AGENT_LOOP


def run_async_safe(coro_or_fn: Any, timeout: Optional[float] = None) -> Any:
    """
    Executes an async coroutine, workflow handler, or async callable safely
    on the persistent background agent event loop.
    Supports multi-turn execution without closing the loop or invalidating HTTP clients.
    """
    loop = get_agent_event_loop()

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if current_loop is loop:
        res = coro_or_fn() if callable(coro_or_fn) else coro_or_fn
        if inspect.isawaitable(res):
            raise RuntimeError("Cannot invoke run_async_safe synchronously from within the agent loop itself.")
        return res

    async def _coro_runner():
        res = coro_or_fn() if callable(coro_or_fn) else coro_or_fn
        if inspect.isawaitable(res):
            return await res
        return res

    fut = asyncio.run_coroutine_threadsafe(_coro_runner(), loop)
    return fut.result(timeout=timeout)


# =========================================================================
# 2. Dynamic SQLite Ticker & Company Name Resolver
# =========================================================================

@functools.lru_cache(maxsize=1)
def get_db_ticker_map(db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Dynamically loads all (ticker, company_name) pairs from SQLite into an in-memory lookup map.
    No hardcoded stocks - populated purely from the database table 'stock_fundamentals'.
    """
    target_db = db_path or CONFIG.fundamentals_db
    mapping: Dict[str, str] = {}

    try:
        with sqlite3.connect(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ticker, company_name FROM stock_fundamentals")
            rows = cursor.fetchall()
            for ticker, company_name in rows:
                if not ticker:
                    continue
                t_clean = ticker.strip().upper()
                mapping[t_clean] = t_clean

                if company_name:
                    c_clean = company_name.strip().upper()
                    mapping[c_clean] = t_clean
                    # Strip corporate suffixes for simplified matching (e.g. "APPLE INC." -> "APPLE")
                    for suffix in (" INC.", " CORP.", " CORPORATION", " CO.", " LTD.", " PLC", " LLC"):
                        if c_clean.endswith(suffix):
                            simplified = c_clean[:-len(suffix)].strip()
                            if simplified:
                                mapping[simplified] = t_clean
    except Exception:
        pass

    return mapping



def resolve_ticker(input_val: str, db_path: Optional[str] = None) -> str:
    """
    Dynamically resolves an input string (stock symbol or company name) to the official ticker
    using the SQLite database.
    
    Examples:
        resolve_ticker("NVDA")        -> "NVDA"
        resolve_ticker("Nvidia")      -> "NVDA"
        resolve_ticker("Apple Inc.")  -> "AAPL"
        resolve_ticker("apple")       -> "AAPL"

    Returns:
        str: The official uppercase ticker, or the uppercase input if no match is found.
    """
    if not input_val or not isinstance(input_val, str):
        return ""

    query = input_val.strip().upper()
    mapping = get_db_ticker_map(db_path)

    # 1. Exact match against known tickers or company names
    if query in mapping:
        return mapping[query]

    # 2. Substring matching against company names in database
    for name, ticker in mapping.items():
        if query in name or name in query:
            return ticker

    # 3. Direct SQL fallback with LIKE in case database was updated without cache clear
    try:
        target_db = db_path or CONFIG.fundamentals_db
        with sqlite3.connect(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ticker FROM stock_fundamentals 
                WHERE UPPER(ticker) = UPPER(?) 
                   OR UPPER(company_name) LIKE '%' || UPPER(?) || '%' 
                ORDER BY LENGTH(company_name) ASC 
                LIMIT 1
                """,
                (query, query)
            )
            row = cursor.fetchone()
            if row and row[0]:
                return row[0].strip().upper()
    except Exception:
        pass

    return query


def resolve_ticker_list(
    tickers: Union[List[str], str],
    db_path: Optional[str] = None
) -> List[str]:
    """
    Normalizes and resolves a list or comma-separated string of tickers/company names.
    Removes duplicates while preserving order.

    Examples:
        resolve_ticker_list("Apple, Nvidia, MSFT") -> ["AAPL", "NVDA", "MSFT"]
        resolve_ticker_list(["Google", "AMD"])      -> ["GOOGL", "AMD"]
    """
    if isinstance(tickers, str):
        raw_items = [t.strip() for t in tickers.replace(";", ",").split(",") if t.strip()]
    elif isinstance(tickers, list):
        raw_items = [str(t).strip() for t in tickers if str(t).strip()]
    else:
        return []

    resolved_list = []
    seen = set()
    for item in raw_items:
        t = resolve_ticker(item, db_path=db_path)
        if t and t not in seen:
            seen.add(t)
            resolved_list.append(t)

    return resolved_list


# =========================================================================
# 3. Ratio Normalizer (Percentage to Decimal)
# =========================================================================

def normalize_ratio(val: Optional[Union[float, int]]) -> Optional[float]:
    """
    LLM fault tolerance: If the model mistakenly passes an integer percentage
    (e.g., 15 or 25 representing 15% or 25%), automatically normalize it to a decimal (0.15, 0.25).
    """
    if val is None:
        return None
    try:
        f_val = float(val)
        return f_val / 100.0 if f_val > 1.0 else f_val
    except (ValueError, TypeError):
        return None
