"""
get_data.py - Pure In-Memory Data Acquisition & Financial Ratio Calculations

Handles network fetching and calculations for:
1. Slickcharts constituent tickers (NASDAQ-100, S&P 500)
2. yfinance financial statement parsing & ratio calculations (ROIC, EBITDA, CAGR, Margins)
3. yfinance market news extraction, parsing, and title-based deterministic UUIDs

Pure in-memory operations with zero disk or JSON persistence coupling.
"""

import sys
import time
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import yfinance as yf

# Self-reference so 'from get_data import get_data' works seamlessly when running inside get_data/
get_data = sys.modules[__name__]


# Standardized fundamental metric keys used across the application
FUNDAMENTAL_METRIC_KEYS = (
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

__all__ = [
    "FUNDAMENTAL_METRIC_KEYS",
    "convert_nan_to_none",
    "load_json",
    "save_json",
    "get_slickcharts_ticker",
    "fundamental_analysis",
    "batch_fetch_fundamentals",
    "get_news_id",
    "get_ticker_news",
    "batch_fetch_news",
]


# ==========================================
# 0. Data Cleaning & Utility Helpers
# ==========================================

def load_json(filepath: str, default: Any = None) -> Any:
    """Safe helper to load JSON files."""
    import os, json
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(filepath: str, data: Any, indent: int = 4) -> None:
    """Safe helper to save data to JSON with directory creation."""
    import os, json
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def convert_nan_to_none(obj: Any) -> Any:
    """
    Recursively converts NaN float values and numpy/pandas scalars to clean Python primitives for JSON/DB.
    """
    if isinstance(obj, dict):
        return {key: convert_nan_to_none(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


# ==========================================
# 1. Ticker Scraping via Slickcharts
# ==========================================

def get_slickcharts_ticker(
    index_type: str = "nasdaq100", 
    limit: int = 100
) -> List[str]:
    """
    Scrapes constituent tickers from Slickcharts for a given index (e.g. 'nasdaq100', 'sp500').
    Returns tickers strictly in-memory.
    """
    url = f"https://www.slickcharts.com/{index_type}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/139.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    response = session.get(url, headers=headers, timeout=20)
    if response.status_code == 403:
        raise PermissionError(f"Slickcharts blocked request (403 Forbidden): {url}")

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    links = soup.select("div.col-lg-7 table tbody tr td:nth-of-type(3) a")[:limit]
    tickers = [link.get_text(strip=True).replace(".", "-") for link in links]

    if not tickers:
        raise ValueError(f"No tickers found on {url}")

    print(f"[GetData] Retrieved {len(tickers)} tickers for {index_type}.")
    return tickers


# ==========================================
# 2. Financial Ratio & Math Calculation Helpers
# ==========================================

def _safe_val(df: Optional[pd.DataFrame], row_name: str, col: int = 0) -> float:
    """Safely extracts a float value from a yfinance financial statement DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan
    try:
        if row_name in df.index and len(df.columns) > col:
            val = df.loc[row_name].iloc[col]
            return float(val) if pd.notna(val) else np.nan
    except Exception:
        pass
    return np.nan


def _safe_div(
    numerator: Optional[float], 
    denominator: Optional[float], 
    require_positive_denom: bool = True
) -> float:
    """Safely computes division, guarding against zero, negative (if required), and NaN values."""
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    if require_positive_denom and denominator <= 0:
        return np.nan
    if not require_positive_denom and denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _safe_growth(curr: Optional[float], prev: Optional[float]) -> float:
    """Computes 1-year growth rate: (curr / prev) - 1."""
    if pd.notna(curr) and pd.notna(prev) and prev > 0:
        return float(curr / prev - 1.0)
    return np.nan


def _safe_cagr(end_val: Optional[float], start_val: Optional[float], years: int = 3) -> float:
    """Computes Compound Annual Growth Rate: (end_val / start_val) ** (1 / years) - 1."""
    if pd.notna(end_val) and pd.notna(start_val) and end_val > 0 and start_val > 0 and years > 0:
        return float((end_val / start_val) ** (1.0 / years) - 1.0)
    return np.nan


def _calc_fcf(ocf: Optional[float], capex: Optional[float]) -> float:
    """
    Computes Free Cash Flow (FCF = OCF + Capex).
    Note: yfinance reports Capex as a negative number in cash flows.
    """
    if pd.notna(ocf) and pd.notna(capex):
        return float(ocf + capex)
    return np.nan


# ==========================================
# 3. Fundamental Metrics Analysis via yfinance
# ==========================================

def fundamental_analysis(ticker_symbol: str) -> Optional[Dict[str, Any]]:
    """
    Pulls fundamental financial metrics for a given ticker symbol using yfinance and computes key ratios:
    P/E, ROIC, ROE, Margins, Growth Rates, and Debt-to-EBITDA.
    """
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info or {}
        inc = t.income_stmt
        bs = t.balance_sheet
        cf = t.cashflow

        # Income statement
        revenue = _safe_val(inc, "Total Revenue", 0)
        revenue_1y = _safe_val(inc, "Total Revenue", 1)
        revenue_3y = _safe_val(inc, "Total Revenue", 3)

        gross_profit = _safe_val(inc, "Gross Profit", 0)
        operating_income = _safe_val(inc, "Operating Income", 0)
        operating_income_1y = _safe_val(inc, "Operating Income", 1)
        net_income = _safe_val(inc, "Net Income", 0)
        ebitda = _safe_val(inc, "EBITDA", 0)

        # EPS
        eps = _safe_val(inc, "Diluted EPS", 0)
        eps_1y = _safe_val(inc, "Diluted EPS", 1)
        eps_3y = _safe_val(inc, "Diluted EPS", 3)

        # Balance sheet
        equity = _safe_val(bs, "Stockholders Equity", 0)
        debt = _safe_val(bs, "Total Debt", 0)
        cash = _safe_val(bs, "Cash Cash Equivalents And Short Term Investments", 0)

        # Cash flow
        ocf = _safe_val(cf, "Operating Cash Flow", 0)
        capex = _safe_val(cf, "Capital Expenditure", 0)
        fcf = _calc_fcf(ocf, capex)

        ocf_1y = _safe_val(cf, "Operating Cash Flow", 1)
        capex_1y = _safe_val(cf, "Capital Expenditure", 1)
        fcf_1y = _calc_fcf(ocf_1y, capex_1y)

        # ROIC calculation
        tax = _safe_val(inc, "Tax Provision", 0)
        pretax = _safe_val(inc, "Pretax Income", 0)
        tax_rate = np.clip(tax / pretax if pd.notna(tax) and pd.notna(pretax) and pretax > 0 else 0.21, 0, 1)
        nopat = operating_income * (1 - tax_rate) if pd.notna(operating_income) else np.nan
        invested_capital = (equity if pd.notna(equity) else 0) + (debt if pd.notna(debt) else 0) - (cash if pd.notna(cash) else 0)
        roic = _safe_div(nopat, invested_capital, require_positive_denom=True)

        # Margins & profitability
        gross_margin = _safe_div(gross_profit, revenue)
        operating_margin = _safe_div(operating_income, revenue)
        net_margin = _safe_div(net_income, revenue)
        op_margin_1y = _safe_div(operating_income_1y, revenue_1y)
        op_margin_trend = (operating_margin - op_margin_1y) if pd.notna(operating_margin) and pd.notna(op_margin_1y) else np.nan

        roe = _safe_div(net_income, equity)
        fcf_to_ni = _safe_div(fcf, net_income)
        fcf_margin = _safe_div(fcf, revenue)
        net_debt = (debt - cash) if pd.notna(debt) and pd.notna(cash) else np.nan
        net_debt_ebitda = _safe_div(net_debt, ebitda)

        return {
            "ticker": ticker_symbol,
            "company_name": info.get("longName") or info.get("shortName") or ticker_symbol,
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "fundamentals": {
                "pe_ratio": info.get("trailingPE"),
                "revenue_growth_1y": _safe_growth(revenue, revenue_1y),
                "revenue_cagr_3y": _safe_cagr(revenue, revenue_3y),
                "eps_growth_1y": _safe_growth(eps, eps_1y),
                "eps_cagr_3y": _safe_cagr(eps, eps_3y),
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "net_margin": net_margin,
                "operating_margin_trend": op_margin_trend,
                "roe": roe,
                "roic": roic,
                "fcf_to_net_income": fcf_to_ni,
                "fcf_margin": fcf_margin,
                "fcf_growth_1y": _safe_growth(fcf, fcf_1y),
                "net_debt_to_ebitda": net_debt_ebitda,
            },
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"[GetData] {ticker_symbol} Fundamental Analysis Failed: {e}")
        return None


def batch_fetch_fundamentals(
    tickers: List[str], 
    max_workers: int = 6, 
    batch_size: int = 25, 
    delay: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Concurrently fetches fundamental metrics for a list of tickers in batches.
    Returns clean results in memory (List[Dict]).
    """
    results = []
    total = len(tickers)

    for start in range(0, total, batch_size):
        batch = tickers[start:start + batch_size]
        print(f"[GetData] Processing fundamentals {start + 1}-{min(start + batch_size, total)}/{total}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fundamental_analysis, ticker): ticker for ticker in batch}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                        print(f"  [Fetched] {ticker} ({res.get('company_name', ticker)})")
                except Exception as e:
                    print(f"[GetData] Error processing {ticker}: {e}")

        if start + batch_size < total and delay > 0:
            time.sleep(delay)

    return convert_nan_to_none(results)


# ==========================================
# 4. News Harvesting & Parsing via yfinance
# ==========================================

def _parse_news_item(item: Dict[str, Any], ticker_symbol: str) -> Optional[Dict[str, Any]]:
    """
    Standardizes a single yfinance news item into the pipeline schema.
    If yfinance provides no UUID, generates a deterministic UUID from the title.
    """
    if not isinstance(item, dict):
        return None
    content = item.get("content", {}) if isinstance(item.get("content"), dict) else item

    title = content.get("title") or item.get("title") or ""
    title = title.strip()
    if not title:
        return None

    # Extract unique UUID from yfinance
    news_uuid = (
        content.get("id") or 
        item.get("id") or 
        content.get("uuid") or 
        item.get("uuid") or 
        ""
    )

    # Deterministic fallback: Use article title if yfinance has no UUID
    if not news_uuid:
        seed = title
        news_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    else:
        news_uuid = str(news_uuid).strip()

    publisher = (
        content.get("provider", {}).get("displayName")
        if isinstance(content.get("provider"), dict)
        else item.get("publisher") or "Unknown"
    )
    pub_time = content.get("pubDate") or item.get("providerPublishTime") or ""
    summary = content.get("summary") or content.get("description") or item.get("summary") or ""
    link = (
        content.get("canonicalUrl", {}).get("url")
        if isinstance(content.get("canonicalUrl"), dict)
        else item.get("link") or ""
    )

    return {
        "yf_uuid": news_uuid,
        "doc_id": news_uuid,
        "ticker": ticker_symbol,
        "title": title,
        "publisher": publisher,
        "publish_time": str(pub_time),
        "summary": summary.strip(),
        "link": link
    }


def get_news_id(item: Dict[str, Any]) -> str:
    """Return the canonical news doc_id / yf_uuid across schemas."""
    return str(
        item.get("doc_id")
        or item.get("yf_uuid")
        or item.get("uuid")
        or item.get("id")
        or ""
    ).strip()


def get_ticker_news(ticker_symbol: str, max_news: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches latest news items with UUIDs for a ticker using yfinance.
    Falls back to yf.Search if t.news is empty (e.g. Yahoo Finance endpoint changes).
    Returns parsed list of articles in memory.
    """
    try:
        t = yf.Ticker(ticker_symbol)
        raw_news = t.news or []

        # Fallback to yf.Search if t.news is empty
        if not raw_news:
            search = yf.Search(ticker_symbol, news_count=max_news)
            raw_news = search.news or []

        parsed_news = []
        for item in raw_news[:max_news]:
            parsed = _parse_news_item(item, ticker_symbol)
            if parsed:
                parsed_news.append(parsed)

        return parsed_news

    except Exception as e:
        print(f"[GetData] {ticker_symbol} News Fetch Failed: {e}")
        return []


def batch_fetch_news(
    tickers: List[str], 
    max_workers: int = 6, 
    delay: float = 0.0,
    max_news_per_ticker: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Concurrently fetches news with unique UUIDs for a list of tickers (pure in-memory).
    Returns mapping of ticker -> List[Dict].
    """
    all_news: Dict[str, List[Dict[str, Any]]] = {}
    print(f"[GetData] Fetching news for {len(tickers)} tickers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_ticker_news, t, max_news_per_ticker): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                news = future.result()
                all_news[ticker] = news
                print(f"  [News Fetched] {ticker} ({len(news)} articles)")
            except Exception as e:
                print(f"[GetData] Error fetching news for {ticker}: {e}")
                all_news[ticker] = []

    return all_news


# Sibling module exposure so 'from get_data import get_data, sql_handler, chroma_handler, data_pipeline'
# works seamlessly regardless of whether Python resolves get_data as the root package or this module
try:
    from . import sql_handler, chroma_handler, data_pipeline
except (ImportError, ValueError):
    try:
        import sql_handler, chroma_handler, data_pipeline
    except ImportError:
        pass
