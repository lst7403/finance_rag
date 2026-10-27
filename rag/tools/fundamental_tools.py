"""
fundamental_tools.py - Raw Fundamental Data Tools for Financial RAG

Provides direct deterministic access to raw quantitative financial metrics 
(P/E, ROIC, ROE, Margins, Growth Rates, Debt Ratios) loaded from the SQLite database,
wrapped as LlamaIndex FunctionTools for autonomous LLM agent tool-calling.
"""

import json
from typing import Any, Dict, List, Optional, Union
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent

from config import CONFIG
from get_data import load_fundamentals
from .tool_utils import (
    normalize_ratio,
    resolve_ticker,
    resolve_ticker_list,
    run_async_safe,
)

__all__ = [
    "compare_stocks_fundamentals",
    "create_fundamental_agent",
    "create_fundamental_agent_tool",
    "get_fundamental_tools",
    "get_stock_fundamentals",
    "list_available_tickers",
    "screen_stocks_by_fundamentals",
]

# In-memory data cache to prevent repeated disk reads on each tool call
_CACHED_FUNDAMENTALS: Optional[List[Dict[str, Any]]] = None
_CACHED_BY_TICKER: Optional[Dict[str, Dict[str, Any]]] = None
_CACHED_FILE_PATH: Optional[str] = None


def _get_all_fundamentals() -> List[Dict[str, Any]]:
    """Fetches fundamental data and caches it in-memory as a singleton."""
    global _CACHED_FUNDAMENTALS, _CACHED_BY_TICKER, _CACHED_FILE_PATH
    db_path = CONFIG.fundamentals_db
    if _CACHED_FUNDAMENTALS is None or _CACHED_FILE_PATH != db_path:
        _CACHED_FUNDAMENTALS = load_fundamentals(db_path=db_path)
        _CACHED_BY_TICKER = {
            item.get("ticker", "").strip().upper(): item
            for item in _CACHED_FUNDAMENTALS
            if item.get("ticker")
        }
        _CACHED_FILE_PATH = db_path
    return _CACHED_FUNDAMENTALS


def _get_fundamentals_by_ticker() -> Dict[str, Dict[str, Any]]:
    """Returns an in-memory dictionary mapping uppercase ticker to its fundamental profile for O(1) lookups."""
    _get_all_fundamentals()
    return _CACHED_BY_TICKER or {}


def get_stock_fundamentals(ticker: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Retrieves exact raw quantitative fundamental financial metrics for one or more stock tickers.
    Supports single tickers (e.g. 'NVDA'), company names ('Nvidia'), comma-separated strings ('NVDA, AAPL'),
    or lists of tickers (['NVDA', 'AAPL']).
    Use this tool whenever you need exact numerical values for valuation (P/E ratio),
    profitability (gross margin, operating margin, net margin), return on capital (ROE, ROIC),
    growth rates (1Y and 3Y revenue and EPS growth), cash generation (FCF margin, FCF growth),
    or balance sheet solvency (Net Debt to EBITDA).

    Args:
        ticker (Union[str, List[str]]): Stock ticker symbol(s) or company name(s).

    Returns:
        Dict[str, Any]: Company profile and raw fundamental metrics, or comparison dictionary for multiple tickers.
    """
    if not ticker:
        return {"status": "error", "error": "Ticker symbol is required."}

    ticker_list = resolve_ticker_list(ticker)
    if not ticker_list:
        return {"status": "error", "error": f"Could not resolve any valid tickers from input: {ticker}"}

    by_ticker = _get_fundamentals_by_ticker()

    # Single stock query
    if len(ticker_list) == 1:
        ticker_clean = ticker_list[0]
        if ticker_clean in by_ticker:
            item = by_ticker[ticker_clean]
            return {
                "status": "success",
                "ticker": item.get("ticker"),
                "company_name": item.get("company_name"),
                "sector": item.get("sector"),
                "industry": item.get("industry"),
                "fundamentals": item.get("fundamentals", {}),
                "last_updated": item.get("last_updated"),
            }
        available = list(by_ticker.keys())
        return {
            "status": "not_found",
            "error": f"Stock ticker '{ticker_clean}' not found in the fundamentals database.",
            "sample_available_tickers": available[:15],
            "total_available": len(available),
        }

    # Multi-stock query
    comparison = {}
    missing = []
    for t_clean in ticker_list:
        if t_clean in by_ticker:
            item = by_ticker[t_clean]
            comparison[t_clean] = {
                "company_name": item.get("company_name"),
                "sector": item.get("sector"),
                "industry": item.get("industry"),
                "fundamentals": item.get("fundamentals", {}),
            }
        else:
            missing.append(t_clean)

    return {
        "status": "success" if comparison else "not_found",
        "compared_stocks": comparison,
        "missing_tickers": missing,
    }


def screen_stocks_by_fundamentals(
    min_roic: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_gross_margin: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    max_debt_to_ebitda: Optional[float] = None,
    min_roe: Optional[float] = None,
    sector: Optional[str] = None,
    sort_by: str = "roic",
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Screens and filters stocks deterministically across the entire fundamentals database based on 
    numerical quantitative criteria (e.g., minimum ROIC, maximum P/E ratio, minimum gross margin, 
    minimum revenue growth, maximum Net Debt / EBITDA ratio, and sector).

    Args:
        min_roic (Optional[float]): Minimum Return on Invested Capital (e.g., 0.20 or 20 for 20%).
        max_pe (Optional[float]): Maximum Price-to-Earnings ratio (e.g., 35.0).
        min_gross_margin (Optional[float]): Minimum gross profit margin (e.g., 0.40 or 40 for 40%).
        min_revenue_growth (Optional[float]): Minimum 1-year revenue growth rate (e.g., 0.10 or 10 for 10%).
        max_debt_to_ebitda (Optional[float]): Maximum Net Debt to EBITDA ratio (e.g., 2.0).
        min_roe (Optional[float]): Minimum Return on Equity (e.g., 0.15 or 15 for 15%).
        sector (Optional[str]): Target sector filter (e.g., 'Technology', 'Healthcare', 'Financial Services').
        sort_by (str): Metric to sort by: 'roic', 'roe', 'gross_margin', 'revenue_growth_1y', 'pe_ratio', 'net_debt_to_ebitda'.
        top_n (int): Maximum number of top matching stocks to return. Defaults to 10.

    Returns:
        Dict[str, Any]: Filtered matching stocks sorted by the specified criteria with raw metrics.
    """
    # Automatically normalize percentage/ratio arguments via shared utility
    norm_min_roic = normalize_ratio(min_roic)
    norm_min_gm = normalize_ratio(min_gross_margin)
    norm_min_rev = normalize_ratio(min_revenue_growth)
    norm_min_roe = normalize_ratio(min_roe)

    stocks = _get_all_fundamentals()
    matches = []

    for stock in stocks:
        funds = stock.get("fundamentals", {})
        if not funds:
            continue

        # Sector filter
        if sector:
            stock_sector = str(stock.get("sector", "")).lower()
            if sector.strip().lower() not in stock_sector:
                continue

        # ROIC
        if norm_min_roic is not None:
            roic = funds.get("roic")
            if roic is None or roic < norm_min_roic:
                continue

        # P/E ratio (P/E <= 0 typically indicates loss-making companies, excluded from standard P/E screening)
        if max_pe is not None:
            pe = funds.get("pe_ratio")
            if pe is None or pe <= 0 or pe > max_pe:
                continue

        # Gross margin
        if norm_min_gm is not None:
            gm = funds.get("gross_margin")
            if gm is None or gm < norm_min_gm:
                continue

        # Revenue growth rate
        if norm_min_rev is not None:
            rev_growth = funds.get("revenue_growth_1y")
            if rev_growth is None or rev_growth < norm_min_rev:
                continue

        # Net leverage ratio (Net Debt / EBITDA)
        if max_debt_to_ebitda is not None:
            debt_ratio = funds.get("net_debt_to_ebitda")
            if debt_ratio is None or debt_ratio > max_debt_to_ebitda:
                continue

        # ROE
        if norm_min_roe is not None:
            roe = funds.get("roe")
            if roe is None or roe < norm_min_roe:
                continue

        matches.append({
            "ticker": stock.get("ticker"),
            "company_name": stock.get("company_name"),
            "sector": stock.get("sector"),
            "industry": stock.get("industry"),
            "fundamentals": funds,
        })

    # Sorting direction: P/E and Net Debt/EBITDA are better when smaller (ascending); other return/margin metrics are better when larger (descending)
    ascending_metrics = {"pe_ratio", "net_debt_to_ebitda"}
    is_ascending = sort_by.strip().lower() in ascending_metrics

    def sort_key(item: Dict[str, Any]) -> float:
        val = item["fundamentals"].get(sort_by)
        if val is None:
            return float("inf") if is_ascending else float("-inf")
        # If sorting ascending and P/E is negative (representing losses), place at the end of the ranking
        if is_ascending and sort_by == "pe_ratio" and val <= 0:
            return float("inf")
        return float(val)

    matches.sort(key=sort_key, reverse=not is_ascending)
    top_matches = matches[: max(1, top_n)]

    return {
        "status": "success",
        "total_matched": len(matches),
        "matched_count": len(matches),
        "returned_count": len(top_matches),
        "criteria": {
            "min_roic": norm_min_roic,
            "max_pe": max_pe,
            "min_gross_margin": norm_min_gm,
            "min_revenue_growth": norm_min_rev,
            "max_debt_to_ebitda": max_debt_to_ebitda,
            "min_roe": norm_min_roe,
            "sector": sector,
            "sort_by": sort_by,
        },
        "results": top_matches,
    }


def compare_stocks_fundamentals(tickers: Union[List[str], str]) -> Dict[str, Any]:
    """
    Fetches and aligns raw fundamental metrics for multiple stocks side-by-side.
    Delegates directly to get_stock_fundamentals.
    """
    return get_stock_fundamentals(tickers)


def list_available_tickers(
    sector: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Lists available stock tickers in the fundamentals database along with their company names, sectors, and industries.
    Supports filtering by market sector (e.g. 'Technology', 'Healthcare', 'Energy') or searching by keyword.
    Use when checking what companies are tracked, exploring a sector, or discovering peer stocks.

    Args:
        sector (Optional[str]): Filter stocks by market sector (e.g. 'Technology', 'Healthcare', 'Financial Services').
        search (Optional[str]): Keyword to search across company names, industries, or tickers (e.g. 'semiconductor', 'bank').
        limit (int): Maximum number of stocks to return (default: 50, to preserve LLM context).

    Returns:
        Dict[str, Any]: Matching stocks with ticker, company_name, sector, industry, and total counts.
    """
    stocks = _get_all_fundamentals()
    all_sectors = sorted(list(set(s.get("sector") for s in stocks if s.get("sector"))))

    filtered = stocks
    if sector and isinstance(sector, str):
        sec_clean = sector.strip().upper()
        filtered = [
            s for s in filtered
            if s.get("sector") and (sec_clean in s.get("sector", "").upper() or s.get("sector", "").upper() in sec_clean)
        ]

    if search and isinstance(search, str):
        term = search.strip().upper()
        filtered = [
            s for s in filtered
            if (term in s.get("ticker", "").upper() or
                term in s.get("company_name", "").upper() or
                term in s.get("industry", "").upper())
        ]

    results = [
        {
            "ticker": s.get("ticker"),
            "company_name": s.get("company_name"),
            "sector": s.get("sector"),
            "industry": s.get("industry"),
        }
        for s in filtered
        if s.get("ticker")
    ]

    return {
        "status": "success",
        "matched_count": len(results),
        "total_available": len(stocks),
        "applied_filters": {
            "sector": sector,
            "search": search,
        },
        "available_sectors": all_sectors,
        "stocks": results[:max(1, limit)],
    }


# ==========================================
# LlamaIndex FunctionTools Factory
# ==========================================

def get_fundamental_tools() -> List[FunctionTool]:
    """
    Constructs and returns raw fundamental financial tools as LlamaIndex FunctionTools.
    Streamlined to 3 core tools to eliminate agent tool duplication.
    """
    get_stock_fundamentals_tool = FunctionTool.from_defaults(
        fn=get_stock_fundamentals,
        name="get_stock_fundamentals",
        description=(
            "Fetch exact raw quantitative fundamental financial metrics for one or more stock tickers "
            "(e.g. 'NVDA', 'AAPL', or multiple 'NVDA, AMD'). Returns exact numerical metrics: P/E ratio, "
            "gross/operating/net margins, ROE, ROIC, 1Y and 3Y revenue and EPS growth, FCF margin, "
            "and Net Debt to EBITDA. Supports single stocks or side-by-side multi-stock comparison."
        ),
    )

    screen_stocks_tool = FunctionTool.from_defaults(
        fn=screen_stocks_by_fundamentals,
        name="screen_stocks_by_fundamentals",
        description=(
            "Screen and filter all stocks in the database based on quantitative financial criteria such as min_roic (e.g. 0.20), "
            "max_pe (e.g. 35.0), min_gross_margin (e.g. 0.40), min_revenue_growth (e.g. 0.15), max_debt_to_ebitda (e.g. 2.0), "
            "and sector. Returns top matching stocks with exact raw metrics."
        ),
    )

    list_tickers_tool = FunctionTool.from_defaults(
        fn=list_available_tickers,
        name="list_available_tickers",
        description=(
            "List stock tickers available in the database with company names, sectors, and industries. "
            "Accepts optional sector filter (e.g. 'Technology', 'Healthcare') and keyword search (e.g. 'semiconductor'). "
            "Use when checking what companies are tracked, exploring a sector, or finding available peer stocks."
        ),
    )

    return [
        get_stock_fundamentals_tool,
        screen_stocks_tool,
        list_tickers_tool,
    ]


# ==========================================
# Fundamental Analyst Agent & Supervisor Tool
# ==========================================

FUNDAMENTAL_ANALYST_SYSTEM_PROMPT = (
    "You are a Senior Quantitative Financial Analyst. Analyze corporate valuation, profitability, "
    "capital efficiency, growth, and solvency strictly using exact figures from SQLite tools.\n\n"
    "=== OPERATIONAL RULES ===\n"
    "1. Tool Discipline: Always call your fundamental tools (`get_stock_fundamentals`, "
    "`screen_stocks_by_fundamentals`, `list_available_tickers`). Pass uppercase tickers or exact company names "
    "(you can pass multiple tickers to `get_stock_fundamentals` for comparisons).\n"
    "2. Strict Precision: Cite exact figures with units (e.g., P/E: 38.1x, ROIC: 45.2%, Net Margin: 55.3%). "
    "Never fabricate, estimate, or confuse 'N/A' (missing) with '0.0'.\n"
    "3. Evidence Tagging: Format data citations as `[SQLite: {TICKER} | {Metric}: {Value}]`.\n"
    "4. Scope Boundary: Quantitative metrics only. Do not speculate on market rumors or news sentiment.\n\n"
    "=== RESPONSE FORMAT ===\n"
    "- Targeted Queries (e.g., 'What is AAPL P/E and ROIC?'): Answer directly and concisely with exact figures and evidence tags.\n"
    "- Multi-Stock Comparisons: Present a side-by-side Markdown comparison table (Valuation, Profitability, Growth, Solvency) with key takeaways.\n"
    "- Comprehensive Profiles: Group into:\n"
    "  • Valuation & Multiples: P/E, Forward P/E, PEG\n"
    "  • Profitability & Returns: Gross/Operating/Net Margins, ROE, ROIC\n"
    "  • Growth & Cash Flow: Revenue/EPS Growth (1Y/3Y), Free Cash Flow Margin\n"
    "  • Solvency & Health: Net Debt/EBITDA, Total Debt, Debt Coverage"
)



def create_fundamental_agent(
    llm,
    verbose: bool = False,
    max_iterations: int = 10
) -> ReActAgent:
    """
    Constructs the Fundamental Analyst ReActAgent equipped with all tools from get_fundamental_tools().
    """
    tools = get_fundamental_tools()
    if hasattr(ReActAgent, "from_tools"):
        return ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=verbose,
            max_iterations=max_iterations,
            system_prompt=FUNDAMENTAL_ANALYST_SYSTEM_PROMPT,
        )
    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        streaming=False,
        system_prompt=FUNDAMENTAL_ANALYST_SYSTEM_PROMPT,
    )


def create_fundamental_agent_tool(llm) -> FunctionTool:
    """
    Wraps the Fundamental Agent into a single FunctionTool for the CIO Supervisor.
    Supervisor calls: fundamental_analyst(query: str) -> str
    """
    agent = create_fundamental_agent(llm=llm)

    async def _async_run_agent(query: str) -> str:
        try:
            res = await agent.run(user_msg=query)
            tool_details = []
            for tc in getattr(res, "tool_calls", []) or []:
                t_name = getattr(tc, "tool_name", "Tool")
                t_kwargs = getattr(tc, "tool_kwargs", {})
                raw_out = getattr(tc, "raw_output", getattr(tc, "tool_output", ""))
                out_str = str(getattr(raw_out, "content", raw_out))
                tool_details.append(
                    f"  ⚙️ [SQLite Tool]: {t_name}\n"
                    f"     • Variables Passed: {json.dumps(t_kwargs, default=str)}\n"
                    f"     • Raw Data Returned: {out_str}"
                )

            raw_resp = getattr(res, "response", res)
            content = getattr(raw_resp, "content", raw_resp)
            text_content = str(content).strip()

            sections = []
            if tool_details:
                sections.append("=== 🛠️ SUB-AGENT INTERNAL TOOL CALLS ===")
                sections.extend(tool_details)
            sections.append("=== 📝 SUB-AGENT SYNTHESIS ===")
            sections.append(text_content)
            return "\n".join(sections)
        except Exception as e:
            return f"Error executing fundamental analyst: {str(e)}"

    def fundamental_analyst(query: str) -> str:
        """
        Delegates a quantitative financial analysis task to the Fundamental Analyst Agent.
        Use this tool for:
        - Stock valuation multiples (P/E, forward P/E, PEG)
        - Profitability and margins (gross, operating, net margins)
        - Return metrics (ROE, ROIC)
        - Growth rates (1Y and 3Y revenue and EPS growth)
        - Balance sheet solvency & debt (Net Debt/EBITDA, Total Debt, Free Cash Flow)
        - Quantitative stock screening and multi-stock financial comparisons
        """
        # Prioritize async runner with full sub-agent tool call traces and citations
        if hasattr(agent, "run"):
            try:
                return run_async_safe(lambda: _async_run_agent(query))
            except Exception as e:
                return f"Error executing fundamental analyst: {str(e)}"

        if hasattr(agent, "chat"):
            try:
                res = agent.chat(query)
                return str(res)
            except Exception:
                pass

        return str(agent)

    return FunctionTool.from_defaults(
        fn=fundamental_analyst,
        async_fn=_async_run_agent,
        name="fundamental_analyst",
        description=(
            "Expert quantitative financial analyst. Use for retrieving exact corporate financial metrics, "
            "P/E ratios, ROE, ROIC, margins, growth rates, Net Debt/EBITDA, stock screening, and financial comparisons."
        ),
    )