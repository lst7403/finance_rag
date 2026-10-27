"""
rag/tools - Hierarchical Multi-Agent Financial Architecture

Structure:
┌─────────────────────────────────────────────────────────────┐
│ 1. cio_supervisor.py                                        │
│    Top-Level Chief Investment Officer (CIO) ReActAgent      │
│    Directs and synthesizes domain sub-agents                │
├──────────────────────────────┬──────────────────────────────┤
│ 2. fundamental_tools.py      │ 3. news_tools.py             │
│    Quantitative SQL Tools    │    Vector Search & Reranking │
│    + Fundamental Sub-Agent   │    + News Sub-Agent          │
└──────────────────────────────┴──────────────────────────────┘
"""

# Shared Tool Utilities & Safe Async Runner
from .tool_utils import (
    normalize_ratio,
    resolve_ticker,
    resolve_ticker_list,
    run_async_safe,
)

# 1. CIO Supervisor Agent
from .cio_supervisor import (
    CIO_SYSTEM_PROMPT,
    create_cio_agent,
    execute_cio_query,
    aexecute_cio_query,
    astream_cio_query,
    get_supervisor_tools,
)

# 2. Fundamental Tools & Analyst Sub-Agent (Deterministic SQLite)
from .fundamental_tools import (
    compare_stocks_fundamentals,
    create_fundamental_agent,
    create_fundamental_agent_tool,
    get_fundamental_tools,
    get_stock_fundamentals,
    list_available_tickers,
    screen_stocks_by_fundamentals,
)

# 3. Market News Tools & Analyst Sub-Agent (Semantic Vector + Reranking)
from .news_tools import (
    create_news_agent,
    create_news_agent_tool,
    create_news_query_engine_tool,
    get_latest_news_headlines,
    get_news_tools,
)


__all__ = [
    # 1. CIO Supervisor Agent
    "CIO_SYSTEM_PROMPT",
    "create_cio_agent",
    "execute_cio_query",
    "aexecute_cio_query",
    "astream_cio_query",
    "get_supervisor_tools",
    # 2. Fundamental Analyst & Tools
    "create_fundamental_agent",
    "create_fundamental_agent_tool",
    "get_fundamental_tools",
    "get_stock_fundamentals",
    "screen_stocks_by_fundamentals",
    "compare_stocks_fundamentals",
    "list_available_tickers",
    # 3. Market News Analyst & Tools
    "create_news_agent",
    "create_news_agent_tool",
    "create_news_query_engine_tool",
    "get_news_tools",
    "get_latest_news_headlines",
    # 4. Shared Utilities
    "run_async_safe",
    "resolve_ticker",
    "resolve_ticker_list",
    "normalize_ratio",
]
