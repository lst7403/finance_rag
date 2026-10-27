"""
news_tools.py - Semantic Vector News & Catalyst Tools for Financial RAG

Wraps ChromaDB vector retrieval and cross-encoder reranking into LlamaIndex
QueryEngineTools and FunctionTools for autonomous LLM agents.
"""

import json
from typing import List, Optional, Dict, Any, Union
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.tools import QueryEngineTool, FunctionTool, BaseTool
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from config import CONFIG
from llama_index.core.agent import ReActAgent
from .tool_utils import resolve_ticker, resolve_ticker_list, run_async_safe

__all__ = [
    "create_news_agent",
    "create_news_agent_tool",
    "create_news_query_engine_tool",
    "get_latest_news_headlines",
    "get_news_tools",
]


DEFAULT_NEWS_QA_PROMPT = PromptTemplate(
    "You are a Senior Financial News Analyst. Answer the following question based on the provided news context.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Provide factual, objective analysis highlighting catalysts, sentiment, dates, and direct business impacts:\n"
)


# In-memory cached ChromaDB news collection singleton
_CACHED_CHROMA_COLLECTION = None


def _get_news_collection():
    """Returns the cached ChromaDB news collection singleton to prevent repeated disk re-initializations."""
    global _CACHED_CHROMA_COLLECTION
    if _CACHED_CHROMA_COLLECTION is None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CONFIG.chroma_dir)
            _CACHED_CHROMA_COLLECTION = client.get_collection(CONFIG.chroma_collection)
        except Exception:
            _CACHED_CHROMA_COLLECTION = None
    return _CACHED_CHROMA_COLLECTION


def get_latest_news_headlines(
    ticker: Union[str, List[str]], 
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieves the latest parsed news headlines and summaries for one or more stock tickers or company names.
    Queries the primary ChromaDB vector database directly, using news.json only as a fallback.
    
    Args:
        ticker (Union[str, List[str]]): Stock symbol(s) or company name(s) (e.g. 'NVDA', 'Apple', 'MSFT, GOOGL', or ['NVDA', 'AMD']).
        limit (int): Maximum number of headlines to return per stock (default: 5).
        
    Returns:
        List[Dict[str, Any]]: List of news items with title, publisher, publish_time, and summary.
    """
    if not ticker:
        return []

    tickers = resolve_ticker_list(ticker)
    if not tickers:
        return []

    collection = _get_news_collection()
    all_articles = []

    for clean_ticker in tickers:
        ticker_articles = []
        if collection is not None:
            try:
                result = collection.get(
                    where={"ticker": clean_ticker},
                    include=["metadatas", "documents"],
                    limit=max(1, limit) * 2
                )
                metadatas = result.get("metadatas", [])
                documents = result.get("documents", [])
                for meta, doc_text in zip(metadatas, documents):
                    if meta and meta.get("doc_type") == "news":
                        ticker_articles.append({
                            "id": meta.get("yf_uuid") or meta.get("doc_id"),
                            "ticker": clean_ticker,
                            "title": meta.get("title", ""),
                            "publisher": meta.get("publisher", "Market News"),
                            "publish_time": meta.get("publish_time", ""),
                            "link": meta.get("link", ""),
                            "summary": meta.get("summary", "") or doc_text,
                        })
            except Exception:
                pass

        all_articles.extend(ticker_articles[:max(1, limit)])

    return all_articles


def create_news_query_engine_tool(
    index: VectorStoreIndex,
    llm,
    reranker=None,
    name: str = "financial_news_and_catalyst_search",
    description: Optional[str] = None,
    similarity_top_k: int = 8,
    rerank_top_n: int = 4,
    qa_prompt: Optional[PromptTemplate] = None,
) -> QueryEngineTool:
    """
    Wraps the vector store retriever and cross-encoder reranker into a QueryEngineTool
    for semantic search over market news catalysts, headlines, sentiment, and corporate developments.
    """
    default_desc = (
        "Search vector database for recent market news catalysts, headlines, corporate developments, "
        "earnings announcements, sentiment shifts, and qualitative financial events."
    )
    news_filter = [ExactMatchFilter(key="doc_type", value="news")]
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        filters=MetadataFilters(filters=news_filter)
    )
    
    node_postprocessors = [reranker] if reranker else []
    
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt or DEFAULT_NEWS_QA_PROMPT,
        response_mode="compact"
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors
    )
    
    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=name,
        description=description or default_desc
    )


def get_news_tools(
    index: Optional[VectorStoreIndex] = None, 
    llm=None, 
    reranker=None
) -> List[BaseTool]:
    """
    Returns news and market catalyst tools for LLM agent tool-calling.
    """
    tools: List[BaseTool] = [
        FunctionTool.from_defaults(
            fn=get_latest_news_headlines,
            name="get_latest_news_headlines",
            description=(
                "Retrieves recent news headlines, publishers, and release dates for a stock ticker or company name. "
                "Supports single or multiple companies (e.g. 'NVDA', 'Apple', or 'NVDA, AMD'). "
                "Use this tool to quickly review recent announcements for known companies."
            )
        )
    ]
    if index is not None and llm is not None:
        tools.append(create_news_query_engine_tool(index=index, llm=llm, reranker=reranker))
    return tools


# ==========================================
# Market News Analyst Agent & Supervisor Tool
# ==========================================

NEWS_ANALYST_SYSTEM_PROMPT = (
    "You are a Senior Market Intelligence Analyst. Research recent catalysts, corporate announcements, "
    "earnings, macro developments, and market sentiment strictly using news vector and headline tools.\n\n"
    "=== OPERATIONAL RULES ===\n"
    "1. Tool Discipline: Always query `get_latest_news_headlines` (for recent items) and "
    "`financial_news_and_catalyst_search` (for semantic search & deep context). Pass tickers or company names.\n"
    "2. Grounded Evidence: Cite every catalyst with headline, publisher, and publication date formatted as "
    "`[News: '{Headline}' | {Publisher} | {Date}]`. Never invent dates, sources, or events.\n"
    "3. Objective Sentiment: Classify sentiment as Bullish, Neutral, or Bearish strictly substantiated by documented news facts.\n"
    "4. Scope Boundary: Qualitative news and market intelligence only. Refer balance sheet requests to quantitative tools.\n\n"
    "=== RESPONSE FORMAT ===\n"
    "- Targeted Queries (e.g., 'Latest NVDA earnings headlines'): Return a concise list of verified headlines with dates and publishers.\n"
    "- Multi-Stock Inquiries: Group findings by ticker/company with distinct catalyst summaries.\n"
    "- Comprehensive Reports: Structure as:\n"
    "  • Key Verified Catalysts: Bulleted events with evidence tags `[News: Title | Publisher | Date]`\n"
    "  • Operational & Commercial Impact: Demand trends, product releases, regulatory or supply chain impact\n"
    "  • Sentiment Assessment: Bullish / Neutral / Bearish rating with factual justification"
)



def create_news_agent(
    index: VectorStoreIndex,
    llm,
    reranker=None,
    verbose: bool = False,
    max_iterations: int = 10
) -> ReActAgent:
    """
    Constructs the Market News Analyst ReActAgent equipped with all tools from get_news_tools().
    """
    tools = get_news_tools(index=index, llm=llm, reranker=reranker)
    if hasattr(ReActAgent, "from_tools"):
        return ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=verbose,
            max_iterations=max_iterations,
            system_prompt=NEWS_ANALYST_SYSTEM_PROMPT,
        )
    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        streaming=False,
        system_prompt=NEWS_ANALYST_SYSTEM_PROMPT,
    )


def create_news_agent_tool(
    index: VectorStoreIndex,
    llm,
    reranker=None
) -> FunctionTool:
    """
    Wraps the News Agent into a single FunctionTool for the CIO Supervisor.
    Supervisor calls: market_news_analyst(query: str) -> str
    """
    agent = create_news_agent(index=index, llm=llm, reranker=reranker)

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
                    f"  📰 [Vector Tool]: {t_name}\n"
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
            return f"Error executing market news analyst: {str(e)}"

    def market_news_analyst(query: str) -> str:
        """
        Delegates a market news, catalyst, or sentiment research task to the Market News Analyst Agent.
        Use this tool for:
        - Recent financial news headlines and breaking developments
        - Earnings reports commentary and conference call takeaways
        - Corporate growth catalysts and product launches
        - Macroeconomic, regulatory, or competitive market events
        - Investor sentiment shifts and qualitative market events
        """
        # Prioritize async runner with full sub-agent tool call traces and citations
        if hasattr(agent, "run"):
            try:
                return run_async_safe(lambda: _async_run_agent(query))
            except Exception as e:
                return f"Error executing market news analyst: {str(e)}"

        if hasattr(agent, "chat"):
            try:
                res = agent.chat(query)
                return str(res)
            except Exception:
                pass

        return str(agent)

    return FunctionTool.from_defaults(
        fn=market_news_analyst,
        async_fn=_async_run_agent,
        name="market_news_analyst",
        description=(
            "Expert market news and catalyst analyst. Use for researching recent news headlines, "
            "earnings announcements, corporate catalysts, competitive developments, and sentiment shifts."
        ),
    )
