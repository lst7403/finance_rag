"""
app.py - FastAPI Backend for Financial RAG Stock Selection Assistant

Exposes REST and SSE streaming endpoints for natural language stock queries,
single-stock Bull/Bear thesis generation, news extraction, and system status checks.
"""

import os
import sys
import json
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from typing import Optional, List, Dict, Any
from config import CONFIG
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from get_data import (
    load_json,
    save_json,
    load_fundamentals,
    save_fundamentals,
    load_news,
    save_news,
    fetch_and_save_news,
    batch_fetch_fundamentals,
    get_slickcharts_ticker,
    FUNDAMENTALS_FILE,
    FUNDAMENTALS_DB,
    NEWS_FILE
)
from rag import (
    index_all_data,
    FinancialRAGEngine,
    check_ollama_status,
    get_stock_fundamentals,
    screen_stocks_by_fundamentals,
    compare_stocks_fundamentals,
    list_available_tickers
)

WORKSPACE_DIR = CONFIG.workspace_dir
DATA_DIR = CONFIG.data_dir
WEB_DIR = CONFIG.web_dir

app = FastAPI(
    title="Financial RAG Stock Selection API",
    description="LlamaIndex + Qwen 2.5 Local Financial RAG Engine with 2-Stage Cross-Encoder Reranking & CUDA Acceleration",
    version="1.1.0"
)

# Enable CORS for local web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine lazy initialization
_rag_engine: Optional[FinancialRAGEngine] = None


def get_engine() -> FinancialRAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = FinancialRAGEngine()
    return _rag_engine


# Request Schemas
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(6, ge=1, le=50)
    initial_top_k: Optional[int] = Field(None, ge=1, le=200)
    ticker_filter: Optional[str] = None
    sector_filter: Optional[str] = None
    use_reranker: bool = True
    use_router: bool = True
    session_id: Optional[str] = "default"
    reset_session: bool = False


class AnalyzeRequest(BaseModel):
    ticker: str
    use_reranker: bool = True


class CompareRequest(BaseModel):
    ticker_a: str
    ticker_b: str
    use_reranker: bool = True


class RefreshRequest(BaseModel):
    refresh_news: bool = True
    refresh_fundamentals: bool = True
    sample_limit: Optional[int] = Field(None, ge=1, le=500)


class ScreenRequest(BaseModel):
    min_roic: Optional[float] = None
    max_pe: Optional[float] = None
    min_gross_margin: Optional[float] = None
    min_revenue_growth: Optional[float] = None
    max_debt_to_ebitda: Optional[float] = None
    min_roe: Optional[float] = None
    sector: Optional[str] = None
    sort_by: str = "roic"
    top_n: int = Field(10, ge=1, le=500)


class CompareFundamentalsRequest(BaseModel):
    tickers: List[str]


class AgentQueryRequest(BaseModel):
    query: str
    max_iterations: int = Field(10, ge=1, le=50)
    session_id: Optional[str] = "default"
    reset_session: bool = False



# --- API Routes ---

@app.get("/api/system/status")
def get_system_status():
    """Returns Ollama, CUDA GPU, Vector DB, RouterRetriever, and Cross-Encoder Reranker status."""
    ollama_info = check_ollama_status()
    try:
        import torch
        device = CONFIG.rag_device
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        device_info = {"device": device, "device_name": device_name, "cuda_available": device == "cuda"}
    except Exception:
        device_info = {"device": "cpu", "device_name": "CPU", "cuda_available": False}

    fundamentals = load_fundamentals()
    news_data = load_news()
    total_news = sum(len(articles) for articles in news_data.values()) if isinstance(news_data, dict) else 0
    news_tickers = sum(1 for articles in news_data.values() if articles) if isinstance(news_data, dict) else 0
    
    embed_device_desc = f"{CONFIG.embed_model_name} (CUDA: {device_info['device_name']})" if device_info['cuda_available'] else f"{CONFIG.embed_model_name} (CPU)"
    rerank_device_desc = f"{CONFIG.rerank_model_name} (CUDA: {device_info['device_name']})" if device_info['cuda_available'] else f"{CONFIG.rerank_model_name} (CPU)"

    return {
        "ollama": ollama_info,
        "cuda": device_info,
        "stocks_count": len(fundamentals),
        "fundamentals_database": "SQLite (./data/database/foundamental_sql.db)",
        "news_tickers_count": news_tickers,
        "total_news_count": total_news,
        "vector_store": "ChromaDB (./data/database/news_vector)",
        "framework": "LlamaIndex",
        "embedding_model": embed_device_desc,
        "architecture": "Hierarchical Multi-Agent",
        "supervisor": "CIO Supervisor (LlamaIndex ReActAgent)",
        "sub_agents": [
            "Fundamental Analyst (SQLite condition queries)",
            "Market News Analyst (Semantic Vector & Reranking)"
        ],
        "multi_agent": True,
        "router_retriever": False,
        "two_stage_retrieval": True
    }


@app.get("/api/stocks")
def get_stocks():
    """Returns all stocks with flattened fundamental metrics from SQLite and news counts."""
    fundamentals = load_fundamentals()
    news_data = load_news()

    enriched_stocks = []
    for s in fundamentals:
        ticker = s.get("ticker", "").upper()
        news_list = news_data.get(ticker, []) if isinstance(news_data, dict) else []
        funds = s.get("fundamentals", {})
        
        item = {
            "ticker": ticker,
            "company_name": s.get("company_name", ticker),
            "sector": s.get("sector", "N/A"),
            "industry": s.get("industry", "N/A"),
            **funds,
            "news_count": len(news_list),
            "latest_headline": news_list[0]["title"] if news_list else None,
            "latest_news_date": news_list[0]["publish_time"] if news_list else None,
            "last_updated": s.get("last_updated")
        }
        enriched_stocks.append(item)

    return {
        "count": len(enriched_stocks),
        "source": "SQLite (data/database/foundamental_sql.db)",
        "database": "foundamental_sql.db",
        "table": "stock_fundamentals",
        "stocks": enriched_stocks
    }


@app.get("/api/sql/stocks")
def get_sql_stocks():
    """Directly queries and returns raw fundamental records from the SQLite stock_fundamentals table."""
    from get_data import fetch_all_fundamentals
    raw_stocks = fetch_all_fundamentals()
    flattened_stocks = []
    for s in raw_stocks:
        funds = s.get("fundamentals", {}) if isinstance(s.get("fundamentals"), dict) else {}
        flattened_stocks.append({
            **s,
            **funds
        })
    return {
        "count": len(flattened_stocks),
        "source": "SQLite (data/database/foundamental_sql.db)",
        "database": "foundamental_sql.db",
        "table": "stock_fundamentals",
        "stocks": flattened_stocks
    }



@app.get("/api/news/{ticker}")
def get_stock_news(ticker: str):
    """Returns all parsed news articles with UUIDs for a specific ticker."""
    ticker = ticker.upper()
    news_data = load_news()
    articles = news_data.get(ticker, []) if isinstance(news_data, dict) else []
    
    # If no cached news exists, try fetching on demand
    if not articles:
        updated_news = fetch_and_save_news([ticker], max_news_per_ticker=8)
        articles = updated_news.get(ticker, [])
        if articles:
            index_all_data()
            if _rag_engine:
                _rag_engine.reload_index()

    return {
        "ticker": ticker,
        "count": len(articles),
        "news": articles
    }


@app.post("/api/rag/query")
async def rag_query(req: QueryRequest):
    """Executes a 3-stage RAG query with RouterRetriever & Cross-Encoder Reranker (Native Async)."""
    print(f"\n[API] [QUERY] Received POST /api/rag/query: '{req.query}' (use_router={req.use_router}, use_reranker={req.use_reranker})")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    engine = get_engine()
    try:
        result = await engine.aquery(
            query_text=req.query,
            top_k=req.top_k,
            initial_top_k=req.initial_top_k,
            ticker_filter=req.ticker_filter,
            sector_filter=req.sector_filter,
            use_reranker=req.use_reranker,
            use_router=req.use_router,
            session_id=req.session_id or "default",
            reset_session=req.reset_session,
        )
        print(f"[API] [OK] Completed /api/rag/query in {result.get('latency_seconds')}s")
        return result
    except Exception as e:
        print(f"[API] [ERR] Query error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")


@app.get("/api/rag/stream")
async def rag_stream_query(
    query: str = Query(..., description="Investor natural language query"),
    use_reranker: bool = Query(True, description="Enable 2-stage Cross-Encoder reranking"),
    use_router: bool = Query(True, description="Enable LLM RouterRetriever"),
    session_id: str = Query("default", description="Chat session ID for multi-turn history"),
    reset_session: bool = Query(False, description="Reset chat memory for this session"),
):
    """
    Streams RAG pipeline execution events and response tokens via Server-Sent Events (SSE) in real-time.
    Supports multi-turn interactive chat sessions using session_id.
    Runs natively async directly on FastAPI/Uvicorn ASGI event loop.
    """
    print(f"\n[API] [STREAM] Received GET /api/rag/stream: '{query}' (session={session_id}, reset={reset_session})")
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    engine = get_engine()

    async def event_generator():
        try:
            async for event in engine.astream_pipeline_events(
                query,
                use_reranker=use_reranker,
                use_router=use_router,
                session_id=session_id,
                reset_session=reset_session,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            print(f"[API] [ERR] SSE Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.post("/api/rag/analyze")
async def rag_analyze_stock(req: AnalyzeRequest):
    """Generates a structured Bull/Bear thesis for a single stock ticker (Native Async)."""
    print(f"\n[API] [ANALYZE] Received POST /api/rag/analyze for ticker: {req.ticker} (use_reranker={req.use_reranker})")
    if not req.ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")

    engine = get_engine()
    try:
        result = await engine.agenerate_stock_thesis(req.ticker, use_reranker=req.use_reranker)
        print(f"[API] [OK] Completed /api/rag/analyze for {req.ticker} in {result.get('latency_seconds')}s")
        return result
    except Exception as e:
        print(f"[API] [ERR] Analyze error: {e}")
        raise HTTPException(status_code=500, detail=f"Thesis generation failed: {str(e)}")


@app.post("/api/rag/compare")
async def rag_compare_stocks(req: CompareRequest):
    """Generates a side-by-side comparative investment analysis between two stocks (Native Async)."""
    print(f"\n[API] [COMPARE] Received POST /api/rag/compare: {req.ticker_a} vs {req.ticker_b} (use_reranker={req.use_reranker})")
    engine = get_engine()
    try:
        result = await engine.acompare_stocks(req.ticker_a, req.ticker_b, use_reranker=req.use_reranker)
        print(f"[API] [OK] Completed comparison {req.ticker_a} vs {req.ticker_b} in {result.get('latency_seconds')}s")
        return result
    except Exception as e:
        print(f"[API] [ERR] Compare error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# --- Direct Raw Fundamental Tool Routes & Agent Endpoints ---

@app.get("/api/tools/fundamentals/{ticker}")
def api_get_stock_fundamentals(ticker: str):
    """Directly fetches raw fundamental metrics dictionary for a specific stock ticker."""
    res = get_stock_fundamentals(ticker)
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res.get("error"))
    return res


@app.post("/api/tools/screen")
def api_screen_fundamentals(req: ScreenRequest):
    """Directly screens the fundamentals database by quantitative financial thresholds."""
    res = screen_stocks_by_fundamentals(
        min_roic=req.min_roic,
        max_pe=req.max_pe,
        min_gross_margin=req.min_gross_margin,
        min_revenue_growth=req.min_revenue_growth,
        max_debt_to_ebitda=req.max_debt_to_ebitda,
        min_roe=req.min_roe,
        sector=req.sector,
        sort_by=req.sort_by,
        top_n=req.top_n
    )
    return res


@app.post("/api/tools/compare")
def api_compare_fundamentals(req: CompareFundamentalsRequest):
    """Directly compares raw fundamentals across a list of stock tickers."""
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Must provide at least one ticker.")
    return compare_stocks_fundamentals(req.tickers)


@app.get("/api/tools/tickers")
def api_list_tickers():
    """Lists all available tickers tracked in the fundamental database."""
    return list_available_tickers()


@app.post("/api/agent/query")
async def api_agent_query(req: AgentQueryRequest):
    """
    Executes an autonomous ReActAgent query equipped with raw fundamental tools
    and semantic vector news search (Native Async).
    """
    print(f"\n[API] [AGENT] Received POST /api/agent/query: '{req.query}'")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    engine = get_engine()
    try:
        result = await engine.aquery(
            req.query,
            max_iterations=req.max_iterations,
            session_id=req.session_id or "default",
            reset_session=req.reset_session,
        )
        return result
    except Exception as e:
        print(f"[API] [ERR] Agent query error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent query failed: {str(e)}")


@app.post("/api/chat/reset")
def reset_chat_session(session_id: str = Query("default")):
    """Resets conversational memory for a specific chat session (e.g. New Chat button)."""
    engine = get_engine()
    engine.reset_session_memory(session_id)
    return {"status": "success", "session_id": session_id, "message": f"Memory reset for session '{session_id}'."}


@app.post("/api/pipeline/refresh")
def refresh_pipeline(req: RefreshRequest, background_tasks: BackgroundTasks):
    """Refreshes news/fundamentals and updates ChromaDB vector index in the background."""
    print(f"\n[API] 🔄 Received POST /api/pipeline/refresh (refresh_news={req.refresh_news}, refresh_funds={req.refresh_fundamentals})")
    def run_refresh():
        print("[Pipeline Background Task] 🚀 Starting unified background pipeline refresh...")
        from get_data import run_pipeline
        run_pipeline(
            refresh_fundamentals=req.refresh_fundamentals,
            refresh_news=req.refresh_news,
        )
        # Reload engine index
        if _rag_engine:
            print("[Pipeline Background Task] ♻️ Reloading FinancialRAGEngine in-memory index...")
            _rag_engine.reload_index()
        print("[Pipeline Background Task] ✅ Background refresh complete!")

    background_tasks.add_task(run_refresh)
    return {"status": "started", "message": "Pipeline refresh and re-indexing started in background."}


# Mount data folder for static downloads / fallbacks
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


# Serve root index.html from web directory
@app.get("/")
def serve_dashboard():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Dashboard index.html not found in web/")


# Catch-all route to serve static assets from web directory (styles.css, script.js, tooltipData.js, images)
@app.get("/{filename:path}")
def serve_static(filename: str):
    file_path = os.path.join(WEB_DIR, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
