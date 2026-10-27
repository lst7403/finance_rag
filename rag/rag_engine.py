"""
rag_engine.py - Hierarchical Multi-Agent Financial Runtime & Service Engine

Manages models, vector store indices, and runtime execution for the Hierarchical Multi-Agent system:
- CIO Supervisor ReActAgent (Top-level strategist)
- Fundamental Analyst Sub-Agent (Deterministic SQLite metrics by condition)
- Market News Analyst Sub-Agent (ChromaDB semantic vector search + cross-encoder reranking)
"""

import json
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

import requests
from config import CONFIG
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import BaseTool
from llama_index.llms.ollama import Ollama

from get_data.chroma_handler import (
    get_chroma_document_count,
    get_chroma_vector_store,
    get_embedding_model,
)
from .tools import (
    create_cio_agent,
    aexecute_cio_query,
    astream_cio_query,
)


def get_reranker(
    top_n: int = 6,
    model_name: Optional[str] = None
) -> Optional[SentenceTransformerRerank]:
    """Initializes the SentenceTransformer cross-encoder reranker with CUDA/CPU acceleration."""
    target_model = model_name or CONFIG.rerank_model_name
    device = CONFIG.rag_device

    try:
        return SentenceTransformerRerank(
            model=target_model,
            top_n=top_n,
            device=device
        )
    except Exception as e:
        print(f"[RAGEngine] [WARN] Failed to load reranker '{target_model}' on {device}: {e}. Continuing without reranking.")
        return None


def get_llm() -> Ollama:
    """Initializes Ollama Qwen 2.5 LLM with extended timeout for institutional financial synthesis."""
    return Ollama(
        model=CONFIG.ollama_model,
        base_url=CONFIG.ollama_base_url,
        temperature=0.15,
        request_timeout=300.0
    )


class FinancialRAGEngine:
    """
    Hierarchical Multi-Agent Financial RAG Engine:
    Maintains in-memory model singletons and delegates queries to the CIO Supervisor Agent.
    """

    # =========================================================================
    # 1. Index & Runtime Lifecycle Management
    # =========================================================================

    def __init__(self):
        import torch
        device = CONFIG.rag_device
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        self.device_info = {
            "device": device,
            "device_name": device_name,
            "cuda_available": device == "cuda"
        }
        print(f"[RAGEngine] Hardware Acceleration: {device.upper()} | Device: {device_name}")

        self.embed_model = get_embedding_model()
        Settings.embed_model = self.embed_model

        self.rerank_model_name = CONFIG.rerank_model_name
        self.llm = get_llm()
        if self.llm:
            Settings.llm = self.llm

        self.index = self._load_existing_index()
        self.sessions: Dict[str, ChatMemoryBuffer] = {}

    def _load_existing_index(self) -> VectorStoreIndex:
        """Loads the persistent ChromaDB collection from disk."""
        doc_count = get_chroma_document_count(
            chroma_dir=CONFIG.chroma_dir,
            collection_name=CONFIG.chroma_collection,
        )
        if doc_count == 0:
            raise RuntimeError("ChromaDB index is empty. Run data indexing before querying.")

        print(f"[RAGEngine] Loading ChromaDB index ({doc_count} documents)...")
        vector_store = get_chroma_vector_store(
            chroma_dir=CONFIG.chroma_dir,
            collection_name=CONFIG.chroma_collection
        )
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )

    def reload_index(self) -> bool:
        """Hot-reloads the ChromaDB vector store index into memory without restarting."""
        self.index = self._load_existing_index()
        return True

    def get_session_memory(
        self,
        session_id: str = "default",
        token_limit: int = 3000
    ) -> ChatMemoryBuffer:
        """
        Retrieves or creates a rolling ChatMemoryBuffer for the specified session.
        Old turns are automatically evicted when token_limit is reached,
        preventing context window overflow while preserving conversational context.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMemoryBuffer.from_defaults(
                token_limit=token_limit
            )
        return self.sessions[session_id]

    def reset_session_memory(self, session_id: str = "default") -> None:
        """Clears conversation memory for a session (e.g. on New Chat)."""
        if session_id in self.sessions:
            self.sessions[session_id].reset()

    # =========================================================================
    # 2. Multi-Agent Query & Streaming Execution
    # =========================================================================

    async def aquery(
        self,
        query_text: str,
        ticker_filter: Optional[str] = None,
        ticker_filters: Optional[List[str]] = None,
        sector_filter: Optional[str] = None,
        max_iterations: int = 10,
        session_id: str = "default",
        reset_session: bool = False,
        event_callback: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Native async execution of an institutional financial inquiry through the CIO Supervisor Agent.
        Runs directly on FastAPI's ASGI event loop.
        Supports multi-turn interactive conversations via session-based ChatMemoryBuffer.
        """
        if reset_session:
            self.reset_session_memory(session_id)

        augmented_query = query_text
        if ticker_filter:
            augmented_query += f" (Focused on ticker: {ticker_filter})"
        elif ticker_filters:
            augmented_query += f" (Comparing tickers: {', '.join(ticker_filters)})"
        if sector_filter:
            augmented_query += f" (Sector: {sector_filter})"

        agent = self.create_cio_agent(verbose=True, max_iterations=max_iterations)
        memory = self.get_session_memory(session_id=session_id)
        result = await aexecute_cio_query(
            agent,
            augmented_query,
            model_name=CONFIG.ollama_model,
            memory=memory,
            event_callback=event_callback,
        )

        result["query"] = query_text
        result["session_id"] = session_id
        result.setdefault("routed", True)
        result.setdefault("agent", "cio_supervisor")
        result.setdefault("multi_agent", True)
        return result

    def query(
        self,
        query_text: str,
        ticker_filter: Optional[str] = None,
        ticker_filters: Optional[List[str]] = None,
        sector_filter: Optional[str] = None,
        max_iterations: int = 10,
        session_id: str = "default",
        reset_session: bool = False,
        event_callback: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous compatibility wrapper for execution in CLI scripts or legacy callers.
        Delegates directly to aquery using the persistent agent background runner.
        """
        from .tools.tool_utils import run_async_safe
        return run_async_safe(
            lambda: self.aquery(
                query_text=query_text,
                ticker_filter=ticker_filter,
                ticker_filters=ticker_filters,
                sector_filter=sector_filter,
                max_iterations=max_iterations,
                session_id=session_id,
                reset_session=reset_session,
                event_callback=event_callback,
                **kwargs
            )
        )

    async def astream_pipeline_events(
        self,
        query_text: str,
        session_id: str = "default",
        reset_session: bool = False,
        max_iterations: int = 10,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Native Async Generator: Streams real-time execution steps and synthesized tokens via Server-Sent Events.
        Zero thread creation, zero nested event loops. Direct async iteration on ASGI event loop.
        """
        if reset_session:
            self.reset_session_memory(session_id)

        agent = self.create_cio_agent(verbose=True, max_iterations=max_iterations)
        memory = self.get_session_memory(session_id=session_id)

        async for ev in astream_cio_query(
            agent=agent,
            query_text=query_text,
            model_name=CONFIG.ollama_model,
            memory=memory,
        ):
            if ev.get("type") == "done" and "metadata" in ev:
                ev["metadata"]["session_id"] = session_id
            yield ev

    def stream_pipeline_events(
        self,
        query_text: str,
        session_id: str = "default",
        reset_session: bool = False,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous generator wrapper for backward compatibility with sync streaming clients.
        """
        import asyncio
        import queue
        import threading

        q: queue.Queue = queue.Queue()

        def _runner():
            async def _consume():
                try:
                    async for event in self.astream_pipeline_events(
                        query_text=query_text,
                        session_id=session_id,
                        reset_session=reset_session,
                        **kwargs
                    ):
                        q.put(event)
                except Exception as e:
                    q.put({"type": "error", "message": str(e)})
                finally:
                    q.put(None)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_consume())
            finally:
                loop.close()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

    async def agenerate_stock_thesis(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """Generates a structured Bull/Bear investment thesis for a specific stock ticker using Multi-Agent (async)."""
        ticker = ticker.upper()
        prompt = (
            f"Draft a comprehensive institutional Bull/Bear investment thesis for {ticker}. "
            f"Use fundamental_analyst for valuation, margins, growth, and debt ratios from SQLite, "
            f"and market_news_analyst for recent catalysts, corporate announcements, and sentiment. "
            f"Provide an explicit investment verdict (BULLISH, NEUTRAL, or BEARISH)."
        )
        res = await self.aquery(prompt)
        return {
            "ticker": ticker,
            "thesis": res.get("answer", ""),
            "citations": res.get("citations", res.get("sources", [])),
            "latency_seconds": res.get("latency_seconds", 0.0),
            "model": CONFIG.ollama_model,
            "multi_agent": True
        }

    def generate_stock_thesis(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for agenerate_stock_thesis."""
        from .tools.tool_utils import run_async_safe
        return run_async_safe(lambda: self.agenerate_stock_thesis(ticker, **kwargs))

    async def acompare_stocks(self, ticker_a: str, ticker_b: str, **kwargs) -> Dict[str, Any]:
        """Compares two tickers across valuation, margins, ROE/ROIC, debt, and news catalysts (async)."""
        ticker_a = ticker_a.upper()
        ticker_b = ticker_b.upper()
        query_text = (
            f"Directly compare {ticker_a} versus {ticker_b}. Analyze their valuation (P/E), "
            f"profitability (Gross/Operating/FCF Margins), return on capital (ROE, ROIC), "
            f"1Y and 3Y growth rates, debt balance sheet strength, and recent news catalysts. "
            f"Conclude with a clear comparative investment verdict on which stock presents a better risk-reward."
        )
        return await self.aquery(query_text)

    def compare_stocks(self, ticker_a: str, ticker_b: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for acompare_stocks."""
        from .tools.tool_utils import run_async_safe
        return run_async_safe(lambda: self.acompare_stocks(ticker_a, ticker_b, **kwargs))

    # =========================================================================
    # 3. Agent & Tool Construction Factories
    # =========================================================================

    def create_cio_agent(
        self,
        additional_tools: Optional[List[BaseTool]] = None,
        verbose: bool = True,
        max_iterations: int = 10,
    ) -> ReActAgent:
        """
        Builds the Top-Level CIO (Chief Investment Officer) Supervisor ReActAgent.
        Delegates construction to rag.tools.cio_supervisor.create_cio_agent.
        """
        reranker = get_reranker(top_n=4, model_name=self.rerank_model_name)
        return create_cio_agent(
            index=self.index,
            llm=self.llm,
            reranker=reranker,
            additional_tools=additional_tools,
            verbose=verbose,
            max_iterations=max_iterations,
        )


# =============================================================================
# 4. System Health & Diagnostics
# =============================================================================

def check_ollama_status() -> Dict[str, Any]:
    """Checks if Ollama service is reachable and lists installed models."""
    try:
        res = requests.get(f"{CONFIG.ollama_base_url}/api/tags", timeout=3)
        if res.status_code == 200:
            models_data = res.json().get("models", [])
            model_names = [m.get("name") for m in models_data]
            has_target = CONFIG.ollama_model in model_names
            return {
                "status": "online" if has_target else "model_missing",
                "target_model": CONFIG.ollama_model,
                "target_model_installed": has_target,
                "available_models": model_names,
                "base_url": CONFIG.ollama_base_url
            }
    except Exception:
        pass

    return {
        "status": "offline",
        "target_model": CONFIG.ollama_model,
        "target_model_installed": False,
        "available_models": [],
        "base_url": CONFIG.ollama_base_url,
        "hint": f"Ensure Ollama is running. Run: 'ollama run {CONFIG.ollama_model}' in terminal."
    }


if __name__ == "__main__":
    status = check_ollama_status()
    print("Ollama Status:", json.dumps(status, indent=2))
