"""
cio_supervisor.py - Top-Level Chief Investment Officer (CIO) Supervisor Agent

Orchestrates domain-specialized sub-agents:
1. fundamental_analyst: Autonomous quantitative agent querying SQLite by condition
2. market_news_analyst: Autonomous qualitative agent querying ChromaDB vector news with reranking
"""

import sys
import re
import time
from typing import Any, Dict, List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool

from .fundamental_tools import create_fundamental_agent_tool
from .news_tools import create_news_agent_tool
from .tool_utils import run_async_safe

__all__ = [
    "CIO_SYSTEM_PROMPT",
    "create_cio_agent",
    "execute_cio_query",
    "aexecute_cio_query",
    "astream_cio_query",
    "get_supervisor_tools",
    "safe_print",
]


def safe_print(*args, **kwargs):
    """Prints safely to stdout, handling Windows CP950/CP437 unicode encoding mismatches."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        clean_args = [
            str(a).encode(enc, errors="replace").decode(enc, errors="replace")
            for a in args
        ]
        print(*clean_args, **kwargs)

# --- CIO Persona & Evidence-Driven Delegation Instructions ---

CIO_SYSTEM_PROMPT = (
    "You are the Chief Investment Officer (CIO). Direct two specialist domain analysts:\n"
    "1. fundamental_analyst: Quantitative metrics (P/E, PEG, margins, ROE, ROIC, growth, debt) from SQLite.\n"
    "2. market_news_analyst: Qualitative news headlines, catalysts, and sentiment from vector search.\n"
    "You may pass stock tickers or company names to both analysts.\n\n"
    "=== EVIDENTIARY MANDATE ===\n"
    "- Ground all findings: Never state numbers or catalysts without citing analyst evidence.\n"
    "- Inline citations: Embed tags for facts, e.g. `[Evidence: P/E 38.1 via SQLite]` or `[Evidence: Bloomberg, Aug 2024]`.\n"
    "- Never extrapolate or fabricate unverified data. Disclose missing metrics honestly.\n\n"
    "=== RESPONSE PROTOCOL ===\n"
    "- Simple queries (e.g. 'What is NVDA ROIC?'): Provide a direct, concise answer with the inline evidence tag.\n"
    "- Comprehensive / Verdict queries: Structure as an institutional memo:\n"
    "  1. Executive Verdict (Bullish / Neutral / Bearish)\n"
    "  2. Quantitative Fundamentals (Multiples, Margins, Debt)\n"
    "  3. Qualitative Catalysts & Sentiment (News, Events)\n"
    "  4. Key Risks\n"
    "  5. Evidence Summary Table (Source | Key Data Point)"
)




def get_supervisor_tools(
    index: Optional[VectorStoreIndex] = None,
    llm=None,
    reranker=None
) -> List[BaseTool]:
    """
    Assembles the CIO Supervisor's high-level domain sub-agent tools:
    1. fundamental_analyst: Autonomous quantitative agent wrapping all fundamental tools
    2. market_news_analyst: Autonomous qualitative agent wrapping news vector search and catalysts
    """
    tools: List[BaseTool] = []
    if llm is not None:
        tools.append(create_fundamental_agent_tool(llm=llm))
    if index is not None and llm is not None:
        tools.append(create_news_agent_tool(index=index, llm=llm, reranker=reranker))
    return tools


def create_cio_agent(
    index: Optional[VectorStoreIndex] = None,
    llm=None,
    reranker=None,
    additional_tools: Optional[List[BaseTool]] = None,
    verbose: bool = True,
    max_iterations: int = 10,
    system_prompt: Optional[str] = None
) -> ReActAgent:
    """
    Constructs the Top-Level CIO (Chief Investment Officer) Supervisor ReActAgent.
    Equipped with fundamental_analyst and market_news_analyst sub-agent tools.
    """
    tools = get_supervisor_tools(index=index, llm=llm, reranker=reranker)
    if additional_tools:
        tools.extend(additional_tools)

    prompt = system_prompt or CIO_SYSTEM_PROMPT

    if hasattr(ReActAgent, "from_tools"):
        return ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=verbose,
            max_iterations=max_iterations,
            system_prompt=prompt,
        )
    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        system_prompt=prompt,
    )


def _parse_citations_and_trace(response: Any, streamed_sources: List[Dict[str, Any]], query_text: str):
    """Parses tool invocations, granular inner tool calls, and structured citation cards."""
    sources = []
    invoked_agents = []
    invoked_tools = []

    call_items = getattr(response, "sources", None) or getattr(response, "tool_calls", None) or []
    if not call_items and streamed_sources:
        call_items = [
            type("ToolCallWrapper", (), {
                "tool_name": s["tool_name"],
                "tool_kwargs": s["tool_kwargs"],
                "raw_output": s["output"]
            })()
            for s in streamed_sources
        ]

    for src in call_items:
        tool_name = getattr(src, "tool_name", "Unknown")
        raw_out = getattr(src, "raw_output", getattr(src, "tool_output", ""))
        if hasattr(raw_out, "content"):
            tool_output = str(raw_out.content)
        elif hasattr(raw_out, "blocks") and raw_out.blocks:
            tool_output = str(raw_out.blocks[0].text if hasattr(raw_out.blocks[0], "text") else raw_out.blocks[0])
        else:
            tool_output = str(raw_out)

        # Track invoked sub-agents
        if tool_name in ["fundamental_analyst", "market_news_analyst"]:
            if tool_name not in invoked_agents:
                invoked_agents.append(tool_name)
        else:
            if tool_name not in invoked_tools:
                invoked_tools.append(tool_name)

        # Parse granular inner tool calls executed by sub-agents (e.g. [SQLite Tool]: get_stock_fundamentals)
        inner_matches = re.findall(r"\[(?:SQLite|Vector)\s+Tool\]:\s*([a-zA-Z0-9_]+)", tool_output)
        for it in inner_matches:
            if it not in invoked_tools:
                invoked_tools.append(it)

        sources.append({
            "tool_name": tool_name,
            "tool_kwargs": getattr(src, "tool_kwargs", {}),
            "agent": tool_name if tool_name in ["fundamental_analyst", "market_news_analyst"] else "supervisor",
            "output": tool_output,
            "output_preview": tool_output
        })

    citations = []
    for src in sources:
        agent_name = src.get("agent") or src.get("tool_name", "")
        kwargs = src.get("tool_kwargs", {})
        output_text = src.get("output", "")
        kw_str = str(kwargs)

        ticker_match = re.search(r"\b([A-Z]{1,5})\b", kw_str) or re.search(r"\b([A-Z]{1,5})\b", query_text)
        ticker_sym = ticker_match.group(1) if ticker_match else "EVIDENCE"

        if "fundamental" in agent_name.lower():
            citations.append({
                "ticker": ticker_sym,
                "title": f"SQLite Quantitative Financials [{ticker_sym}]",
                "publisher": "SQLite Database (stock_fundamentals)",
                "link": "#",
                "excerpt": output_text[:280].strip() + ("..." if len(output_text) > 280 else ""),
                "agent": "fundamental_analyst",
                "type": "quantitative_data"
            })
        elif "news" in agent_name.lower():
            citations.append({
                "ticker": ticker_sym,
                "title": f"Market News Catalysts & Intelligence [{ticker_sym}]",
                "publisher": "ChromaDB Vector Store + Cross-Encoder",
                "link": "#",
                "excerpt": output_text[:280].strip() + ("..." if len(output_text) > 280 else ""),
                "agent": "market_news_analyst",
                "type": "qualitative_intelligence"
            })
        else:
            citations.append({
                "ticker": ticker_sym,
                "title": f"Research Tool: {src.get('tool_name', 'Analyst')}",
                "publisher": "Institutional CIO System",
                "link": "#",
                "excerpt": output_text[:280].strip(),
                "agent": agent_name,
                "type": "general_research"
            })

    trace = {
        "supervisor": "cio_react_agent",
        "agents": invoked_agents,
        "tools": invoked_tools if invoked_tools else invoked_agents,
        "iterations": len(sources)
    }

    return sources, citations, trace, invoked_agents, invoked_tools


async def aexecute_cio_query(
    agent: ReActAgent,
    query_text: str,
    model_name: Optional[str] = None,
    memory: Optional[Any] = None,
    event_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Native Async execution of the CIO Supervisor ReActAgent directly on FastAPI's ASGI event loop.
    Prints every Thought, Action, and Observation and streams step events in real time.
    """
    start_time = time.time()
    safe_print(f"\n[CIO Supervisor] Executing Multi-Agent Query (Native Async): '{query_text}'")

    response = None
    resp_str = ""
    is_error = False
    error_msg = None
    streamed_sources = []

    try:
        run_kwargs = {"user_msg": query_text}
        if memory is not None:
            run_kwargs["memory"] = memory

        if hasattr(agent, "run"):
            handler = agent.run(**run_kwargs)
            async for ev in handler.stream_events():
                ev_type = type(ev).__name__

                if ev_type == "AgentOutput":
                    resp_msg = getattr(ev, "response", None)
                    content = str(getattr(resp_msg, "content", "")) if resp_msg else ""
                    if content:
                        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nAnswer:|$)", content, re.DOTALL)
                        if thought_match:
                            thought = thought_match.group(1).strip()
                            if thought:
                                safe_print(f"\nThought: {thought}")
                                if event_callback:
                                    event_callback({
                                        "type": "step",
                                        "stage": "reasoning",
                                        "title": "Agent Thought",
                                        "message": thought,
                                        "log_line": f"Thought: {thought}"
                                    })

                elif ev_type == "ToolCall":
                    tool_name = getattr(ev, "tool_name", "Tool")
                    tool_kwargs = getattr(ev, "tool_kwargs", {})
                    safe_print(f"\nAction: {tool_name}")
                    safe_print(f"Action Input: {tool_kwargs}")
                    if event_callback:
                        event_callback({
                            "type": "step",
                            "stage": "tool_call",
                            "title": f"Action: {tool_name}",
                            "message": f"Calling {tool_name} with {tool_kwargs}",
                            "log_line": f"Action: {tool_name} | Input: {tool_kwargs}"
                        })

                elif ev_type == "ToolCallResult":
                    tool_name = getattr(ev, "tool_name", "Tool")
                    tool_kwargs = getattr(ev, "tool_kwargs", {})
                    raw_out = getattr(ev, "tool_output", "")
                    out_text = str(getattr(raw_out, "content", raw_out))
                    preview = out_text.strip()
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                    safe_print(f"\nObservation: {preview}")

                    streamed_sources.append({
                        "tool_name": tool_name,
                        "tool_kwargs": tool_kwargs,
                        "output": out_text
                    })

                    if event_callback:
                        event_callback({
                            "type": "step",
                            "stage": "observation",
                            "title": f"Observation from {tool_name}",
                            "message": preview,
                            "log_line": f"Observation: {preview}"
                        })

            response = await handler
            raw = getattr(response, "response", response)
            if hasattr(raw, "content"):
                resp_str = str(raw.content)
            elif hasattr(raw, "blocks") and raw.blocks:
                resp_str = str(getattr(raw.blocks[0], "text", raw.blocks[0]))
            else:
                resp_str = str(raw)
        elif hasattr(agent, "achat"):
            response = await agent.achat(query_text)
            resp_str = str(response)
        elif hasattr(agent, "chat"):
            response = agent.chat(query_text)
            resp_str = str(response)
        else:
            resp_str = str(agent)

    except Exception as e:
        safe_print(f"[CIO Supervisor] [WARN] Query execution error: {e}")
        resp_str = f"Error during CIO agent reasoning: {e}"
        is_error = True
        error_msg = str(e)

    elapsed = round(time.time() - start_time, 2)
    safe_print(f"\nAnswer: {resp_str}\n")

    sources, citations, trace, invoked_agents, invoked_tools = _parse_citations_and_trace(
        response=response,
        streamed_sources=streamed_sources,
        query_text=query_text
    )

    safe_print(f"[CIO Supervisor] [OK] Finished in {elapsed}s | Sub-agents: {invoked_agents} | Tools: {invoked_tools} | Calls: {len(sources)}")

    return {
        "status": "error" if is_error else "success",
        "error": error_msg,
        "query": query_text,
        "answer": resp_str,
        "response": resp_str,
        "sources": sources,
        "citations": citations,
        "trace": trace,
        "latency_seconds": elapsed,
        "model": model_name or "Qwen 2.5"
    }


async def astream_cio_query(
    agent: ReActAgent,
    query_text: str,
    model_name: Optional[str] = None,
    memory: Optional[Any] = None,
):
    """
    Native Async Generator: Streams real-time Thought, Action, Observation,
    Word-by-word token generation, and the final done event directly via Server-Sent Events.
    Zero thread spawning, zero loop recreation.
    """
    start_time = time.time()
    tool_desc = "CIO Supervisor (Multi-Agent)"
    safe_print(f"\n[CIO Supervisor] Native Async SSE Stream started for: '{query_text}'")

    response = None
    resp_str = ""
    streamed_sources = []
    is_error = False
    error_msg = None

    try:
        run_kwargs = {"user_msg": query_text}
        if memory is not None:
            run_kwargs["memory"] = memory

        if hasattr(agent, "run"):
            handler = agent.run(**run_kwargs)
            async for ev in handler.stream_events():
                ev_type = type(ev).__name__

                if ev_type == "AgentOutput":
                    resp_msg = getattr(ev, "response", None)
                    content = str(getattr(resp_msg, "content", "")) if resp_msg else ""
                    if content:
                        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nAnswer:|$)", content, re.DOTALL)
                        if thought_match:
                            thought = thought_match.group(1).strip()
                            if thought:
                                safe_print(f"\nThought: {thought}")
                                yield {
                                    "type": "step",
                                    "stage": "reasoning",
                                    "title": "Agent Thought",
                                    "message": thought,
                                    "log_line": f"Thought: {thought}"
                                }

                elif ev_type == "ToolCall":
                    tool_name = getattr(ev, "tool_name", "Tool")
                    tool_kwargs = getattr(ev, "tool_kwargs", {})
                    safe_print(f"\nAction: {tool_name}")
                    safe_print(f"Action Input: {tool_kwargs}")
                    yield {
                        "type": "step",
                        "stage": "tool_call",
                        "title": f"Action: {tool_name}",
                        "message": f"Calling {tool_name} with {tool_kwargs}",
                        "log_line": f"Action: {tool_name} | Input: {tool_kwargs}"
                    }

                elif ev_type == "ToolCallResult":
                    tool_name = getattr(ev, "tool_name", "Tool")
                    tool_kwargs = getattr(ev, "tool_kwargs", {})
                    raw_out = getattr(ev, "tool_output", "")
                    out_text = str(getattr(raw_out, "content", raw_out))
                    preview = out_text.strip()
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                    safe_print(f"\nObservation: {preview}")

                    streamed_sources.append({
                        "tool_name": tool_name,
                        "tool_kwargs": tool_kwargs,
                        "output": out_text
                    })

                    yield {
                        "type": "step",
                        "stage": "observation",
                        "title": f"Observation from {tool_name}",
                        "message": preview,
                        "log_line": f"Observation: {preview}"
                    }

            response = await handler
            raw = getattr(response, "response", response)
            if hasattr(raw, "content"):
                resp_str = str(raw.content)
            elif hasattr(raw, "blocks") and raw.blocks:
                resp_str = str(getattr(raw.blocks[0], "text", raw.blocks[0]))
            else:
                resp_str = str(raw)
        else:
            resp_str = str(agent)

    except Exception as e:
        safe_print(f"[CIO Supervisor] [WARN] Streaming error: {e}")
        resp_str = f"Error during CIO agent reasoning: {e}"
        is_error = True
        error_msg = str(e)
        yield {"type": "error", "message": error_msg}
        return

    elapsed = round(time.time() - start_time, 2)
    safe_print(f"\nAnswer: {resp_str}\n")

    sources, citations, trace, invoked_agents, invoked_tools = _parse_citations_and_trace(
        response=response,
        streamed_sources=streamed_sources,
        query_text=query_text
    )

    # Stream synthesized answer tokens
    chunk_size = 4
    words = resp_str.split(" ") if resp_str else []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]) + " "
        yield {
            "type": "token",
            "content": chunk
        }

    done_msg = f"Multi-Agent analysis finished in {elapsed}s | Sources: {len(sources)}"

    # Completion Event
    yield {
        "type": "done",
        "citations": citations,
        "log_line": f"[Pipeline Finished] {done_msg}",
        "metadata": {
            "routed": True,
            "multi_agent": True,
            "agent": "cio_supervisor",
            "model": model_name or "Qwen 2.5",
            "latency_seconds": elapsed,
            "tool": tool_desc,
            "trace": trace,
            "top_k": len(sources),
        }
    }


def execute_cio_query(
    agent: ReActAgent,
    query_text: str,
    model_name: Optional[str] = None,
    memory: Optional[Any] = None,
    event_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Synchronous compatibility wrapper around aexecute_cio_query for CLI / Jupyter Notebook usage.
    """
    return run_async_safe(
        lambda: aexecute_cio_query(
            agent=agent,
            query_text=query_text,
            model_name=model_name,
            memory=memory,
            event_callback=event_callback,
        )
    )

