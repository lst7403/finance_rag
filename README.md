# Financial RAG Stock Selection Assistant

A privacy-first, local multi-agent Financial Intelligence and Stock Selection platform built with **LlamaIndex**, **Ollama (Qwen 2.5:7B)**, **ChromaDB**, **SQLite**, and **FastAPI**.

It combines deterministic quantitative fundamental screening (SQLite) with semantic market news intelligence (ChromaDB + Cross-Encoder Reranker) under an autonomous **Chief Investment Officer (CIO) Supervisor Agent** to generate structured, evidence-backed Bull/Bear investment theses with real-time token streaming.

---

## 🌟 Key Features

* **Multi-Agent Decision Architecture:**
  * **CIO Supervisor Agent:** Orchestrates specialized sub-agents, aggregates fundamental metrics and market sentiment, and delivers balanced Bull/Bear theses with clear citation tags (`[SQLite: TICKER | Metric]` and `[News: TICKER | Headline]`).
  * **Fundamental Analyst Agent:** Performs deterministic quantitative queries directly against SQLite. Analyzes P/E multiples, ROIC, ROE, margins, 1Y/3Y CAGRs, FCF yield, and Net Debt-to-EBITDA solvency.
  * **Market News Analyst Agent:** Conducts vector semantic search over financial news via ChromaDB, reranked by a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), with dynamic real-time fallbacks.
* **Dual-Layer Persistence:**
  * **SQLite (`data/database/foundamental_sql.db`):** High-speed relational storage with indexed financial metrics for deterministic filtering and ranking.
  * **ChromaDB (`data/database/news_vector`):** Vector store with persistent embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and unique document deduplication.
* **Automated Data Ingestion Pipeline:**
  * Constituent ticker scraping (NASDAQ-100 / S&P 500 via Slickcharts).
  * Batch fetching of financial statements, balance sheets, cash flows, and news via `yfinance`.
  * In-memory ratio calculations and atomic upserts.
* **Modern Web Dashboard:**
  * Built with Vanilla HTML/CSS/JS (no heavy frontend framework dependencies).
  * Real-time Server-Sent Events (SSE) word-by-word streaming with intermediate agent execution steps.
  * Quantitative stock screener with threshold filters (min ROIC, max P/E, min Gross Margin, etc.).
  * Side-by-side comparative stock valuation matrix.
* **100% Local & Privacy-Preserving:**
  * All LLM inferences run locally through Ollama. No proprietary cloud API keys (e.g. OpenAI) or subscription tokens required.

---

## 🏗️ Architecture Overview

```
                      +-----------------------------+
                      |   Modern Web Dashboard      |
                      | (HTML/CSS/JS + SSE Client)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     FastAPI Web Server      |
                      |     (http://127.0.0.1:8000) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    CIO Supervisor Agent     |
                      |   (LlamaIndex + Qwen 2.5)   |
                      +-------+-------------+-------+
                              |             |
           +------------------+             +------------------+
           |                                                   |
           v                                                   v
+-------------------------------+             +-------------------------------+
|   Fundamental Analyst Agent   |             |   Market News Analyst Agent   |
|   (Quantitative Reasoning)   |             |    (Semantic RAG & Rerank)    |
+---------------+---------------+             +---------------+---------------+
                |                                             |
                v                                             v
+-------------------------------+             +-------------------------------+
|       SQLite Database         |             |      ChromaDB Vector Store    |
| (data/database/foundamental_   |             | (data/database/news_vector)   |
|            sql.db)            |             | + Cross-Encoder Reranker      |
+-------------------------------+             +-------------------------------+
```

---

## 📋 Prerequisites

1. **Python 3.10+** (tested on Python 3.10, 3.11, and 3.14).
2. **[Ollama](https://ollama.com/)** installed and running locally.
3. Download the target LLM model in Ollama:
   ```bash
   ollama pull qwen2.5:7b
   ```
4. *(Optional but Recommended)* An NVIDIA GPU with CUDA drivers installed for GPU-accelerated embedding inference and reranking.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd finance_rag
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on PyTorch & CUDA:**
> To enable NVIDIA GPU acceleration for Hugging Face embeddings and Cross-Encoder reranking:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Configuration (Optional)
The project defaults to sensible local values (`http://localhost:11434`, `qwen2.5:7b`, etc.). If you want to customize settings, copy `.env.example`:
```bash
cp .env.example .env
```
Available environment variables:
* `OLLAMA_BASE_URL`: Ollama service endpoint (default: `http://localhost:11434`)
* `OLLAMA_MODEL`: Target model tag (default: `qwen2.5:7b`)
* `EMBED_MODEL_NAME`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
* `RERANK_MODEL_NAME`: Cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)

### 5. Ingest Initial Data (First Run)
Populate your local SQLite database and ChromaDB vector index with NASDAQ-100 fundamentals and market news:
```bash
python get_data/data_pipeline.py
```
*Options:*
* `--index nasdaq100` (default) or `--index sp500`
* `--workers 6` (parallel worker threads)
* `--max-news 10` (max articles to fetch per ticker)

*(You can also trigger or refresh data anytime directly from the Web Dashboard).*

### 6. Start the Server
```bash
python run_server.py
```
Access the application dashboard at:
👉 `http://127.0.0.1:8000`

---

## 💻 Features & Usage Walkthrough

### 1. Natural Language Financial Queries & CIO Supervisor
Ask complex investment questions via the chat input:
* *"Analyze NVDA's current valuation, profit margins, and recent AI data center demand."*
* *"Compare Apple and Microsoft in terms of ROIC, Free Cash Flow, and cloud growth."*
* *"Find technology companies with ROIC > 20% and P/E under 35."*

The CIO Supervisor streams its internal reasoning steps (tool executions) and outputs a structured thesis with:
* **Valuation & Multiples:** P/E, Forward P/E, PEG
* **Profitability & Returns:** Gross Margin, Operating Margin, ROE, ROIC
* **Growth & Cash Generation:** Revenue/EPS 1Y & 3Y CAGR, FCF Margin
* **Solvency & Risk Factors:** Net Debt/EBITDA, Liquidity
* **Catalysts & Bear Case Risks:** Summarized from indexed news

### 2. Quantitative Fundamentals Screener
Directly screen all tracked stocks across customizable criteria:
* **Minimum ROIC** (e.g., `>= 15%`)
* **Maximum P/E Ratio** (e.g., `<= 30x`)
* **Minimum Gross Margin** (e.g., `>= 40%`)
* **Minimum 1Y Revenue Growth** (e.g., `>= 10%`)
* **Maximum Net Debt / EBITDA** (e.g., `<= 2.5x`)
* **Sector Filter** (e.g., `Technology`, `Healthcare`, `Consumer Cyclical`)

### 3. Single Stock Bull / Bear Thesis Generator
Input any ticker symbol (e.g., `TSLA`, `GOOGL`, `AMZN`) to generate an automated side-by-side Bull vs. Bear argument with citations.

### 4. Background Data Sync
Click the **"Sync Data"** button on the dashboard to pull the latest SEC statements, valuation multiples, and market news in a background worker without interrupting your active sessions.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Dashboard user interface |
| `GET` | `/api/system/status` | Health check for Ollama, CUDA GPU, Vector DB, and SQLite count |
| `GET` | `/api/stocks` | Lists all stocks with fundamental metrics and news counts |
| `GET` | `/api/stocks/{ticker}/news` | Retrieves parsed news articles for a specific stock ticker |
| `GET` | `/api/rag/stream` | Server-Sent Events (SSE) endpoint for real-time word-by-word streaming |
| `POST` | `/api/rag/query` | Synchronous 3-stage RAG query with RouterRetriever & Reranker |
| `POST` | `/api/rag/analyze` | Generates a structured Bull/Bear thesis for a single stock |
| `POST` | `/api/rag/compare` | Generates a comparative financial analysis between two tickers |
| `POST` | `/api/fundamentals/screen` | Deterministic screening across quantitative financial thresholds |
| `POST` | `/api/pipeline/refresh` | Triggers background data harvesting and re-indexing |
| `POST` | `/api/chat/reset` | Clears conversation memory for a session ID |

---

## 📁 Repository Structure

```text
finance_rag/
├── app.py                      # FastAPI web server and REST/SSE endpoints
├── config.py                   # Centralized application configuration
├── run_server.py               # Launch script with system and Ollama verification
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variable overrides
├── .gitignore                  # Git tracking rules (excludes databases and caches)
├── get_data/                   # Data acquisition & storage subsystem
│   ├── get_data.py             # Slickcharts scraping & yfinance ratio calculations
│   ├── sql_handler.py          # SQLite database layer for fundamental metrics
│   ├── chroma_handler.py       # ChromaDB vector store indexing & deduplication
│   └── data_pipeline.py        # Automated daily pipeline runner
├── rag/                        # Multi-Agent RAG reasoning engine
│   ├── rag_engine.py           # Core FinancialRAGEngine & LlamaIndex orchestrator
│   └── tools/                  # Agent tool definitions
│       ├── cio_supervisor.py   # Chief Investment Officer supervisor agent
│       ├── fundamental_tools.py# Deterministic SQLite query & screening tools
│       ├── news_tools.py       # ChromaDB vector search & Cross-Encoder reranking
│       └── tool_utils.py       # Normalization & async execution helpers
├── web/                        # Web dashboard frontend
│   ├── index.html              # Dashboard markup
│   ├── styles.css              # Custom styling & responsive layouts
│   ├── script.js               # Dashboard logic & SSE streaming client
│   └── tooltipData.js          # Financial metric explanation glossary
└── data/                       # Local data storage directory
    └── database/               # Ignored in git; created on first run
        ├── .gitkeep            # Directory placeholder
        ├── foundamental_sql.db # Local SQLite database
        └── news_vector/        # ChromaDB vector embeddings
```

---

## ⚖️ Financial & Investment Disclaimer

> [!WARNING]
> **Not Financial Advice:** This application and its AI-generated outputs (including quantitative screenings, ratio comparisons, and Bull/Bear theses) are strictly for **educational, informational, and research purposes only**. 
> None of the content constitutes an offer, solicitation, or recommendation to buy, hold, or sell any security or financial asset. Always perform your own independent research and consult a licensed financial advisor before making any investment decisions.

---

## 📄 License

This project is licensed under the **MIT License** with an explicit financial disclaimer. See the `LICENSE` file for complete legal terms.
