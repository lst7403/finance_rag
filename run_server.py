"""
run_server.py - Launch FastAPI Backend for Financial RAG

Runs uvicorn on http://127.0.0.1:8000 serving both API endpoints and the Web Dashboard.
"""

import sys
import uvicorn
from rag import check_ollama_status

if __name__ == "__main__":
    print("=======================================================")
    print("Financial RAG Stock Selection Assistant Backend")
    print("=======================================================")
    
    status = check_ollama_status()
    print(f"Ollama Status: {status['status'].upper()}")
    if status['status'] == 'online':
        print(f"Target Model:  {status['target_model']} (Installed: {status['target_model_installed']})")
    else:
        print(f"Note: {status.get('hint', 'Start Ollama for local LLM inference.')}")
    print("=======================================================")
    print("Starting FastAPI Web Server at: http://127.0.0.1:8000")
    print("=======================================================\n")
    
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
