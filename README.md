# Document Intelligence Assistant

A local web app that analyses business documents such as vendor proposals, contracts and reports. Upload PDF or TXT files, then ask questions across them, compare them, extract structured fields, summarise them and download a PDF report. Every answer cites the document and page it came from, and if the documents don't contain the answer, the app says so instead of guessing.

It runs fully locally with Ollama (Llama 3.2), so no paid API is needed.

## Features

- **Document management:** upload multiple PDF/TXT files, track processing status, group them into collections
- **Q&A:** ask questions across several documents, with answers that cite document and page
- **Comparison:** side-by-side table of prices, terms and dates, plus a diff between two versions of a document
- **Extraction:** pull structured fields (vendor, price, payment terms and so on) with sources and export to JSON or CSV
- **Summaries:** executive summary, key points, risks and dates/values/obligations
- **PDF report** and a **history** of saved analyses

## Tech stack

React, Vite, TypeScript, FastAPI, Pydantic, LangChain, LangGraph, Ollama (Llama 3.2), FAISS, BM25, SQLite, pytest

## How it works

1. **Upload:** the file is validated, its text is extracted page by page and the text is split into chunks that never cross a page boundary. Chunks are stored in SQLite.
2. **Routing:** a LangGraph workflow sends each request to Q&A, comparison, extraction or summarisation.
3. **Retrieval:** the question is matched against the chunks using semantic search (FAISS) and keyword search (BM25), combined into one score.
4. **Generation:** the model sees only the retrieved passages and returns JSON, which is validated with Pydantic.
5. **Verification:** the code, not the model, maps citations back to the document and page. Values the model can't support are flagged and when evidence is too weak the app answers "not enough information".
6. **Result:** the answer is saved to history and shown as a chat, table or summary.

## Setup 

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8000
```

Frontend (in a second terminal):

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Model settings are in `backend/.env` and in the app's Settings page.

## Limitations

- No OCR, so scanned PDFs are rejected with a message.
- A small local model can miss details. Check important results against the original documents.
