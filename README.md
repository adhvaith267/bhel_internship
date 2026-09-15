# Enterprise RAG Engine

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4285F4?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

**A high-performance, privacy-preserving Retrieval-Augmented Generation engine for enterprise document intelligence.**

[Features](#key-features) · [Architecture](#architecture-overview) · [API](#api-endpoints) · [Setup](#quick-start) · [Configuration](#configuration)

---

## Overview

The Enterprise RAG Engine provides a fully local, GPU-accelerated document QA system that retrieves grounded answers with exact page citations from indexed PDFs. No data ever leaves the machine. The only entry point is the web interface — no terminal chat, no CLI tools.

---

## Key Features

- **Two-Stage Hybrid Retrieval Pipeline**
  - Dense semantic search via FAISS using `all-MiniLM-L6-v2`
  - Sparse lexical search via BM25Okapi
  - Reciprocal Rank Fusion (RRF) for balanced merging
  - Multi-query expansion for technical document tokenization

- **Neural Cross-Encoder Re-ranking**
  - Re-ranks candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Configurable alpha blending with RRF scores

- **Maximal Marginal Relevance (MMR) Diversification**
  - Reduces redundancy in retrieved passages

- **Fully Local & Private**
  - Runs via Ollama or llama.cpp GGUF models
  - Zero data leaves the machine

- **Native Document Ingestion**
  - High-speed PyMuPDF text normalization and table extraction
  - Persistent disk caching with SHA-256 validation
  - Incremental re-indexing
  - Optional OCR fallback via Tesseract CLI

- **Production-Grade Faithfulness**
  - Citation verification against retrieved passages
  - Corrective retry on ungrounded claims
  - Abstention gate for low-confidence queries

---

## Architecture Overview

```mermaid
flowchart TD
    PDF["docs/*.pdf"] -->|PyMuPDF Text & Tables| Ingest["src/indexing/parser.py"]
    Ingest -->|Chunks & Markdown Tables| Cache[".rag_cache (FAISS + BM25)"]

    UserQuery["User Question"] -->|Hybrid Query| RRF["Stage 1: Hybrid Search (FAISS + BM25 RRF)"]
    Cache --> RRF
    RRF -->|Top-20 Candidates| Rerank["Stage 2: Neural Cross-Encoder Reranker"]
    Rerank -->|MMR Diversified Top-N| MMR["Stage 3: Maximal Marginal Relevance"]
    MMR -->|Top-N Passages| Prompt["Grounded Prompt + Citations"]

    Prompt --> LLM["LLM Engine (Ollama / llama.cpp GGUF)"]
    LLM --> Answer["Grounded Answer with Page Citations"]

    Answer --> Verify["Citation Verification"]
    Verify -->|Ungrounded?| Retry["Corrective Retry (Strict Prompt)"]
    Retry --> LLM
    Verify -->|Verified| Final["Final Response"]
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web-based chat interface |
| `POST` | `/ask` | RAG question-answering with citations |
| `POST` | `/api/chat` | JSON & SSE streaming endpoint (`stream: true`) |
| `GET` | `/api/documents` | List indexed PDFs, file sizes, and chunk statistics |
| `POST` | `/api/reindex` | Trigger fresh re-indexing of all documents |
| `GET` | `/api/status` | System health, active model, device, and chunk count |
| `GET` | `/api/health` | Alias for `/api/status` |

---

## Project Structure

```
bhel_internship/              # Repository root
├── src/                      # Python package (all source code)
│   ├── api/                  # FastAPI backend
│   │   ├── routes.py         # FastAPI app factory + route definitions
│   │   ├── schemas.py        # Pydantic request/response models
│   │   └── deps.py           # Shared engine dependency
│   ├── core/                 # Core engine components
│   │   ├── config.py         # Settings + path & tuning constants
│   │   ├── logger.py         # Rich colorized structured logging
│   │   └── exceptions.py     # Shared domain exceptions
│   ├── engine/               # RAG orchestration
│   │   ├── pipeline.py       # RAG pipeline + singleton accessor
│   │   └── grounding.py      # Citation verification + refusal detection
│   ├── indexing/             # Document processing
│   │   ├── parser.py         # PDF text/table extraction, headings, OCR fallback
│   │   └── chunker.py        # Recursive text splitting with overlap
│   ├── llm/                  # Language model integration
│   │   ├── provider.py       # LLM connectors (Ollama & llama-cpp-python)
│   │   └── prompts.py        # Grounded system prompt
│   ├── retrieval/            # Hybrid retrieval
│   │   └── hybrid.py         # FAISS + BM25 + RRF + cross-encoder reranker
│   ├── ui/                   # Web UI
│   │   └── templates/
│   │       └── index.html    # ChatGPT-style interface
│   ├── __init__.py
│   └── __main__.py           # `python -m src` — starts the web server
├── docs/                     # Drop PDFs here to index them
├── models/                   # Place GGUF model files here
├── .env.example              # Environment configuration template
├── .gitignore
└── requirements.txt          # Python dependencies
```

---

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Clone repository
git clone https://github.com/adhvaith267/bhel_internship.git
cd bhel_internship

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Install local model
ollama pull qwen2.5:7b  # or set LLM_PROVIDER=llamacpp
```

### 2. Add Documents

Place PDF files in the `docs/` directory:
```bash
cp /path/to/documents/*.pdf docs/
```

### 3. Start the Server

```bash
python -m src
```

Open `http://localhost:5000` in your browser.

### 4. Using the Web Interface

- Type questions in the chat box and press Enter
- Filter context to a specific document using the dropdown
- Citations are shown with page numbers and relevance scores
- Use the streaming API for token-by-token responses

---

## API Usage

**Single question:**
```python
import httpx

response = httpx.post(
    "http://localhost:5000/ask",
    json={
        "question": "What are the attendance requirements for theory courses?",
        "top_n": 5,
        "doc_name": "B.TechCSE-2026-27-Curriculum-Syllabi.pdf"
    }
)
print(response.json())
```

**Streaming:**
```python
import httpx, json

with httpx.stream("POST", "http://localhost:5000/api/chat",
                  json={"question": "Summarize the document", "stream": True}) as r:
    for line in r.iter_lines():
        if line.startswith("data:") and "[DONE]" not in line:
            print(json.loads(line[5:]))
```

---

## Configuration

All settings can be overridden via environment variables or a `.env` file at the repo root. Copy `.env.example` to `.env` and edit as needed.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `5000` | Server port |
| `DEBUG` | `false` | Enable uvicorn auto-reload |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Retrieval Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `650` | Character chunk length |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `TOP_K_CANDIDATES` | `20` | Candidate chunks from stage-1 hybrid search |
| `TOP_N_RERANK` | `5` | Passages selected after neural reranking |
| `RRF_K` | `60` | RRF rank constant |
| `DENSE_WEIGHT` | `0.6` | Weight of dense (FAISS) retrieval in RRF |
| `BM25_WEIGHT` | `0.4` | Weight of BM25 in RRF |
| `RERANK_FUSION_ALPHA` | `0.85` | Cross-encoder vs RRF blend (1.0 = cross-encoder only) |
| `MMR_ENABLED` | `true` | Enable Maximal Marginal Relevance diversification |
| `MMR_LAMBDA` | `0.5` | MMR relevance/diversity trade-off (1.0 = pure relevance) |
| `USE_RERANKER` | `true` | Enable neural cross-encoder reranker |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |

### Faithfulness

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ABSTENTION` | `true` | Abstain when evidence score is too low |
| `ABSTAIN_MIN_SCORE` | `-10.0` | Minimum cross-encoder logit to attempt an answer |
| `ENABLE_CORRECTIVE_RETRY` | `true` | Retry with a stricter prompt on ungrounded citations |

### Ingestion & OCR

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_OCR` | `true` | OCR fallback via Tesseract CLI for image-only pages |
| `OCR_DPI` | `300` | DPI for OCR rasterization |
| `OCR_MIN_CHARS` | `50` | Min characters extracted before OCR is triggered |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | Provider: `auto`, `ollama`, or `llamacpp` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model name |
| `GGUF_MODEL_PATH` | `models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf` | Path to GGUF model file |

---

*Built for Enterprise — Local. Private. Grounded.*
