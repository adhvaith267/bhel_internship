<div align="center">

# DocuRAG — Enterprise RAG Engine with Local LLMs

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system for querying enterprise PDF documents using quantized open-source LLMs — no external API calls, no data leaving your infrastructure.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C.svg)](https://www.langchain.com/)
[![llama.cpp](https://img.shields.io/badge/Inference-llama.cpp-000000.svg)](https://github.com/ggerganov/llama.cpp)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-005571.svg)](https://github.com/facebookresearch/faiss)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Two Entry Points: Web App vs. CLI](#two-entry-points-web-app-vs-cli)
- [Supported Models](#supported-models)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Overview

DocuRAG is a Retrieval-Augmented Generation pipeline built to run entirely on local infrastructure. It ingests enterprise PDF documents — including both free text and tabular data — chunks and embeds them, and serves natural language answers using quantized GGUF language models run through `llama.cpp`. Because inference, embeddings, and vector search all run locally, no document content or query ever leaves the host machine, making it suitable for confidential or regulated enterprise data.

## Key Features

- **Local-Only Inference** — Runs quantized GGUF models via `llama-cpp-python`; no external LLM API required
- **Selectable Model Backend** — Swap between 17 pre-configured Mistral, Mixtral, and lightweight models via an interactive CLI menu or a hardcoded index
- **Dual-Layer Chunking (Parent/Child Retrieval)** — Uses LangChain's `ParentDocumentRetriever` to embed small, precise child chunks for retrieval while returning larger parent chunks for richer context
- **Table-Aware PDF Extraction** — Extracts both narrative text (via PyMuPDF) and structured tables (via `img2table` + BeautifulSoup) from source PDFs, preserving table structure as JSON
- **Local Embeddings** — Uses a locally-hosted `all-MiniLM-L6-v2` sentence-transformer model for vector embeddings, avoiding external embedding API calls
- **FAISS Vector Store** — Efficient similarity search over embedded document chunks
- **Two Interfaces** — A multi-document Flask web app for production use, and a single-document interactive CLI for quick local testing and experimentation

## Architecture

The system follows a two-stage retrieval architecture rather than naive single-chunk RAG:

1. **Document Extraction**
   `extract_docs_from_pdf.py` processes each source PDF twice: once to pull plain text page-by-page using PyMuPDF (`fitz`), and once to detect and extract tables using `img2table`, converting each table into structured JSON via BeautifulSoup-based HTML parsing.

2. **Parent/Child Splitting**
   Extracted text is wrapped into LangChain `Document` objects ("parent" documents). A `RecursiveCharacterTextSplitter` then derives smaller "child" chunks (500 chars, 30 overlap) from each parent (2000 chars, 1000 overlap), with each child chunk tagged with its parent's ID in metadata.

3. **Embedding and Indexing**
   Parent documents are embedded using a local `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`) and indexed in a FAISS vector store. Parent documents are also cached in an `InMemoryStore` docstore, keyed by parent ID.

4. **Retrieval**
   A `ParentDocumentRetriever` performs similarity search over the child-chunk embedding space, then resolves matches back to their full parent document — balancing retrieval precision (small chunks) with generation context quality (larger, coherent chunks).

5. **Answer Generation**
   The retrieved parent context and user query are wrapped into a Mistral-style instruction prompt (`[INST] ... [END]`) and passed to a locally-running GGUF model via `llama-cpp-python`, which returns a generated answer.

## Two Entry Points: Web App vs. CLI

The repository exposes two independent ways to run the RAG pipeline, each with a different scope:

| | `rag.py` → `app.py` (Web) | `rag_cli.py` (CLI) |
|---|---|---|
| Document scope | **All** documents in `docs/` (via `select_all_docs()`), indexed together | **One** document, chosen interactively at startup (via `select_doc()`) |
| Model selection | Hardcoded index passed to `select_model(index)` | Interactive menu via `select_model()` with no argument |
| Interface | Flask REST API (`POST /ask`) + web UI (`index.html`) | Terminal loop (`input()` / `print()`) |
| Use case | Persistent service querying the full enterprise document set | Quick, ad-hoc testing against a single document |

Both share the same underlying retrieval logic (parent/child chunking, FAISS, `ParentDocumentRetriever`) and the same local LLM inference approach, just wired up differently.

## Supported Models

Model selection is configurable via `models.py`, which maintains an indexed registry of local GGUF checkpoints. At runtime, `select_model()` either accepts a model index directly or presents an interactive menu:

| Family        | Example Variants                                                    |
|---------------|------------------------------------------------------------------------|
| Mistral 7B    | Instruct v0.2 (Q6_K), Instruct v0.3 (Q4_1, Q8_0, F16), Code 16K QLoRA (Q8_0) |
| Mistral Nemo  | Instruct 2407 (Q4_K_M)                                                |
| Mistral Small | 24B Instruct 2501 (BF16, Q8_0, Q4_K_M)                                |
| Mixtral 8x7B  | Instruct v0.1 (Q8_0), Base v0.1 (Q4_K_M)                              |
| Lightweight   | Phi-2 (Q4_K_M), TinyLlama 1.1B Chat (Q5_K_M)                          |

Models are swapped without code changes by editing the model index passed to `select_model()`, making it straightforward to trade off inference speed against answer quality depending on available hardware.

## Tech Stack

| Layer                | Technology                                              |
|----------------------|------------------------------------------------------------|
| LLM Inference        | `llama-cpp-python` (GGUF quantized models)               |
| Orchestration        | LangChain (`langchain-community`, `langchain-core`, `langchain-text-splitters`) |
| Retrieval            | `ParentDocumentRetriever`, `InMemoryStore`                |
| Vector Store         | FAISS                                                     |
| Embeddings           | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) |
| PDF Text Extraction  | PyMuPDF (`fitz`)                                          |
| PDF Table Extraction | `img2table`, BeautifulSoup                                |
| Backend              | Python, Flask, Flask-CORS                                  |
| Frontend             | HTML templates, static assets                             |

## Prerequisites

- Python 3.8 or higher
- `pip` for package management
- Local GGUF model file(s) downloaded and placed under `/models/` (paths configured in `models.py`)
- A local copy of the `all-MiniLM-L6-v2` embedding model under `/models/all-MiniLM-L6-v2`
- Sufficient RAM/VRAM for your chosen quantized model (lightweight options like TinyLlama or Phi-2 are recommended for constrained hardware)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/adhvaith267/DocuRAG.git
cd DocuRAG
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your documents

Place PDF files into the `docs/` directory. All files here are picked up by `select_all_docs()` for the web app, or offered one-by-one by `select_doc()` for the CLI.

### 5. Configure model paths

Update the `path` values in `models.py` to point to your local GGUF model files, and set the embedding model path in `rag.py` / `rag_cli.py` if it differs from `/models/all-MiniLM-L6-v2`.

## Configuration

| File                       | Purpose                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| `models.py`                | Registry of available GGUF models and interactive selector             |
| `select_docs.py`           | `select_doc()` — interactive single-file picker for the CLI; `select_all_docs()` — returns every file in `docs/` for the web app |
| `extract_docs_from_pdf.py` | Text and table extraction logic for source PDFs                        |
| `rag.py`                   | Multi-document RAG pipeline (chunking, embedding, indexing, retrieval) used by `app.py` |
| `rag_cli.py`               | Standalone single-document RAG pipeline with an interactive terminal chat loop |
| `app.py`                   | Flask app exposing `GET /` (renders UI with selected documents) and `POST /ask` (retrieval + generation endpoint) |

## Usage

### Web interface

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

The homepage (`GET /`) renders `index.html` along with the list of currently selected documents (via `select_all_docs()`). Questions are submitted through a JSON API:

**`POST /ask`**

```json
{ "question": "What are the payment terms in the contract?" }
```

Internally, this route:
1. Retrieves relevant parent-document context using the `ParentDocumentRetriever`
2. Joins the retrieved chunks into a single context block
3. Wraps the context and question in an instruction-style prompt (`[INST] ... [END]`, matching Mistral's chat format)
4. Passes the prompt to the local LLM via `ask()`
5. Converts the model's response into HTML paragraphs before returning it

**Response**

```json
{ "response": "<p>...</p><p>...</p>" }
```

`flask-cors` is enabled, so the API can be called from a separate frontend origin if needed.

### Command-line interface

```bash
python rag_cli.py
```

You'll be prompted to:
1. Select a single document from `docs/` (via `select_doc()`)
2. Select a model from the numbered menu (via `select_model()`)

Then ask questions directly in the terminal:

```
Your question please [Enter 'exit' to Exit]: What is the notice period for termination?
```

Type `exit` to end the session. This mode is useful for quickly testing retrieval quality or a new model against a single document without spinning up the web app.

## Project Structure

```
rag-model/
├── docs/                       # Source documents indexed by the app / offered by the CLI
├── rawdocs/                    # Raw/unprocessed source PDF documents
├── static/
│   └── images/                 # Static assets for the web UI
├── templates/                 
│   └── index.html              # Web UI HTML templates
├── .gitignore
├── app.py                      # Flask web application entry point
├── extract_docs_from_pdf.py    # PDF text and table extraction
├── models.py                   # GGUF model registry and selector
├── rag.py                      # Multi-document RAG pipeline (used by app.py)
├── rag_cli.py                  # Standalone single-document CLI with its own retriever/LLM
├── select_docs.py              # Document selection utilities (single-file and all-files)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
