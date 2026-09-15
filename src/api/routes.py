"""FastAPI application factory and route definitions."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.deps import get_engine
from src.api.schemas import AskRequest, ChatRequest
from src.core.config import settings
from src.core.logger import logger
from src.engine.pipeline import EnterpriseRAGEngine
from src.indexing.parser import get_all_pdf_paths


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("[bold cyan]Starting Enterprise RAG Engine server...[/bold cyan]")
    get_engine()  # warm up retriever + LLM on startup
    yield
    logger.info("[bold yellow]Shutting down Enterprise RAG Engine server...[/bold yellow]")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (import-safe factory)."""
    application = FastAPI(
        title="Enterprise RAG Engine API",
        description="Local RAG API: hybrid retrieval, neural re-ranking, grounded answers.",
        version="2.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.static_dir.exists():
        application.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    templates = Jinja2Templates(directory=str(settings.templates_dir))

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request, engine: EnterpriseRAGEngine = Depends(get_engine)):
        pdf_files = [p.name for p in get_all_pdf_paths()]
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "selected_documents": pdf_files,
                "model_name": engine.llm.active_model_name,
            },
            # Never cache the SPA shell: the UI iterates fast and must
            # always match the running backend.
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/ask")
    async def ask_endpoint(payload: AskRequest, engine: EnterpriseRAGEngine = Depends(get_engine)):
        question = payload.question.strip()
        if question.lower() == "exit" or not question:
            return {
                "response": "Session ended or invalid input.",
                "sources": [],
                "latency_ms": 0,
                "model": engine.llm.active_model_name,
                "abstained": True,
                "grounding": {},
            }

        result = engine.query(question=question, top_n=payload.top_n, doc_name=payload.doc_name)
        return {
            "response": result["response"],
            "raw_text": result["response"],
            "sources": result["sources"],
            "latency_ms": result["latency_ms"],
            "model": result["model"],
            "abstained": result.get("abstained", False),
            "grounding": result.get("grounding", {}),
        }

    @application.post("/api/chat")
    async def chat_api(payload: ChatRequest, engine: EnterpriseRAGEngine = Depends(get_engine)):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        if payload.stream:

            async def event_generator():
                for event in engine.stream_query(
                    question, top_n=payload.top_n, doc_name=payload.doc_name
                ):
                    yield f"data: {json.dumps(event)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        return engine.query(question=question, top_n=payload.top_n, doc_name=payload.doc_name)

    @application.get("/api/documents")
    async def list_documents(engine: EnterpriseRAGEngine = Depends(get_engine)):
        pdf_paths = get_all_pdf_paths()
        docs_info = []
        for p in pdf_paths:
            stat = p.stat()
            doc_chunks = sum(1 for c in engine.retriever.chunks if c.get("doc_name") == p.name)
            docs_info.append(
                {
                    "name": p.name,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "chunks_count": doc_chunks,
                    "modified_time": time.ctime(stat.st_mtime),
                }
            )
        return {
            "documents": docs_info,
            "total_documents": len(docs_info),
            "total_indexed_chunks": len(engine.retriever.chunks),
        }

    @application.post("/api/reindex")
    async def reindex_documents(engine: EnterpriseRAGEngine = Depends(get_engine)):
        t0 = time.time()
        engine.reindex()
        return {
            "status": "success",
            "message": f"Successfully reindexed {len(engine.retriever.chunks)} chunks.",
            "chunks_count": len(engine.retriever.chunks),
            "time_taken_seconds": round(time.time() - t0, 2),
        }

        t0 = time.time()
        engine.reindex()
        return {
            "status": "success",
            "message": f"Successfully reindexed {len(engine.retriever.chunks)} chunks.",
            "chunks_count": len(engine.retriever.chunks),
            "time_taken_seconds": round(time.time() - t0, 2),
        }

    @application.get("/api/status")
    @application.get("/api/health")
    async def health_check(engine: EnterpriseRAGEngine = Depends(get_engine)):
        return {
            "status": "healthy",
            "backend": "FastAPI",
            "active_model": engine.llm.active_model_name,
            "provider": engine.llm.provider,
            "indexed_chunks": len(engine.retriever.chunks),
            "retriever_device": engine.retriever.device,
            "reranker_active": engine.retriever.reranker is not None,
        }


# Default application instance for `uvicorn main:app` / `uvicorn src.api.routes:app`.
app = create_app()
