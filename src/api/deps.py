"""FastAPI dependencies (singleton engine injection)."""

from __future__ import annotations

from src.engine.pipeline import EnterpriseRAGEngine, get_rag_engine


def get_engine() -> EnterpriseRAGEngine:
    """Dependency returning the shared, already-initialized RAG engine."""
    return get_rag_engine()
