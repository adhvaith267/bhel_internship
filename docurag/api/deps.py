"""FastAPI dependencies (singleton engine injection)."""

from __future__ import annotations

from functools import lru_cache

from docurag.engine.pipeline import DocuRAGEngine, get_rag_engine


@lru_cache(maxsize=1)
def _cached_engine() -> DocuRAGEngine:
    return get_rag_engine()


def get_engine() -> DocuRAGEngine:
    """Dependency returning the shared, already-initialized RAG engine."""
    return _cached_engine()
