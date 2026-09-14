"""Enterprise RAG Engine — local, privacy-preserving Retrieval-Augmented Generation engine."""

__version__ = "2.0.0"

__all__ = ["__version__", "EnterpriseRAGEngine", "get_rag_engine"]


def __getattr__(name: str):
    # Lazy re-exports so `import bhel_internship` never pulls heavy deps
    # (torch / faiss / sentence-transformers) unless explicitly requested.
    if name in {"EnterpriseRAGEngine", "get_rag_engine"}:
        from bhel_internship.engine.pipeline import EnterpriseRAGEngine, get_rag_engine

        return {"EnterpriseRAGEngine": EnterpriseRAGEngine, "get_rag_engine": get_rag_engine}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
