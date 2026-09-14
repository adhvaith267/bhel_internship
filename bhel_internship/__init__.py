"""DocuRAG — local, privacy-preserving Retrieval-Augmented Generation engine."""

__version__ = "2.0.0"

__all__ = ["__version__", "DocuRAGEngine", "get_rag_engine"]


def __getattr__(name: str):
    # Lazy re-exports so `import bhel_internship` never pulls heavy deps
    # (torch / faiss / sentence-transformers) unless explicitly requested.
    if name in {"DocuRAGEngine", "get_rag_engine"}:
        from bhel_internship.engine.pipeline import DocuRAGEngine, get_rag_engine

        return {"DocuRAGEngine": DocuRAGEngine, "get_rag_engine": get_rag_engine}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
