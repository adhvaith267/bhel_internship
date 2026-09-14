"""Shared domain exceptions for DocuRAG."""

from __future__ import annotations


class DocuRAGError(Exception):
    """Base class for all DocuRAG errors."""


class IndexNotReadyError(DocuRAGError):
    """Raised when retrieval is attempted before the index is built."""


class DocumentNotFoundError(DocuRAGError):
    """Raised when a requested document filter matches nothing."""


class LLMProviderError(DocuRAGError):
    """Raised when all configured LLM providers fail."""
