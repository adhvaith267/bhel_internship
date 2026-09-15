"""Shared domain exceptions for Enterprise RAG Engine."""

from __future__ import annotations


class EnterpriseRAGError(Exception):
    """Base class for all Enterprise RAG Engine errors."""


class IndexNotReadyError(EnterpriseRAGError):
    """Raised when retrieval is attempted before the index is built."""


class DocumentNotFoundError(EnterpriseRAGError):
    """Raised when a requested document filter matches nothing."""


class LLMProviderError(EnterpriseRAGError):
    """Raised when all configured LLM providers fail."""
