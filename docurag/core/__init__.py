"""Core shared primitives: settings, logging, exceptions."""

from docurag.core.config import settings
from docurag.core.exceptions import (
    DocuRAGError,
    DocumentNotFoundError,
    IndexNotReadyError,
    LLMProviderError,
)
from docurag.core.logger import logger

__all__ = [
    "settings",
    "logger",
    "DocuRAGError",
    "DocumentNotFoundError",
    "IndexNotReadyError",
    "LLMProviderError",
]
