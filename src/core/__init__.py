"""Core shared primitives: settings, logging, exceptions."""

from src.core.config import settings
from src.core.exceptions import (
    EnterpriseRAGError,
    DocumentNotFoundError,
    IndexNotReadyError,
    LLMProviderError,
)
from src.core.logger import logger

__all__ = [
    "settings",
    "logger",
    "EnterpriseRAGError",
    "DocumentNotFoundError",
    "IndexNotReadyError",
    "LLMProviderError",
]
