"""Core shared primitives: settings, logging, exceptions."""

from bhel_internship.core.config import settings
from bhel_internship.core.exceptions import (
    EnterpriseRAGError,
    DocumentNotFoundError,
    IndexNotReadyError,
    LLMProviderError,
)
from bhel_internship.core.logger import logger

__all__ = [
    "settings",
    "logger",
    "EnterpriseRAGError",
    "DocumentNotFoundError",
    "IndexNotReadyError",
    "LLMProviderError",
]
