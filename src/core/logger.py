"""Structured Rich logging for Enterprise RAG Engine."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(name: str = "EnterpriseRAG") -> logging.Logger:
    """Create (or reuse) a Rich-formatted logger honoring ``LOG_LEVEL``."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False
    else:
        logger.setLevel(level)
    return logger


logger = setup_logger()

__all__ = ["console", "logger", "setup_logger"]
