"""Central application settings.

Single source of truth for paths, retrieval tuning, model selection and
server options. Values can be overridden via environment variables or a
`.env` file at the repository root.

Module-level constants (``HOST``, ``PORT``, ...) are kept so existing
``from bhel_internship.core.config import HOST`` imports keep working.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repository root: bhel_internship/core/config.py -> core -> bhel_internship -> root.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Package directory (bhel_internship/): where ui/ lives after restructure.
_PKG_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


class Settings:
    """Application settings loaded from environment / `.env`."""

    def __init__(self) -> None:
        # --- Paths ---
        self.docs_dir = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "docs")))
        self.cache_dir = Path(os.getenv("CACHE_DIR", str(BASE_DIR / ".rag_cache")))
        self.models_dir = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
        self.static_dir = Path(os.getenv("STATIC_DIR", str(_PKG_DIR / "ui" / "static")))
        self.templates_dir = Path(os.getenv("TEMPLATES_DIR", str(_PKG_DIR / "ui" / "templates")))

        # --- Retrieval tuning ---
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "650"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.top_k_candidates = int(os.getenv("TOP_K_CANDIDATES", "20"))
        self.top_n_rerank = int(os.getenv("TOP_N_RERANK", "5"))
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.dense_weight = float(os.getenv("DENSE_WEIGHT", "0.6"))
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", "0.4"))
        # Weight of the cross-encoder score when fusing it with the RRF
        # score for final ordering (1.0 = cross-encoder only).
        self.rerank_fusion_alpha = float(os.getenv("RERANK_FUSION_ALPHA", "0.85"))
        # Maximal Marginal Relevance diversification of the final top-N.
        self.mmr_enabled = _get_bool("MMR_ENABLED", "true")
        self.mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.5"))

        # --- Faithfulness (abstention + corrective retry) ---
        self.enable_abstention = _get_bool("ENABLE_ABSTENTION", "true")
        # Minimum best cross-encoder score (raw-logit space) to attempt an
        # answer; only applies when the reranker is active. Logits are NOT
        # probabilities — ms-marco models routinely score relevant passages
        # around -5..-9, so keep this a backstop for catastrophic cases, not
        # a hair trigger. Recalibrate by watching the "Best:" value in the
        # [RAG.Reranker] log line for good queries on your corpus.
        self.abstain_min_score = float(os.getenv("ABSTAIN_MIN_SCORE", "-10.0"))
        self.enable_corrective_retry = _get_bool("ENABLE_CORRECTIVE_RETRY", "true")

        # --- Ingestion (OCR fallback via tesseract CLI, if installed) ---
        self.enable_ocr = _get_bool("ENABLE_OCR", "true")
        self.ocr_dpi = int(os.getenv("OCR_DPI", "300"))
        self.ocr_min_chars = int(os.getenv("OCR_MIN_CHARS", "50"))

        # --- Models ---
        self.reranker_model_name = os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.use_reranker = _get_bool("USE_RERANKER", "true")

        # --- LLM ---
        self.llm_provider = os.getenv("LLM_PROVIDER", "auto")  # auto | ollama | llamacpp
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.resolved_gguf_path = Path(
            os.getenv(
                "GGUF_MODEL_PATH",
                str(self.models_dir / "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"),
            )
        )

        _local_embedding = self.models_dir / "all-MiniLM-L6-v2"
        self.embedding_model_name = (
            str(_local_embedding)
            if _local_embedding.exists()
            else "sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- Server ---
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "5000"))
        self.debug = _get_bool("DEBUG", "false")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def ensure_dirs(self) -> None:
        for directory in (self.docs_dir, self.cache_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# --- Flat constants (same names as the original root config.py) ---
DOCS_DIR = settings.docs_dir
CACHE_DIR = settings.cache_dir
MODELS_DIR = settings.models_dir
STATIC_DIR = settings.static_dir
TEMPLATES_DIR = settings.templates_dir

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K_CANDIDATES = settings.top_k_candidates
TOP_N_RERANK = settings.top_n_rerank
RRF_K = settings.rrf_k
DENSE_WEIGHT = settings.dense_weight
BM25_WEIGHT = settings.bm25_weight
RERANK_FUSION_ALPHA = settings.rerank_fusion_alpha
MMR_ENABLED = settings.mmr_enabled
MMR_LAMBDA = settings.mmr_lambda
ENABLE_ABSTENTION = settings.enable_abstention
ABSTAIN_MIN_SCORE = settings.abstain_min_score
ENABLE_CORRECTIVE_RETRY = settings.enable_corrective_retry
ENABLE_OCR = settings.enable_ocr
OCR_DPI = settings.ocr_dpi
OCR_MIN_CHARS = settings.ocr_min_chars

LOCAL_EMBEDDING_DIR = settings.models_dir / "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = settings.embedding_model_name
RERANKER_MODEL_NAME = settings.reranker_model_name
USE_RERANKER = settings.use_reranker

LLM_PROVIDER = settings.llm_provider
OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model
GGUF_MODEL_PATH = settings.resolved_gguf_path
DEFAULT_GGUF_PATH = settings.models_dir / "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"

HOST = settings.host
PORT = settings.port
DEBUG = settings.debug
LOG_LEVEL = settings.log_level

__all__ = [
    "BASE_DIR",
    "settings",
    "Settings",
    "DOCS_DIR",
    "CACHE_DIR",
    "MODELS_DIR",
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K_CANDIDATES",
    "TOP_N_RERANK",
    "RRF_K",
    "DENSE_WEIGHT",
    "BM25_WEIGHT",
    "RERANK_FUSION_ALPHA",
    "MMR_ENABLED",
    "MMR_LAMBDA",
    "ENABLE_ABSTENTION",
    "ABSTAIN_MIN_SCORE",
    "ENABLE_CORRECTIVE_RETRY",
    "ENABLE_OCR",
    "OCR_DPI",
    "OCR_MIN_CHARS",
    "LOCAL_EMBEDDING_DIR",
    "EMBEDDING_MODEL_NAME",
    "RERANKER_MODEL_NAME",
    "USE_RERANKER",
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "GGUF_MODEL_PATH",
    "DEFAULT_GGUF_PATH",
    "HOST",
    "PORT",
    "DEBUG",
    "LOG_LEVEL",
]
