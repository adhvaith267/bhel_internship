import json
import pickle
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch

from src.core.config import (
    CACHE_DIR,
    DOCS_DIR,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    USE_RERANKER,
    TOP_K_CANDIDATES,
    TOP_N_RERANK,
    RRF_K,
    DENSE_WEIGHT,
    BM25_WEIGHT,
    RERANK_FUSION_ALPHA,
    MMR_ENABLED,
    MMR_LAMBDA,
)
from src.indexing.parser import (
    extract_pdf_documents,
    get_all_pdf_paths,
    compute_file_hash,
)
from src.core.logger import logger


def tokenize_bm25(text: str) -> List[str]:
    """Lowercase alphanumeric tokens.

    Dots are handled both ways: ``U.S.A`` emits ``usa`` (acronym form)
    while ``Q8.0`` additionally emits ``q8`` + ``0`` (separator form, so
    it matches ``Q8_0`` / ``Q8 0``). Hyphens/underscores act as
    separators so ``state-of-the-art`` matches ``state of the art``.
    """
    lowered = text.lower()
    merged = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", lowered)
    split = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", " ", lowered)
    tokens = re.findall(r"[a-z0-9]+", merged)
    for tok in re.findall(r"[a-z0-9]+", split):
        if tok not in tokens:
            tokens.append(tok)
    return tokens


def expand_query_variants(query: str) -> List[str]:
    """Cheap multi-query expansion without an LLM round-trip.

    Returns the original query plus lightweight rewrites aimed at the
    failure modes of technical docs: hyphenated compounds (``pass-marks``
    vs ``pass marks``) and split alphanumeric codes (``CS 201`` vs
    ``CS201``). BM25 scores are max-fused across variants.
    """
    q = re.sub(r"\s+", " ", query.strip())
    if not q:
        return []
    variants = [q]
    dehyphen = re.sub(r"\s+", " ", q.replace("-", " ").replace("_", " ")).strip()
    if dehyphen.lower() != q.lower():
        variants.append(dehyphen)
    joined_codes = re.sub(r"\b([A-Za-z]+)\s+(\d+)\b", r"\1\2", q)
    if joined_codes != q:
        variants.append(joined_codes)
    seen, unique = set(), []
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def _minmax_norm(values: List[float]) -> np.ndarray:
    """Scale a score list to [0, 1]; constant input maps to all ones."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def _mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    top_n: int,
    lambda_: float,
) -> List[int]:
    """Greedy Maximal Marginal Relevance index selection.

    Picks ``top_n`` items balancing query relevance against redundancy
    with already-selected items. ``embeddings`` must be L2-normalized
    (cosine similarity == dot product). Returns positions in selection
    order, which is already a good display order.
    """
    n = int(embeddings.shape[0])
    top_n = max(0, min(top_n, n))
    if top_n == 0:
        return []
    lam = min(1.0, max(0.0, float(lambda_)))
    sims = embeddings @ embeddings.T
    selected: List[int] = [int(np.argmax(relevance))]
    while len(selected) < top_n:
        best_idx, best_val = -1, float("-inf")
        for idx in range(n):
            if idx in selected:
                continue
            redundancy = float(np.max(sims[idx, selected]))
            val = lam * float(relevance[idx]) - (1.0 - lam) * redundancy
            if val > best_val:
                best_val, best_idx = val, idx
        if best_idx < 0:
            break
        selected.append(best_idx)
    return selected


# Max characters of chunk content sent to the cross-encoder per pair.
# The model silently truncates past its token limit; truncating here keeps
# the head of each passage (where the topic sentence usually sits) instead
# of leaking the decision to tokenizer-specific tail truncation.
_RERANK_MAX_CHARS = 2000

# Bumped whenever the cached chunk schema or embedding contract changes;
# older caches are rebuilt instead of loaded.
SCHEMA_VERSION = 2

# Token shape that is always treated as meaningful (course/model codes like
# CS201 or Q8_0), regardless of corpus frequency. Structural, not a word list.
_CODE_SHAPE_RE = re.compile(r"[a-z]+\d+|\d+[a-z]+", re.IGNORECASE)


def _compute_term_df(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Counts in how many chunks each BM25 token appears (document frequency)."""
    df: Dict[str, int] = {}
    for chk in chunks:
        for token in set(tokenize_bm25(chk.get("content", ""))):
            df[token] = df.get(token, 0) + 1
    return df


class HybridRAGRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Retriever computing device: [bold yellow]{self.device}[/bold yellow]")

        # Load embedding model
        logger.info(f"Loading embedding model: [cyan]{EMBEDDING_MODEL_NAME}[/cyan]")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Load cross-encoder reranker if enabled
        self.reranker: Optional[CrossEncoder] = None
        if USE_RERANKER:
            try:
                logger.info(f"Loading cross-encoder reranker: [cyan]{RERANKER_MODEL_NAME}[/cyan]")
                self.reranker = CrossEncoder(
                    RERANKER_MODEL_NAME, device=self.device, max_length=512
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load CrossEncoder ({e}). Falling back to pure RRF hybrid retrieval."
                )
                self.reranker = None

        self.chunks: List[Dict[str, Any]] = []
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.manifest: Dict[str, Any] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.term_df: Dict[str, int] = {}

    def _get_cache_paths(self):
        return {
            "faiss": CACHE_DIR / "faiss.index",
            "bm25": CACHE_DIR / "bm25.pkl",
            "chunks": CACHE_DIR / "chunks.json",
            "manifest": CACHE_DIR / "manifest.json",
            "embeddings": CACHE_DIR / "embeddings.npy",
        }

    def _is_cache_valid(self, pdf_paths: List[Path]) -> bool:
        paths = self._get_cache_paths()
        if not all(p.exists() for p in paths.values()):
            return False

        try:
            with open(paths["manifest"], "r") as f:
                saved_manifest = json.load(f)

            if saved_manifest.get("schema_version") != SCHEMA_VERSION:
                return False
            if saved_manifest.get("embedding_model") != EMBEDDING_MODEL_NAME:
                return False

            current_hashes = {p.name: compute_file_hash(p) for p in pdf_paths}
            saved_hashes = saved_manifest.get("file_hashes", {})
            return current_hashes == saved_hashes
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def load_cache(self) -> bool:
        """Loads indexed FAISS, BM25, chunk metadata, and embeddings from disk."""
        paths = self._get_cache_paths()
        try:
            logger.info("Loading hybrid index from cache...")
            t0 = time.time()

            with open(paths["chunks"], "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            self.faiss_index = faiss.read_index(str(paths["faiss"]))

            with open(paths["bm25"], "rb") as f:
                self.bm25 = pickle.load(f)

            with open(paths["manifest"], "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

            self.embeddings = np.load(str(paths["embeddings"]))

            self.term_df = _compute_term_df(self.chunks)

            logger.info(
                f"[bold green]✓ Loaded {len(self.chunks)} chunks from cache in {time.time() - t0:.2f}s[/bold green]"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}. Reindexing required.")
            return False

    def save_cache(self, file_hashes: Dict[str, str], embeddings: np.ndarray):
        """Persists indexed structures (including chunk embeddings) to disk."""
        paths = self._get_cache_paths()
        try:
            logger.info("Saving hybrid index to disk cache...")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            with open(paths["chunks"], "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)

            faiss.write_index(self.faiss_index, str(paths["faiss"]))

            with open(paths["bm25"], "wb") as f:
                pickle.dump(self.bm25, f)

            np.save(str(paths["embeddings"]), np.asarray(embeddings, dtype=np.float32))

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "file_hashes": file_hashes,
                "num_chunks": len(self.chunks),
                "created_at": time.time(),
            }
            with open(paths["manifest"], "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            logger.info("[bold green]✓ Cache saved successfully.[/bold green]")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_reusable_state(self) -> Optional[Dict[str, Any]]:
        """Best-effort load of cached chunks+embeddings grouped by document.

        Returns ``{"by_doc": {doc_name: (chunks, embeddings)}, "hashes": ...}``
        or ``None`` when the cache is missing, foreign-schema, or was built
        with a different embedding model (full rebuild required then).
        """
        paths = self._get_cache_paths()
        try:
            if not all(p.exists() for p in paths.values()):
                return None
            with open(paths["manifest"], "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("schema_version") != SCHEMA_VERSION:
                return None
            if manifest.get("embedding_model") != EMBEDDING_MODEL_NAME:
                logger.info("Embedding model changed; full re-index required.")
                return None
            with open(paths["chunks"], "r", encoding="utf-8") as f:
                old_chunks = json.load(f)
            old_embeddings = np.load(str(paths["embeddings"]))
            if len(old_chunks) != len(old_embeddings):
                return None
            by_doc: Dict[str, Any] = {}
            for chk, emb in zip(old_chunks, old_embeddings):
                by_doc.setdefault(chk.get("doc_name"), ([], []))
                by_doc[chk.get("doc_name")][0].append(chk)
                by_doc[chk.get("doc_name")][1].append(emb)
            return {"by_doc": by_doc, "hashes": manifest.get("file_hashes", {})}
        except Exception as e:
            logger.warning(f"Could not reuse cached state ({e}); full re-index.")
            return None

    def build_index(self, pdf_paths: Optional[List[Path]] = None):
        """Extracts and indexes PDFs, reusing cached data for unchanged files.

        Only new or content-changed documents are re-extracted and
        re-encoded; chunks and embeddings of unchanged files are carried
        over, so adding one PDF no longer re-encodes the whole corpus.
        """
        if pdf_paths is None:
            pdf_paths = get_all_pdf_paths()

        if not pdf_paths:
            logger.warning(f"No PDF files found in {DOCS_DIR}. Index is empty.")
            self.chunks = []
            self.term_df = {}
            self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.bm25 = None
            return

        logger.info(f"Starting index build for {len(pdf_paths)} document(s)...")
        t_start = time.time()

        reusable = self._load_reusable_state()
        old_by_doc = reusable["by_doc"] if reusable else {}
        old_hashes = reusable["hashes"] if reusable else {}

        current_hashes = {p.name: compute_file_hash(p) for p in pdf_paths}

        reused_chunks: List[Dict[str, Any]] = []
        reused_embeddings: List[np.ndarray] = []
        fresh_paths: List[Path] = []
        for pdf_path in pdf_paths:
            name = pdf_path.name
            if name in old_by_doc and old_hashes.get(name) == current_hashes[name]:
                chunks, embs = old_by_doc[name]
                reused_chunks.extend(chunks)
                reused_embeddings.extend(embs)
            else:
                fresh_paths.append(pdf_path)

        if reused_chunks:
            logger.info(
                f"Reusing cached index for {len(pdf_paths) - len(fresh_paths)} "
                f"unchanged document(s) ({len(reused_chunks)} chunks)."
            )

        fresh_chunks: List[Dict[str, Any]] = []
        for pdf_path in fresh_paths:
            chunks, _ = extract_pdf_documents(pdf_path)
            fresh_chunks.extend(chunks)

        all_chunks = reused_chunks + fresh_chunks
        self.chunks = all_chunks
        if not self.chunks:
            logger.warning("No text chunks extracted from documents.")
            self.term_df = {}
            self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.bm25 = None
            return

        # 1. Build Dense FAISS Index (encode only fresh chunks)
        if fresh_chunks:
            logger.info(f"Encoding {len(fresh_chunks)} new chunks with {EMBEDDING_MODEL_NAME}...")
            fresh_embeddings = np.asarray(
                self.embedding_model.encode(
                    [c["content"] for c in fresh_chunks],
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
        else:
            fresh_embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)

        if reused_embeddings:
            embeddings = np.vstack(
                [np.asarray(reused_embeddings, dtype=np.float32), fresh_embeddings]
            )
        else:
            embeddings = fresh_embeddings
        self.embeddings = embeddings

        dense_index = faiss.IndexFlatIP(self.embedding_dim)
        dense_index.add(np.array(embeddings, dtype=np.float32))
        self.faiss_index = dense_index

        # 2. Build Sparse BM25 Index
        logger.info("Building BM25 lexical index...")
        tokenized_corpus = [tokenize_bm25(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Term document-frequencies for query-specificity checks.
        self.term_df = _compute_term_df(self.chunks)

        total_time = time.time() - t_start
        logger.info(
            f"[bold green]✓ Indexing completed: {len(self.chunks)} chunks in {total_time:.2f}s[/bold green]"
        )

        # Persist cache
        self.save_cache(current_hashes, embeddings)

    def initialize(self):
        """Initializes retriever from cache or rebuilds if needed."""
        pdf_paths = get_all_pdf_paths()
        if self._is_cache_valid(pdf_paths):
            if not self.load_cache():
                self.build_index(pdf_paths)
        else:
            logger.info("Cache is missing or stale. Rebuilding index...")
            self.build_index(pdf_paths)

    def is_specific_query(self, query: str) -> bool:
        """True when the query carries terms worth retrieving on.

        Specificity is measured against this index, not a word list: a
        token counts when it is code-shaped (CS201) or appears in
        some-but-not-most indexed chunks. Greetings ("yo"), typos
        ("hlo"), gibberish ("xyzabc"), and stopword-only queries fail
        every token and are therefore vague — whatever the corpus is.
        """
        tokens: set = set()
        for variant in expand_query_variants(query or ""):
            tokens.update(tokenize_bm25(variant))
        if not tokens or not self.chunks:
            return False
        ceiling = max(1, len(self.chunks) // 2)
        for token in tokens:
            if _CODE_SHAPE_RE.fullmatch(token):
                return True
            if 0 < self.term_df.get(token, 0) <= ceiling:
                return True
        return False

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CANDIDATES,
        top_n: int = TOP_N_RERANK,
        doc_name: Optional[str] = None,
        use_mmr: Optional[bool] = None,
        rerank_alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes two-stage retrieval:
        1. Dense semantic search (FAISS) + multi-variant lexical search
           (BM25) combined via RRF.
        2. Cross-encoder re-ranking fused with the RRF score, followed by
           MMR diversification of the final top-N.
        """
        query = (query or "").strip()
        if not query:
            return []
        if not self.chunks or self.faiss_index is None:
            logger.warning("Retriever has no indexed documents.")
            return []

        if use_mmr is None:
            use_mmr = MMR_ENABLED
        if rerank_alpha is None:
            rerank_alpha = RERANK_FUSION_ALPHA
        rerank_alpha = min(1.0, max(0.0, float(rerank_alpha)))

        # --- Stage 0: document pre-filter (before fusion, not after) ---
        # Filtering after RRF starves the candidate pool when scoped to one
        # document; restricting the eligible id set up-front keeps the full
        # top-k budget for the selected document.
        allowed_ids: Optional[set] = None
        if doc_name and doc_name.lower() != "all":
            allowed_ids = {i for i, c in enumerate(self.chunks) if c.get("doc_name") == doc_name}
            if not allowed_ids:
                logger.warning(f"No indexed chunks match document filter: {doc_name}")
                return []

        def _is_allowed(idx: int) -> bool:
            return allowed_ids is None or idx in allowed_ids

        t_retrieval_start = time.time()
        pool_size = len(allowed_ids) if allowed_ids is not None else len(self.chunks)
        k_candidates = min(top_k, pool_size)

        # --- Stage 1a: Dense Search (FAISS) ---
        # When filtering by document, search wider to ensure enough candidates survive the filter.
        search_k = (
            min(top_k * 4, len(self.chunks))
            if allowed_ids is not None
            else min(top_k, len(self.chunks))
        )
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True)
        _, dense_indices = self.faiss_index.search(
            np.array(query_vector, dtype=np.float32),
            search_k,
        )
        dense_ranked_ids = [
            int(idx) for idx in dense_indices[0] if idx >= 0 and _is_allowed(int(idx))
        ][:k_candidates]

        # --- Stage 1b: Lexical Search (BM25, max-fused over query variants) ---
        bm25_ranked_ids: List[int] = []
        if self.bm25:
            fused_bm25: Optional[np.ndarray] = None
            for variant in expand_query_variants(query):
                tokens = tokenize_bm25(variant)
                if not tokens:
                    continue
                scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
                fused_bm25 = scores if fused_bm25 is None else np.maximum(fused_bm25, scores)
            if fused_bm25 is not None:
                order = np.argsort(fused_bm25)[::-1]
                bm25_ranked_ids = [int(i) for i in order if _is_allowed(int(i))][:k_candidates]

        # --- Stage 1c: Reciprocal Rank Fusion (RRF) ---
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(dense_ranked_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (DENSE_WEIGHT / (RRF_K + rank + 1))

        for rank, idx in enumerate(bm25_ranked_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (BM25_WEIGHT / (RRF_K + rank + 1))

        # Carry (chunk, rrf) tuples together so scores can never misalign
        # with chunks through dedup / rerank / MMR.
        ranked: List[tuple] = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :k_candidates
        ]
        candidates: List[tuple] = [(self.chunks[idx], float(score)) for idx, score in ranked]

        # Drop exact-duplicate contents (chunk overlap can otherwise fill
        # top-N slots with near-identical text); keep the higher-RRF copy.
        seen_texts: set = set()
        deduped: List[tuple] = []
        for chunk, rrf in candidates:
            key = re.sub(r"\s+", " ", chunk.get("content", "").lower()).strip()
            if key and key in seen_texts:
                continue
            seen_texts.add(key)
            deduped.append((chunk, rrf))
        candidates = deduped

        retrieval_latency = (time.time() - t_retrieval_start) * 1000
        logger.info(
            f"[RAG.Retriever] Query: '{query[:45]}...' -> "
            f"Retrieved {len(candidates)} candidates via Hybrid RRF ({retrieval_latency:.1f}ms)"
        )
        if not candidates:
            return []

        # --- Stage 2: Neural Re-ranking fused with RRF ---
        if self.reranker:
            t_rerank_start = time.time()
            pairs = [(query, chk["content"][:_RERANK_MAX_CHARS]) for chk, _ in candidates]
            cross_scores = [float(s) for s in self.reranker.predict(pairs)]

            # Fuse: min-max normalize both signals, then blend. The raw
            # cross-encoder logit is kept as `relevance_score` for display;
            # the fused score only decides ordering.
            cross_norm = _minmax_norm(cross_scores)
            rrf_norm = _minmax_norm([rrf for _, rrf in candidates])
            order = sorted(
                range(len(candidates)),
                key=lambda i: (
                    rerank_alpha * float(cross_norm[i]) + (1.0 - rerank_alpha) * float(rrf_norm[i])
                ),
                reverse=True,
            )
            ordered = [(candidates[i][0], cross_scores[i]) for i in order]

            rerank_latency = (time.time() - t_rerank_start) * 1000
            top_score = ordered[0][1] if ordered else 0.0
            logger.info(
                f"[RAG.Reranker] Cross-encoder scored {len(candidates)} chunks -> "
                f"Top {min(top_n, len(ordered))} selected (Best: {top_score:.2f}) "
                f"({rerank_latency:.1f}ms)"
            )
            return self._finalize(ordered, top_n=top_n, use_mmr=use_mmr)

        # No reranker: return top-N from RRF.
        results = [(chk, rrf) for chk, rrf in candidates]
        return self._finalize(results, top_n=top_n, use_mmr=use_mmr)

    def _finalize(
        self,
        scored: List[tuple],
        top_n: int,
        use_mmr: bool,
    ) -> List[Dict[str, Any]]:
        """Applies MMR diversification (optional) and builds result dicts."""
        if not scored:
            return []
        if use_mmr and len(scored) > 1:
            try:
                embeddings = np.asarray(
                    self.embedding_model.encode(
                        [chk.get("content", "") for chk, _ in scored],
                        normalize_embeddings=True,
                    ),
                    dtype=np.float64,
                )
                relevance = _minmax_norm([score for _, score in scored])
                picks = _mmr_select(embeddings, relevance, min(top_n, len(scored)), MMR_LAMBDA)
                scored = [scored[i] for i in picks]
            except Exception as e:
                logger.warning(f"MMR diversification failed ({e}); using ranked order.")
                scored = scored[:top_n]
        else:
            scored = scored[:top_n]

        results = []
        for chk, score in scored:
            res = dict(chk)
            res["relevance_score"] = float(score)
            results.append(res)
        return results
