import time
from typing import Dict, Any, List, Optional, Generator
from bhel_internship.retrieval.hybrid import HybridRAGRetriever
from bhel_internship.llm.provider import LLMEngine
from bhel_internship.llm.prompts import SYSTEM_PROMPT
from bhel_internship.core.config import ENABLE_ABSTENTION, ABSTAIN_MIN_SCORE, ENABLE_CORRECTIVE_RETRY
from bhel_internship.core.logger import logger
from bhel_internship.engine.grounding import (
    ABSTENTION_MESSAGE,
    build_clarification,
    empty_report,
    is_refusal,
    verify_citations,
)

# Appended to the prompt on corrective retry: forces strict citation
# discipline instead of letting the model restate an ungrounded answer.
STRICT_RETRY_ADDENDUM = (
    "\n\nIMPORTANT: Every factual claim must carry a citation "
    "[DocumentName, Page X] using ONLY the passages above. "
    "If the passages lack the answer, reply with exactly: "
    "The provided documents do not contain information to answer this question."
)

def format_context(chunks: List[Dict[str, Any]], expand_context: bool = True) -> str:
    """Formats retrieved chunks into clean, labeled context blocks for the LLM."""
    context_blocks = []
    seen_pages = set()

    for idx, chk in enumerate(chunks, 1):
        doc = chk.get("doc_name", "Document")
        page = chk.get("page", 1)
        score = chk.get("relevance_score", 0.0)
        
        # Use parent_context if available and not already shown for this page
        page_key = (doc, page)
        if expand_context and page_key not in seen_pages and chk.get("parent_context"):
            text = chk["parent_context"]
            seen_pages.add(page_key)
        else:
            text = chk.get("content", "")

        block = f"--- Context Passage {idx} [Doc: {doc} | Page: {page} | Score: {score:.2f}] ---\n{text.strip()}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

class EnterpriseRAGEngine:
    def __init__(self):
        logger.info("Initializing RAG Engine...")
        self.retriever = HybridRAGRetriever()
        self.llm = LLMEngine()
        self.is_ready = False

    def initialize(self):
        """Initializes hybrid index and models."""
        self.retriever.initialize()
        self.is_ready = True
        logger.info("[bold green]✓ RAG Engine ready for queries.[/bold green]")

    def reindex(self):
        """Forces full re-indexing of documents."""
        logger.info("Triggering full re-index...")
        self.retriever.build_index()
        self.is_ready = True

    def _query_is_specific(self, question: str) -> bool:
        """Delegates vague-query detection to the retriever's index stats.

        Fails open (specific) when the retriever cannot answer, so exotic
        retriever implementations keep the old behavior.
        """
        check = getattr(self.retriever, "is_specific_query", None)
        if not callable(check):
            return True
        try:
            return bool(check(question))
        except Exception:
            return True

    def _weak_evidence(self, chunks: List[Dict[str, Any]]) -> bool:
        """True when the best retrieval score is too low to answer from.

        Only applies when the cross-encoder reranker is active, since raw
        RRF scores are uncalibrated rank-fusion values, not relevance
        judgments. Compares against ABSTAIN_MIN_SCORE (reranker-logit space).
        """
        if not ENABLE_ABSTENTION or not chunks:
            return False
        if getattr(self.retriever, "reranker", None) is None:
            return False
        try:
            best = max(float(c.get("relevance_score", 0.0)) for c in chunks)
        except (TypeError, ValueError):
            return False
        return best < ABSTAIN_MIN_SCORE

    @staticmethod
    def _build_sources(chunks: List[Dict[str, Any]], snippet_len: int = 260) -> List[Dict[str, Any]]:
        return [
            {
                "document": chk.get("doc_name"),
                "page": chk.get("page"),
                "score": round(float(chk.get("relevance_score", 0.0)), 3),
                "snippet": chk.get("content", "")[:snippet_len] + "...",
                "is_table": chk.get("is_table", False),
            }
            for chk in chunks
        ]

    def query(
        self,
        question: str,
        top_n: int = 5,
        expand_context: bool = True,
        temperature: float = 0.2,
        doc_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline:
        1. Retrieve top-N re-ranked chunks.
        2. Abstain early when evidence is too weak to ground an answer.
        3. Format grounded prompt with source labels.
        4. Generate LLM response.
        5. Verify citations; one corrective retry if any are ungrounded.
        6. Package answer with structured citation + grounding metadata.
        """
        if not self.is_ready:
            self.initialize()

        t_start = time.time()
        question = question.strip()
        if not question:
            return {
                "response": "Please provide a valid question.",
                "sources": [],
                "latency_ms": 0,
                "model": self.llm.active_model_name,
                "abstained": True,
                "grounding": empty_report(),
            }

        # Vague input (greetings/gibberish) has nothing to retrieve on;
        # answer with clarification instead of a confident random summary.
        if not self._query_is_specific(question):
            return {
                "response": build_clarification(getattr(self.retriever, "chunks", [])),
                "sources": [],
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "model": self.llm.active_model_name,
                "abstained": True,
                "grounding": empty_report(),
            }

        # 1. Retrieve candidates
        chunks = self.retriever.retrieve(question, top_n=top_n, doc_name=doc_name)

        if not chunks:
            return {
                "response": "No relevant documents found to answer your question.",
                "sources": [],
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "model": self.llm.active_model_name,
                "abstained": True,
                "grounding": empty_report(),
            }

        # 2. Weak-evidence abstention gate (before spending LLM latency).
        if self._weak_evidence(chunks):
            best = max(float(c.get("relevance_score", 0.0)) for c in chunks)
            logger.info(
                f"[RAG.Pipeline] Abstaining: best retrieval score "
                f"({best:.2f}) below threshold ({ABSTAIN_MIN_SCORE})."
            )
            return {
                "response": ABSTENTION_MESSAGE,
                "sources": self._build_sources(chunks),
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "model": self.llm.active_model_name,
                "abstained": True,
                "grounding": empty_report(),
            }

        # 3. Build context
        context_str = format_context(chunks, expand_context=expand_context)

        # 4. Assemble prompt
        user_prompt = f"""Context:
{context_str}

Question:
{question}

Provide a comprehensive, accurate answer based solely on the context above.
Cite every factual claim using [DocumentName, Page X] format exactly as shown in the context passages."""

        # 5. Generate answer
        answer = self.llm.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature
        )

        # Model-side refusal: respect it, nothing to verify or retry.
        if is_refusal(answer):
            return self._package(
                answer, chunks, t_start, abstained=True, grounding=verify_citations(answer, chunks)
            )

        # 6. Verify citations; single corrective retry on ungrounded claims.
        grounding = verify_citations(answer, chunks)
        if ENABLE_CORRECTIVE_RETRY and grounding["unverified"]:
            logger.info(
                f"[RAG.Pipeline] {len(grounding['unverified'])} ungrounded citation(s); "
                "attempting one corrective retry."
            )
            retry_answer = self.llm.generate(
                prompt=user_prompt + STRICT_RETRY_ADDENDUM,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
            )
            if not is_refusal(retry_answer):
                retry_grounding = verify_citations(retry_answer, chunks)
                if retry_grounding["grounding_rate"] > grounding["grounding_rate"]:
                    answer, grounding = retry_answer, retry_grounding
            # A refusal on retry means the stricter prompt found the evidence
            # lacking; keep the original answer and its grounding report.

        return self._package(answer, chunks, t_start, abstained=False, grounding=grounding)

    def _package(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        t_start: float,
        abstained: bool,
        grounding: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Builds the standard query result dict with grounding metadata."""
        sources = self._build_sources(chunks)
        latency = round((time.time() - t_start) * 1000, 1)
        logger.info(f"[RAG.Pipeline] Generated response in {latency}ms ({len(sources)} sources cited)")
        return {
            "response": answer,
            "sources": sources,
            "latency_ms": latency,
            "model": self.llm.active_model_name,
            "abstained": abstained,
            "grounding": grounding,
        }

    def stream_query(
        self,
        question: str,
        top_n: int = 5,
        expand_context: bool = True,
        doc_name: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams generation chunks token-by-token along with citations.

        The no-evidence and weak-evidence abstention gates apply here too;
        citation verification/retry is a non-streaming concern, so streamed
        answers carry no grounding report.
        """
        if not self.is_ready:
            self.initialize()

        if not self._query_is_specific(question):
            yield {"type": "sources", "data": []}
            yield {"type": "token", "data": build_clarification(getattr(self.retriever, "chunks", []))}
            return

        chunks = self.retriever.retrieve(question, top_n=top_n, doc_name=doc_name)

        if not chunks:
            yield {"type": "sources", "data": []}
            yield {"type": "token", "data": "No relevant documents found to answer your question."}
            return

        if self._weak_evidence(chunks):
            yield {"type": "sources", "data": self._build_sources(chunks, snippet_len=200)}
            yield {"type": "token", "data": ABSTENTION_MESSAGE}
            return

        context_str = format_context(chunks, expand_context=expand_context)

        sources = self._build_sources(chunks, snippet_len=200)

        yield {"type": "sources", "data": sources}

        user_prompt = f"""Context:
{context_str}

Question:
{question}

Provide a comprehensive, accurate answer based solely on the context above.
Cite every factual claim using [DocumentName, Page X] format exactly as shown in the context passages."""

        for token in self.llm.stream_generate(user_prompt, system_prompt=SYSTEM_PROMPT):
            yield {"type": "token", "data": token}


# Global singleton
_engine_instance: Optional[EnterpriseRAGEngine] = None

def get_rag_engine() -> EnterpriseRAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EnterpriseRAGEngine()
        _engine_instance.initialize()
    return _engine_instance
