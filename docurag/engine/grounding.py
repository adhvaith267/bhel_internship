"""Citation grounding helpers (pure functions, no model calls).

The engine instructs the LLM to cite sources as ``[DocumentName, Page X]``.
These helpers parse those markers back out of generated answers and check
them against the retrieved chunks, so the pipeline can abstain or retry
instead of serving ungrounded claims.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

# Matches "[Some Doc.pdf, Page 12]" / "[Some Doc, page 3]".
CITATION_RE = re.compile(r"\[([^,\[\]]+?),\s*[Pp]age\s*(\d+)\]")

# Phrases indicating the model itself refused for lack of evidence.
_REFUSAL_PHRASES = (
    "do not contain information",
    "don't contain information",
    "does not contain information",
    "doesn't contain information",
    "do not contain sufficient",
    "insufficient information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "no relevant documents found",
)

ABSTENTION_MESSAGE = (
    "The retrieved passages do not provide sufficient grounded evidence "
    "to answer this question."
)


def build_clarification(chunks: List[Dict[str, Any]]) -> str:
    """Builds a vague-query response from the actually-indexed documents.

    Names the indexed files and suggests their most-chunked sections as
    example topics, so the message is never hardcoded to one corpus.
    """
    docs = sorted({str(c.get("doc_name", "")) for c in chunks if c.get("doc_name")})
    if not docs:
        return "No documents are indexed yet \u2014 add PDFs to the docs folder, then ask away."
    doc_part = ", ".join(docs[:3])
    if len(docs) > 3:
        doc_part += f" (+{len(docs) - 3} more)"
    sections = [
        section
        for section, _ in Counter(
            str(c.get("section", "")).strip()
            for c in chunks
            if str(c.get("section", "")).strip()
        ).most_common(2)
    ]
    if sections:
        topics = " or ".join(f"\u201c{s}\u201d" for s in sections)
        return (
            f"I can only answer questions about your indexed documents "
            f"\u2014 currently: {doc_part}. Try asking about {topics}."
        )
    return (
        f"I can only answer questions about your indexed documents "
        f"\u2014 currently: {doc_part}."
    )

def _normalize_doc(name: str) -> str:
    """Normalize a document name for comparison (case/extension tolerant)."""
    cleaned = name.strip().strip("\"'").lower()
    if cleaned.endswith(".pdf"):
        cleaned = cleaned[: -len(".pdf")]
    return cleaned


def parse_citations(answer: str) -> List[Dict[str, Any]]:
    """Extract ``[document, page]`` citation markers from an answer."""
    found = []
    for match in CITATION_RE.finditer(answer or ""):
        found.append(
            {
                "document": match.group(1).strip(),
                "page": int(match.group(2)),
                "span": match.group(0),
            }
        )
    return found


def verify_citations(
    answer: str, chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Check every cited (document, page) pair against retrieved chunks.

    Returns a grounding report with ``cited_count``, ``verified_count``,
    ``grounding_rate`` (1.0 when nothing was cited), plus the ``verified``
    and ``unverified`` citation lists.
    """
    citations = parse_citations(answer)
    evidence = {
        (_normalize_doc(str(chk.get("doc_name", ""))), int(chk.get("page", -1)))
        for chk in chunks
    }
    verified, unverified = [], []
    for cite in citations:
        key = (_normalize_doc(cite["document"]), cite["page"])
        (verified if key in evidence else unverified).append(
            {"document": cite["document"], "page": cite["page"]}
        )
    cited = len(citations)
    return {
        "cited_count": cited,
        "verified_count": len(verified),
        "grounding_rate": (len(verified) / cited) if cited else 1.0,
        "verified": verified,
        "unverified": unverified,
    }


def empty_report() -> Dict[str, Any]:
    """Grounding report for answers with no citations to check."""
    return {
        "cited_count": 0,
        "verified_count": 0,
        "grounding_rate": 1.0,
        "verified": [],
        "unverified": [],
    }


def is_refusal(answer: str) -> bool:
    """Detects model abstention phrasing (lack-of-evidence refusal)."""
    lowered = (answer or "").lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)
