"""Goldens-based RAG evaluation: retrieval hits, answer coverage, grounding.

Goldens file format (JSON list of cases)::

    [
      {
        "question": "What is the minimum passing grade?",
        "doc_name": null,
        "expected_doc": "syllabus.pdf",
        "expected_page": 4,
        "expected_keywords": ["passing", "grade B"],
        "top_n": 5
      }
    ]

``expected_page`` may be ``null`` for doc-level-only checks. Each case runs
the full engine pipeline, so the report reflects what users actually get:
the sources shown to the LLM, the generated answer, and its grounding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_doc(name: Optional[str]) -> str:
    cleaned = (name or "").strip().strip("\"'").lower()
    if cleaned.endswith(".pdf"):
        cleaned = cleaned[: -len(".pdf")]
    return cleaned


def load_goldens(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError(f"Goldens file must contain a JSON list: {path}")
    return cases


def evaluate_case(
    engine: Any, case: Dict[str, Any], keyword_threshold: float = 0.5
) -> Dict[str, Any]:
    """Runs one golden case through the full pipeline and scores it."""
    question = case["question"]
    top_n = int(case.get("top_n", 5))
    t0 = time.time()
    result = engine.query(question, top_n=top_n, doc_name=case.get("doc_name"))
    eval_latency = round((time.time() - t0) * 1000, 1)

    sources = result.get("sources", [])
    expected_doc = _normalize_doc(case.get("expected_doc"))
    expected_page = case.get("expected_page")

    if expected_doc:
        matching = [s for s in sources if _normalize_doc(s.get("document")) == expected_doc]
        if expected_page is None:
            context_hit = bool(matching)
        else:
            context_hit = any(int(s.get("page", -1)) == int(expected_page) for s in matching)
    else:
        context_hit = True  # no retrieval expectation for this case

    keywords = [k.lower() for k in case.get("expected_keywords", [])]
    answer_lower = (result.get("response", "") or "").lower()
    hits = [k for k in keywords if k in answer_lower]
    coverage = (len(hits) / len(keywords)) if keywords else 1.0

    grounding = result.get("grounding") or {}
    grounding_rate = float(grounding.get("grounding_rate", 1.0))
    abstained = bool(result.get("abstained", False))

    passed = bool(context_hit and coverage >= keyword_threshold and not abstained)

    return {
        "question": question,
        "context_hit": context_hit,
        "keyword_hits": hits,
        "keyword_coverage": round(coverage, 3),
        "grounding_rate": grounding_rate,
        "abstained": abstained,
        "latency_ms": eval_latency,
        "passed": passed,
    }


def run_eval(
    engine: Any,
    goldens: List[Dict[str, Any]],
    keyword_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Runs all golden cases; returns per-case results plus a summary."""
    cases = [evaluate_case(engine, c, keyword_threshold) for c in goldens]
    n = len(cases)
    summary = {
        "total": n,
        "passed": sum(1 for c in cases if c["passed"]),
        "pass_rate": round(sum(1 for c in cases if c["passed"]) / n, 3) if n else 0.0,
        "context_hit_rate": round(sum(1 for c in cases if c["context_hit"]) / n, 3) if n else 0.0,
        "mean_keyword_coverage": round(sum(c["keyword_coverage"] for c in cases) / n, 3)
        if n
        else 0.0,
        "mean_grounding_rate": round(sum(c["grounding_rate"] for c in cases) / n, 3) if n else 0.0,
        "abstentions": sum(1 for c in cases if c["abstained"]),
    }
    return {"cases": cases, "summary": summary}
