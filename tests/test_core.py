"""Tests for core utilities."""
import tempfile
from pathlib import Path

from bhel_internship.indexing.chunker import split_text_recursive
from bhel_internship.indexing.parser import clean_text, table_to_markdown, compute_file_hash
from bhel_internship.retrieval.hybrid import tokenize_bm25, expand_query_variants, _minmax_norm
from bhel_internship.engine.grounding import parse_citations, verify_citations, is_refusal, build_clarification


def test_clean_text():
    text = "  Hello   World  \n\n\n  Test  "
    assert clean_text(text) == "Hello World\n\nTest"


def test_table_to_markdown():
    table = [
        ["Name", "Age", "City"],
        ["Alice", "30", "NYC"],
        ["Bob", "25", "LA"],
    ]
    md = table_to_markdown(table)
    assert "| Name | Age | City |" in md
    assert "| Alice | 30 | NYC |" in md
    assert "| Bob | 25 | LA |" in md


def test_compute_file_hash():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content")
        path = Path(f.name)
    try:
        hash1 = compute_file_hash(path)
        hash2 = compute_file_hash(path)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
    finally:
        path.unlink()


def test_split_text_recursive():
    text = "Paragraph one.\n\nParagraph two with more content. " * 10
    chunks = split_text_recursive(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 250  # Allow some overflow due to overlap


def test_tokenize_bm25():
    tokens = tokenize_bm25("CS201 and Q8.0 and state-of-the-art")
    assert "cs201" in tokens
    assert "q8" in tokens
    assert "0" in tokens
    assert "state" in tokens
    assert "of" in tokens
    assert "the" in tokens
    assert "art" in tokens


def test_expand_query_variants():
    variants = expand_query_variants("pass-marks CS 201")
    assert "pass-marks CS 201" in variants
    assert "pass marks CS 201" in variants
    assert "pass-marks CS201" in variants


def test_minmax_norm():
    import numpy as np
    arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    normed = _minmax_norm(arr)
    assert normed[0] == 0.0
    assert normed[-1] == 1.0
    # Constant array
    const = _minmax_norm([5.0, 5.0, 5.0])
    assert all(c == 1.0 for c in const)


def test_parse_citations():
    text = "Answer [Doc1.pdf, Page 5] and [Doc2, Page 10]."
    citations = parse_citations(text)
    assert len(citations) == 2
    assert citations[0]["document"] == "Doc1.pdf"
    assert citations[0]["page"] == 5
    assert citations[1]["document"] == "Doc2"
    assert citations[1]["page"] == 10


def test_verify_citations():
    answer = "The grade is B [Syllabus.pdf, Page 12]."
    chunks = [
        {"doc_name": "Syllabus.pdf", "page": 12, "content": "Grade B is passing"},
        {"doc_name": "Other.pdf", "page": 5, "content": "Something else"},
    ]
    report = verify_citations(answer, chunks)
    assert report["cited_count"] == 1
    assert report["verified_count"] == 1
    assert report["grounding_rate"] == 1.0
    assert len(report["verified"]) == 1

    # Unverified citation
    answer2 = "Grade is A [Syllabus.pdf, Page 99]."
    report2 = verify_citations(answer2, chunks)
    assert report2["cited_count"] == 1
    assert report2["verified_count"] == 0
    assert report2["grounding_rate"] == 0.0
    assert len(report2["unverified"]) == 1


def test_is_refusal():
    assert is_refusal("The provided documents do not contain information to answer this question.")
    assert is_refusal("I cannot answer based on the context.")
    assert is_refusal("No relevant documents found.")
    assert not is_refusal("The answer is 42 [Doc.pdf, Page 1].")


def test_build_clarification():
    chunks = [
        {"doc_name": "Doc1.pdf", "section": "Introduction"},
        {"doc_name": "Doc2.pdf", "section": "Overview"},
        {"doc_name": "Doc1.pdf", "section": "Introduction"},
    ]
    msg = build_clarification(chunks)
    assert "Doc1.pdf" in msg
    assert "Doc2.pdf" in msg
    assert "Introduction" in msg or "Overview" in msg


def test_build_clarification_empty():
    msg = build_clarification([])
    assert "No documents are indexed" in msg