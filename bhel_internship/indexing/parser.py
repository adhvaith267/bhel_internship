import hashlib
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF

from bhel_internship.core.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCS_DIR,
    ENABLE_OCR,
    OCR_DPI,
    OCR_MIN_CHARS,
)
from bhel_internship.core.logger import logger
from bhel_internship.indexing.chunker import split_text_recursive

def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file to check for changes."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_text(text: str) -> str:
    """Normalizes excessive whitespace and clean line breaks while preserving paragraphs."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

def table_to_markdown(table_data: List[List[Any]]) -> str:
    """Converts raw table cell grid into a clean Markdown table format."""
    if not table_data or len(table_data) < 2:
        return ""

    rows = []
    for row in table_data:
        cleaned_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        if any(cleaned_row):
            rows.append(cleaned_row)

    if not rows:
        return ""

    header = rows[0]
    num_cols = len(header)
    col_widths = [max(len(header[i]), 3) for i in range(num_cols)]

    for row in rows[1:]:
        for i in range(min(len(row), num_cols)):
            col_widths[i] = max(col_widths[i], len(row[i]))

    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |"
    separator_str = "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |"

    body_lines = []
    for row in rows[1:]:
        padded = [row[i].ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i]) for i in range(num_cols)]
        body_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join([header_str, separator_str] + body_lines)

def detect_headings(page: "fitz.Page") -> List[str]:
    """Detects section headings via font-size analysis (no ML needed).

    Lines set materially larger than the page's median span size are
    treated as headings. Returns heading texts in reading order.
    """
    try:
        data = page.get_text("dict")
    except Exception:
        return []
    lines: List[Tuple[str, float]] = []
    sizes: List[float] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            size = max(float(s.get("size", 0)) for s in spans)
            sizes.append(size)
            if 3 <= len(text) <= 150:
                lines.append((text, size))
    if not lines or not sizes:
        return []
    sizes.sort()
    # Body estimate: 25th percentile (robust to a few large headings and a
    # few small footnote lines; the median fails on sparse pages).
    body = sizes[max(0, len(sizes) // 4 - (1 if len(sizes) < 4 else 0))]
    threshold = max(body * 1.25, body + 1.0)
    return [text for text, size in lines if size >= threshold]


@lru_cache(maxsize=1)
def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page(page: "fitz.Page", dpi: int = OCR_DPI) -> str:
    """Renders a page to an image and OCRs it via the tesseract CLI."""
    pix = page.get_pixmap(dpi=dpi)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pix.save(tmp.name)
        proc = subprocess.run(
            ["tesseract", tmp.name, "stdout", "-l", "eng"],
            capture_output=True,
            text=True,
            timeout=180,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.strip()[:200]}")
    return clean_text(proc.stdout)


def extract_pdf_documents(pdf_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts pages, tables, and structured hierarchical chunks from a PDF.
    Returns:
        chunks: List of chunk metadata dictionaries
        file_hash: SHA-256 hash of the PDF file
    """
    doc_name = pdf_path.name
    file_hash = compute_file_hash(pdf_path)
    logger.info(f"[bold cyan]Ingesting PDF:[/bold cyan] {doc_name} (hash: {file_hash[:8]}...)")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    all_chunks = []
    chunk_counter = 0
    current_section = ""
    blind_pages: List[int] = []
    ocr_pages: List[int] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # Track the most recent section heading (persists across pages).
        try:
            headings = detect_headings(page)
            if headings:
                current_section = headings[-1]
        except Exception as e:
            logger.warning(f"Heading detection failed on {doc_name} page {page_num}: {e}")

        # Extract page text, with OCR fallback for scanned/image-only pages.
        page_text = clean_text(page.get_text("text"))
        ocr_used = False
        if len(page_text) < OCR_MIN_CHARS:
            has_images = bool(page.get_images(full=True))
            if has_images and ENABLE_OCR and _tesseract_available():
                try:
                    ocr_text = ocr_page(page)
                    if len(ocr_text) > len(page_text):
                        page_text = ocr_text
                        ocr_used = True
                        ocr_pages.append(page_num)
                except Exception as e:
                    logger.warning(f"OCR failed on {doc_name} page {page_num}: {e}")

        # Extract tables
        tables_markdown = []
        try:
            tab_finder = page.find_tables()
            if tab_finder.tables:
                for tab in tab_finder.tables:
                    extracted_table = tab.extract()
                    md = table_to_markdown(extracted_table)
                    if md:
                        tables_markdown.append(md)
        except Exception as e:
            logger.warning(f"Could not extract tables from {doc_name} page {page_num}: {e}")

        # Pages with neither text nor tables are invisible to retrieval.
        if not page_text.strip() and not tables_markdown:
            blind_pages.append(page_num)

        # Parent context is the entire page content (section + text + markdown tables)
        combined_parent = page_text
        if current_section:
            combined_parent = f"Section: {current_section}\n\n{combined_parent}"
        if tables_markdown:
            combined_parent += "\n\n### Tables:\n" + "\n\n".join(tables_markdown)

        if not combined_parent.strip():
            continue

        # Split text into granular child chunks for dense and lexical search
        text_chunks = split_text_recursive(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chk in text_chunks:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": f"{pdf_path.stem}_p{page_num}_c{chunk_counter}",
                "doc_name": doc_name,
                "page": page_num,
                "content": chk,
                "parent_context": combined_parent,
                "section": current_section,
                "ocr": ocr_used,
                "is_table": False
            })

        # Add each table as an individual chunk with high semantic weight
        for t_idx, md_table in enumerate(tables_markdown):
            chunk_counter += 1
            table_summary = f"[Table from Page {page_num} of {doc_name}]\n{md_table}"
            all_chunks.append({
                "chunk_id": f"{pdf_path.stem}_p{page_num}_t{t_idx+1}",
                "doc_name": doc_name,
                "page": page_num,
                "content": table_summary,
                "parent_context": combined_parent,
                "section": current_section,
                "ocr": ocr_used,
                "is_table": True
            })

    doc.close()

    if ocr_pages:
        logger.info(f"OCR recovered text on {doc_name} pages: {ocr_pages}")
    if blind_pages:
        hint = "" if (not ENABLE_OCR or _tesseract_available()) else " (enable OCR: install tesseract)"
        logger.warning(
            f"{doc_name} has {len(blind_pages)} page(s) with no extractable text: "
            f"{blind_pages}{hint}. These pages are invisible to retrieval."
        )

    logger.info(f"Extracted [bold green]{len(all_chunks)} chunks[/bold green] across {total_pages} pages from {doc_name}")
    return all_chunks, file_hash

def get_all_pdf_paths() -> List[Path]:
    """Returns sorted list of all PDF file paths in DOCS_DIR."""
    return sorted(list(DOCS_DIR.glob("*.pdf")))
