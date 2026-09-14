from typing import List
from bhel_internship.core.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text_recursive(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Splits text recursively based on paragraph, sentence, and word boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _split(txt: str, seps: List[str]) -> List[str]:
        if len(txt) <= chunk_size:
            return [txt] if txt.strip() else []
        if not seps:
            # Fallback hard chunking
            chunks = []
            start = 0
            while start < len(txt):
                end = min(start + chunk_size, len(txt))
                chunks.append(txt[start:end])
                start += chunk_size - chunk_overlap
            return chunks

        sep = seps[0]
        remaining_seps = seps[1:]
        parts = txt.split(sep)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, remaining_seps))
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current)

        # Re-introduce overlap
        merged = []
        for i, chk in enumerate(chunks):
            if i > 0 and chunk_overlap > 0:
                overlap_text = chunks[i - 1][-chunk_overlap:]
                combined = overlap_text + " " + chk
                merged.append(combined.strip())
            else:
                merged.append(chk.strip())
        return merged

    return _split(text, separators)
