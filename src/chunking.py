from typing import Dict, List


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Split text into fixed-size character chunks
    with overlap between consecutive chunks.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_chunks(
    page_records: List[Dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Create chunks while preserving source and page metadata.
    """

    chunks = []

    chunk_counter = 0

    for record in page_records:
        text_chunks = chunk_text(
            text=record["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk in text_chunks:
            chunk_counter += 1

            chunks.append(
                {
                    "chunk_id": chunk_counter,
                    "source": record["source"],
                    "page": record["page"],
                    "text": chunk,
                }
            )

    return chunks