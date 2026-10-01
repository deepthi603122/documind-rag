import re
from pathlib import Path
from typing import Dict, List

import fitz
from docx import Document


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace,
    standalone page numbers, and empty lines.
    """

    lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Remove standalone page numbers
        if re.fullmatch(r"(page\s*)?\d+", line, flags=re.IGNORECASE):
            continue

        lines.append(line)

    return "\n".join(lines)


def extract_pdf(file_path: Path) -> List[Dict]:
    """
    Extract text from every PDF page.

    Returns one record per page so page-level source
    citations can be preserved.
    """

    pages = []

    document = fitz.open(file_path)

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            text = clean_text(text)

            if text:
                pages.append(
                    {
                        "source": file_path.name,
                        "page": page_number,
                        "text": text,
                    }
                )
    finally:
        document.close()

    return pages


def extract_txt(file_path: Path) -> List[Dict]:
    """
    Extract text from a TXT file.

    TXT files do not contain page information,
    so page is represented as 1.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    text = clean_text(text)

    if not text:
        return []

    return [
        {
            "source": file_path.name,
            "page": 1,
            "text": text,
        }
    ]


def extract_docx(file_path: Path) -> List[Dict]:
    """
    Extract text from a DOCX file.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    text = clean_text("\n".join(paragraphs))

    if not text:
        return []

    return [
        {
            "source": file_path.name,
            "page": 1,
            "text": text,
        }
    ]


def load_single_document(file_path: Path) -> List[Dict]:
    """Load one supported document."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    if extension == ".txt":
        return extract_txt(file_path)

    if extension == ".docx":
        return extract_docx(file_path)

    return []


def load_documents(data_dir: Path) -> List[Dict]:
    """
    Load all supported documents from the data directory.
    """

    supported_extensions = {".pdf", ".txt", ".docx"}

    files = sorted(
        [
            file
            for file in data_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in supported_extensions
        ]
    )

    if not files:
        raise FileNotFoundError(
            f"No PDF, TXT, or DOCX files found in {data_dir}"
        )

    all_pages = []

    print("\nLoading documents...")

    for file_path in files:
        pages = load_single_document(file_path)

        if not pages:
            print(f"  [WARNING] No text extracted: {file_path.name}")
            continue

        total_words = sum(
            len(page["text"].split())
            for page in pages
        )

        print(
            f"  [OK] {file_path.name} "
            f"({total_words} words, {len(pages)} page/section(s))"
        )

        if total_words < 500:
            print(
                "       WARNING: Assignment requires each document "
                "to be at least 500 words."
            )

        all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No readable document content was found.")

    print(
        f"\nLoaded {len(files)} document(s) "
        f"with {len(all_pages)} page/section records."
    )

    return all_pages