"""Read a PDF and split it into chunks."""
from pypdf import PdfReader

import config


def read_pdf(file) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages that contain text."""
    reader = PdfReader(file)
    pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    return [(n, t) for n, t in pages if t]


def chunk_pages(pages, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """Split page text into overlapping chunks: [(page_number, chunk_text), ...]."""
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            chunks.append((page_no, text[start:start + size]))
            start += size - overlap
    return chunks