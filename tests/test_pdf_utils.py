from core.pdf_utils import chunk_pages


def test_chunks_overlap_and_keep_page_numbers():
    pages = [(1, "a" * 2500), (2, "b" * 100)]
    chunks = chunk_pages(pages, size=1000, overlap=100)
    assert chunks[0][0] == 1 and len(chunks[0][1]) == 1000
    assert chunks[-1][0] == 2
    assert all(len(t) <= 1000 for _, t in chunks)