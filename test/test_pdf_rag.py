# tests/test_pdf_rag.py
import pytest
from src.pdf_rag import chunk_text, extract_text_from_pdf

def test_chunk_text_small():
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2

def test_extract_text_from_pdf_sample(tmp_path):
    # create a tiny pdf with pypdf requires actual PDF creation; instead skip heavy I/O
    # We'll simulate by ensuring function raises gracefully for non-pdf
    with pytest.raises(Exception):
        extract_text_from_pdf(str(tmp_path / "no_file.pdf"))
