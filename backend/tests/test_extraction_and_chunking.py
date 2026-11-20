"""Text extraction + chunking."""
from __future__ import annotations

import pytest

from app.core.errors import ExtractionError, UnsupportedFileError
from app.rag.chunking import chunk_pages, chunk_text
from app.services.text_extraction import ExtractedPage, get_extractor
from tests.conftest import make_pdf


def test_pdf_extraction_keeps_page_numbers(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["First page text", "Second page text\nline two"])
    result = get_extractor("a.pdf").extract(pdf)
    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "Second page text" in result.pages[1].text
    assert "line two" in result.pages[1].text


def test_pdf_with_blank_page_warns_but_succeeds(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["Real text here", "", "More text"])
    result = get_extractor("a.pdf").extract(pdf)
    assert result.page_count == 3
    assert [p.page_number for p in result.pages] == [1, 3]
    assert any("no extractable text" in w for w in result.warnings)


def test_pdf_with_no_text_returns_no_pages(tmp_path):
    pdf = make_pdf(tmp_path / "blank.pdf", ["", ""])
    result = get_extractor("blank.pdf").extract(pdf)
    assert result.pages == []


def test_corrupted_pdf_raises_extraction_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is definitely not a real pdf body \x00\x01\x02")
    with pytest.raises(ExtractionError):
        get_extractor("bad.pdf").extract(bad)


def test_non_pdf_content_with_pdf_extension_rejected(tmp_path):
    bad = tmp_path / "fake.pdf"
    bad.write_bytes(b"hello, I am plain text")
    with pytest.raises(ExtractionError, match="not a valid PDF"):
        get_extractor("fake.pdf").extract(bad)


def test_txt_extraction_utf8_and_cp1252(tmp_path):
    utf = tmp_path / "u.txt"
    utf.write_text("Price: ₹5,000\nSecond line", encoding="utf-8")
    assert "₹5,000" in get_extractor("u.txt").extract(utf).pages[0].text
    legacy = tmp_path / "l.txt"
    legacy.write_bytes("caf\xe9 menu".encode("cp1252"))
    assert "caf\xe9" in get_extractor("l.txt").extract(legacy).pages[0].text


def test_txt_has_no_page_number(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    page = get_extractor("a.txt").extract(f).pages[0]
    assert page.page_number is None


def test_empty_txt_yields_no_pages(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("   \n\n  ")
    assert get_extractor("e.txt").extract(f).pages == []


def test_binary_file_as_txt_rejected(tmp_path):
    f = tmp_path / "b.txt"
    f.write_bytes(b"\x00\x01\x02\x03binary" * 50)
    with pytest.raises(ExtractionError):
        get_extractor("b.txt").extract(f)


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        get_extractor("slides.pptx")
    with pytest.raises(UnsupportedFileError):
        get_extractor("noextension")


# ------------------------------------------------------------ chunking ----
def test_chunks_respect_size_limit():
    text = "\n".join(f"Line number {i} with some filler words to take space." for i in range(80))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_cover_all_content():
    lines = [f"unique-token-{i}" for i in range(60)]
    chunks = chunk_text("\n".join(lines), chunk_size=200, overlap=30)
    joined = "\n".join(chunks)
    assert all(token in joined for token in lines)


def test_chunk_overlap_repeats_trailing_lines():
    lines = [f"line-{i:02d} " + "x" * 30 for i in range(30)]
    chunks = chunk_text("\n".join(lines), chunk_size=200, overlap=80)
    first_last_line = chunks[0].split("\n")[-1]
    assert first_last_line in chunks[1]


def test_no_overlap_when_zero():
    lines = [f"line-{i:02d} " + "x" * 30 for i in range(30)]
    chunks = chunk_text("\n".join(lines), chunk_size=200, overlap=0)
    assert chunks[0].split("\n")[-1] not in chunks[1]


def test_very_long_line_is_split():
    chunks = chunk_text("word " * 1000, chunk_size=200, overlap=20)
    assert len(chunks) > 10
    assert all(len(c) <= 200 for c in chunks)


def test_long_unbroken_token_is_split():
    chunks = chunk_text("x" * 1000, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_empty_text_gives_no_chunks():
    assert chunk_text("  \n \n", 200, 20) == []


def test_chunk_pages_never_cross_pages_and_index_is_global():
    pages = [ExtractedPage(1, "alpha " * 100), ExtractedPage(2, "beta " * 100), ExtractedPage(None, "gamma")]
    drafts = chunk_pages(pages, chunk_size=200, overlap=20)
    assert [d.chunk_index for d in drafts] == list(range(len(drafts)))
    for d in drafts:
        if d.page_number == 1:
            assert "beta" not in d.text
        if d.page_number == 2:
            assert "alpha" not in d.text
    assert drafts[-1].page_number is None


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=0)
