import pytest

from app.services.parsing import (
    DocumentParsingError,
    UnsupportedFileTypeError,
    detect_content_type,
    parse_document,
)
from tests.conftest import make_pdf


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.md", "text/markdown"),
        ("README.MARKDOWN", "text/markdown"),
        ("report.pdf", "application/pdf"),
        ("log.txt", "text/plain"),
    ],
)
def test_detect_content_type(filename: str, expected: str):
    assert detect_content_type(filename) == expected


@pytest.mark.parametrize("filename", ["image.png", "archive.zip", "no_extension"])
def test_unsupported_types_are_rejected(filename: str):
    with pytest.raises(UnsupportedFileTypeError):
        detect_content_type(filename)


def test_parse_markdown_keeps_text():
    text = parse_document("doc.md", "# Título\n\nConteúdo em português.".encode())
    assert "Conteúdo em português." in text


def test_parse_strips_utf8_bom():
    assert parse_document("doc.txt", b"\xef\xbb\xbfhello") == "hello"


def test_parse_non_utf8_text_fails():
    with pytest.raises(DocumentParsingError, match="UTF-8"):
        parse_document("doc.txt", "café".encode("utf-16"))


def test_parse_empty_markdown_fails():
    with pytest.raises(DocumentParsingError, match="No extractable text"):
        parse_document("empty.md", b"   \n ")


def test_parse_pdf_extracts_text():
    text = parse_document("doc.pdf", make_pdf("Retrieval Augmented Generation"))
    assert "Retrieval Augmented Generation" in text


def test_parse_invalid_pdf_fails():
    with pytest.raises(DocumentParsingError):
        parse_document("broken.pdf", b"this is not a pdf")
