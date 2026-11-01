import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


class UnsupportedFileTypeError(ValueError):
    """The uploaded file's extension is not one we can parse."""


class DocumentParsingError(ValueError):
    """The file could not be read or contained no extractable text."""


def detect_content_type(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    try:
        return SUPPORTED_TYPES[suffix]
    except KeyError:
        allowed = ", ".join(sorted(SUPPORTED_TYPES))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or filename}'. Allowed: {allowed}"
        ) from None


def parse_document(filename: str, data: bytes) -> str:
    """Extract plain text from a PDF, Markdown or text file."""
    content_type = detect_content_type(filename)
    text = _parse_pdf(data) if content_type == "application/pdf" else _parse_text(data)
    if not text.strip():
        raise DocumentParsingError(f"No extractable text found in '{filename}'")
    return text


def _parse_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise DocumentParsingError(f"Invalid PDF: {exc}") from exc
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _parse_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParsingError("Text files must be UTF-8 encoded") from exc
