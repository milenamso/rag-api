import re

# Preferred break points, from strongest (paragraph) to weakest (word).
_SEPARATORS = ("\n\n", "\n", ". ", " ")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks of at most ``chunk_size`` characters.

    Each chunk ends at the strongest natural break point (paragraph, line,
    sentence, word) found in the second half of the window, so chunks rarely
    cut words or sentences in half.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = _EXCESS_NEWLINES.sub("\n\n", text.replace("\r\n", "\n")).strip()
    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            for separator in _SEPARATORS:
                cut = window.rfind(separator)
                if cut > chunk_size // 2:
                    end = start + cut + len(separator)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(end - chunk_overlap, start + 1)
        # Start the overlap at a word boundary rather than mid-word.
        if next_start < end and not text[next_start - 1].isspace():
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start

    return chunks
