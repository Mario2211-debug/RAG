"""Two chunking strategies: one for code and one for prose."""

from src.config import HEADING_RE, MAX_CHUNK_SIZE, OVERLAP

Span = tuple[int, int, int]


def chunk_python_file(
    text: str,
    index: int,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> tuple[int, list[Span]]:
    """Split code into spans (id, start, end), preferring blank lines."""
    spans: list[Span] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chunk_size, n)
        if end < n:
            # only cut after the midpoint;
            # otherwise the overlap would degenerate
            # into very short spans
            floor = start + max_chunk_size // 2
            boundary = text.rfind("\n\n", start, end)
            if boundary <= floor:
                boundary = text.rfind("\n", start, end)
            if boundary > floor:
                end = boundary
        index += 1
        spans.append((index, start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return index, spans


def chunk_markdown_file(
    text: str,
    index: int,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> tuple[int, list[Span]]:
    """Split text into spans, prioritizing section boundaries (##)."""
    boundaries = [m.start() for m in HEADING_RE.finditer(text)]
    boundaries.append(len(text))
    spans: list[Span] = []
    start = 0
    for boundary in boundaries:
        while boundary - start > max_chunk_size:
            cut = start + max_chunk_size
            # same guard: a break too close to the start would make the loop
            # move forward one character at a time
            paragraph_break = text.rfind("\n\n", start, cut)
            if paragraph_break > start + max_chunk_size // 2:
                cut = paragraph_break
            index += 1
            spans.append((index, start, cut))
            start = max(cut - overlap, start + 1)
        if boundary > start:
            index += 1
            spans.append((index, start, boundary))
            start = boundary
    return index, [s for s in spans if s[2] > s[1]]


def chunk_document(
    path: str,
    text: str,
    index: int,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> tuple[int, list[Span]]:
    """Dispatch by file type.
    This is the only place that knows the extension."""
    if path.endswith(".py"):
        return chunk_python_file(text, index, max_chunk_size)
    return chunk_markdown_file(text, index, max_chunk_size)
