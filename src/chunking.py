from src.models import OVERLAP, MAX_CHUNK_SIZE, HEADING_RE

Span = tuple[int, int, int]


def chunk_python_file(
    text: str,
    index: int,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Corta codigo em spans (id, inicio, fim), preferindo linhas em branco."""
    spans: list[tuple[int, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chunk_size, n)
        if end < n:
            # so corta numa fronteira depois de meio chunk, caso contrario
            # o avanco degenera em spans de poucos caracteres
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
) -> tuple[int, list[tuple[int, int, int]]]:
    """Corta texto em spans, priorizando fronteiras de seccao (##)."""
    boundaries = [m.start() for m in HEADING_RE.finditer(text)]
    boundaries.append(len(text))
    spans: list[tuple[int, int, int]] = []
    start = 0
    for boundary in boundaries:
        while boundary - start > max_chunk_size:
            cut = start + max_chunk_size
            # mesma guarda: uma quebra demasiado perto do inicio faria o
            # loop avancar 1 caracter de cada vez
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


def chunk_document(path: str, text: str, index: int) -> tuple[int, list[Span]]:
    """Despacha por tipo de ficheiro. Unico ponto que sabe a extensao."""
    if path.endswith(".py"):
        return chunk_python_file(text, index)
    return chunk_markdown_file(text, index)
