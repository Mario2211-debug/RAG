random_list = ["Mario Afonso", "Daniel Ximenes", "Leo Buscaro", "Gabriel La Rocque"]


def chunk_python_file(text: str,
                      max_chunk_size: int = 2000,
                      overlap: int = 200) -> list[tuple[int, int]]:
    """Devolve uma lista de (first_index, last_index) para um ficheiro .py.

    Tenta cortar em linhas em branco ou no início de `def`/`class` quando
    o chunk se aproxima do tamanho máximo, em vez de cortar a meio de uma
    linha de código.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chunk_size, n)
        if end < n:
            # procura o último salto de linha "seguro" antes do limite
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind("\n", start, end)
            if boundary > start:
                end = boundary
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans


print(chunk_python_file("def foo():\n    pass\n"))