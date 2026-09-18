import os
import re


HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def print_output(full_path: dict) -> None:
    for key, value in full_path.items():
        print(f"\033[93mCamiho\033[0m: \033[95m{key}\033[0m")
        for path, data in value.items():
            print(f"\033[93mficheiro\033[0m: "
                  f"\033[42m{path}\033[0m\n\n", value[path])
    pass


def load_docs(folder: str) -> tuple[list, list, dict]:
    dir: list = []
    documents: list = []
    full_path: dict = {}

    for item in os.listdir(folder):
        if item.startswith("_") or item.startswith("."):
            continue
        path = os.path.join(folder, item)

        if os.path.isfile(path):
            if path.endswith((".py", ".md")):
                with open(path, "r", encoding="utf-8",
                          errors="ignore") as file:
                    data = file.read()
                    documents.append(data)
                full_path[path] = data
        elif os.path.isdir(path):
            dir.append(path)
            dir_ret, documents_ret, full_ret = load_docs(path)

            full_path.update(full_ret)
            dir.extend(dir_ret)
            documents.extend(documents_ret)
    return dir, documents, full_path


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


def chunk_markdown_file(text: str,
                        max_chunk_size: int = 2000,
                        overlap: int = 200) -> list[tuple[int, int]]:
    """Divide um ficheiro Markdown priorizando fronteiras de secção (##)."""
    boundaries = [m.start() for m in HEADING_RE.finditer(text)] + [len(text)]
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        while boundary - start > max_chunk_size:
            cut = start + max_chunk_size
            paragraph_break = text.rfind("\n\n", start, cut)
            cut = paragraph_break if paragraph_break > start else cut
            spans.append((start, cut))
            start = max(cut - overlap, start + 1)
        if boundary > start:
            spans.append((start, boundary))
            start = boundary
    return [s for s in spans if s[1] > s[0]]


def chunks(full_path: dict) -> tuple:

    py_chunk: list = []
    md_chunk: list = []
    for path, full in full_path.items():
        if path.endswith(".py"):
            py_chunk.append(chunk_python_file(full_path[path]))
        if path.endswith(".md"):
            md_chunk.append(chunk_markdown_file(full_path[path]))
    return (py_chunk, md_chunk)


dir, documents, full_path = load_docs("vllm-0.10.1")
py_chunk, md_chunk = chunks(full_path)



def index_bm25():
    pass

def index_tf_idf():
    pass
"""
"""
if __name__ == "__main__":
    try:
        dir, documents, full_path = load_docs("vllm-0.10.1")
        py_chunk, md_chunk = chunks(full_path)
        query = "torch.compile"
        result: dict = {}
        print("Chunks python", len(py_chunk))
        print("Chunks markdown", len(md_chunk))
        for path, full in full_path.items():
            if path.endswith(".md"):
                data = full_path[path]
                for md_docs in md_chunk:
                    for chunk in md_docs:
                        start, end = chunk
                        if query in data[start:end]:
                            result[query] = data[start:end]
        print(len(result))
        # for chunk in md_chunk:
            # print(chunk, "\n")
        # print_output(full_path)
    except Exception as e:
        print(f"Error: {e}")
