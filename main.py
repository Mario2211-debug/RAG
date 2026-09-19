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

    py_chunk: dict = {}
    md_chunk: dict = {}
    for path, full in full_path.items():
        if path.endswith(".py"):
            py_chunk[path] = (chunk_python_file(full_path[path]))
        if path.endswith(".md"):
            md_chunk[path] = (chunk_markdown_file(full_path[path]))
    return (py_chunk, md_chunk)

def search_query_in_doc(doc: dict) -> None:
    query = "python"
    result: dict = {}
    n = 0
    tf = 0
    idf = 0
    for path, text in doc.items():
        if not path.endswith(".md"):
            continue

        if path not in md_chunk:
            result[path] = []
            continue

        find: list = []
        idf += 1
        data = text
        for span in md_chunk[path]:
            start, end = span
            count = data.count(query, start, end)
            if count > 0:
                find.append((query, (start, end), count))

        result[path] = find
        n = len(result)
        
    return result


def index_bm25():
    pass

def index_tf_idf():
    pass


"""
"""
dir, documents, full_path = load_docs("vllm-0.10.1")
py_chunk, md_chunk = chunks(full_path)
# print(md_chunk["vllm-0.10.1/benchmarks/README.md"])
result = search_query_in_doc(full_path)
n = len(result.items())
idf = 0
for path, value in result.items():
    if result[path] == []:
        continue
    idf += 1
    print(f"{path}: {value}\n")


print(n)
print(idf)