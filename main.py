import os
import re


HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z]+")


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


def chunk_python_file(text: str, index: int,
                      max_chunk_size: int = 2000,
                      overlap: int = 200) -> tuple[int, list[tuple[int, int, int]]]:
    spans: list[tuple[int, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chunk_size, n)
        if end < n:
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


def chunk_markdown_file(text: str, index: int,
                        max_chunk_size: int = 2000,
                        overlap: int = 200) -> tuple[int, list[tuple[int, int, int]]]:
    boundaries = [m.start() for m in HEADING_RE.finditer(text)] + [len(text)]
    spans: list[tuple[int, int, int]] = []
    start = 0
    for boundary in boundaries:
        while boundary - start > max_chunk_size:
            cut = start + max_chunk_size
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


def chunks(full_path: dict) -> tuple[dict, dict]:
    py_chunk: dict = {}
    md_chunk: dict = {}
    index: int = 0

    for path, full in full_path.items():
        if path.endswith(".py"):
            index, py_chunk[path] = chunk_python_file(full, index)
        elif path.endswith(".md"):
            index, md_chunk[path] = chunk_markdown_file(full, index)

    return py_chunk, md_chunk


def tokenizer(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]

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
            # result[path] = []
            continue

        find: list = []
        idf += 1
        data = text
        for span in md_chunk[path]:
            index, start, end = span
            count = data.count(query, start, end)
            if count > 0:
                find.append((query, (start, end), count))
        result[path] = find
        n = len(result)
        
    return result


def build_index(full_path: dict, py_chunk: dict, md_chunk: dict) -> dict:
    index: dict = {}

    for path, spans in {**py_chunk, **md_chunk}.items():
        text = full_path[path]
        for cid, start, end in spans:
            snippet = text[start:end]
            index[cid] = {
                "path": path,
                "start": start,
                "end": end,
                "tokens": tokenizer(snippet),
            }
    return index


def search(query: str, index: dict, top_k: int = 10) -> list[tuple[int, float]]:
    q_tokens = tokenizer(query)
    scores: dict[int, float] = {cid: 0.0 for cid in index}
    for cid, entry in index.items():
        for t in q_tokens:
            scores[cid] += entry["tokens"].count(t)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [kv for kv in ranked[:top_k] if kv[1] > 0]
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
    # print(f"{path}: {value}\n")
# print("Number of files: ", n)
# print("Valid files(idf): ", idf)


text = "Everything starts with the index." \
"# Read the files you judge useful from the vLLM repository shipped in the attachments, split each one into chunks, and persist an index that retrieval can query in milliseconds." \
"Indexing the whole corpus must take at most 5minutes."

boundaries = [m.start() for m in HEADING_RE.finditer(text)] + [len(text)]
# print(boundaries)
print(md_chunk)