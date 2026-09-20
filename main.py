import os
import re
from typing import DefaultDict, Counter

PATH_PREFIX = "data/raw/"
HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z]+")
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}
TEXT_EXT = (".py", ".md", ".txt")

def tokenizer(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]

def print_output(full_path: dict) -> None:
    for key, value in full_path.items():
        print(f"\033[93mCamiho\033[0m: \033[95m{key}\033[0m")
        for path, data in value.items():
            print(f"\033[93mficheiro\033[0m: "
                  f"\033[42m{path}\033[0m\n\n", value[path])
    pass


def load_docs(folder: str) -> dict[str, str]:

    documents: dict[str, str] = {}

    try:
        items = os.listdir(folder)
    except OSError:
        return documents

    for item in items:
        if item.startswith(".") and item in SKIP_DIRS:
            continue
        path = os.path.join(folder, item)
        if os.path.isdir(path):
            if item not in SKIP_DIRS:
                documents.update(load_docs(path))
        elif item.endswith(TEXT_EXT):
            try:
                with open(path, "r", encoding="utf-8",
                        errors="ignore") as file:
                    data = file.read()
                    documents[path] = data
            except OSError:
                continue
    return documents


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


def build_index(full_path: dict) -> tuple[dict, dict]:
    postings: dict[str, list] = DefaultDict(list)
    chunks: dict[int, list] = {}
    doc_len: dict[int, int] = {}
    index = 0

    for path, full in full_path.items():
        if path.endswith(".py"):
            index, span = chunk_python_file(full, index)
        elif path.endswith(".md"):
            index, span = chunk_markdown_file(full, index)
        for cid, start, end in span:
            tokens = tokenizer(full[start:end])
            chunks[cid] = [PATH_PREFIX + path, start, end]
            doc_len[cid] = len(tokens)
            for token, freq in Counter(tokens).items():
                postings[token].append((cid, freq))
    total = len(doc_len)
    avgdl = sum(doc_len.values()) / total if total else 0.0
    print(Counter(tokens).items())
    return {
        "postings": dict(postings),
        "chunks": chunks,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "n_chunks": total
    }



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


documents = load_docs("vllm-0.10.1")
cks = build_index(documents)


text = "Everything starts with the index." \
"# Read the files you judge useful from the vLLM repository shipped in the attachments, split each one into chunks, and persist an index that retrieval can query in milliseconds." \
"Indexing the whole corpus must take at most 5minutes."

docs = load_docs("vllm-0.10.1")
print(len(docs))
print("vllm-0.10.1/vllm/utils/__init__.py" in docs)
print("vllm-0.10.1/CMakeLists.txt" in docs)
for k, v in cks.items():
    data = cks["postings"]
    print(data)