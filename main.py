"""Indexacao lexical (BM25) e pesquisa sobre o corpus vLLM.

Uso:
    python3 main.py index
    python3 main.py search "how do I load a LoRA adapter?" [k]
    python3 main.py evaluate <dataset.json> [k]
    python3 main.py evaluate-all
"""

import os
import re
import sys
import tqdm
import json
import math
import time
import fire
from src.data_process import DataProcess
from src.config import CORPUS_ROOT
from collections import Counter, defaultdict

PATH_PREFIX = "data/raw/"
INDEX_PATH = "data/processed/index.json"

CORPUS_ROOT = "vllm-0.10.1"

MAX_CHUNK_SIZE = 2000
OVERLAP = 200
K1 = 1.5
B = 0.75
MIN_IOU = 0.05

TEXT_EXT = (".py", ".md", ".txt")
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}

HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z]+")


def tokenizer(text: str) -> list[str]:
    """Corta texto em tokens minusculos.

    A mesma funcao corre sobre os chunks (na indexacao) e sobre a
    pergunta (na pesquisa): os dois lados tem de falar o mesmo dialeto.
    """
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


data_processment = DataProcess()


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


def build_index(documents: dict[str, str]) -> dict:
    """Constroi o indice invertido numa unica passagem pelo corpus.

    Nao procura nada: percorre cada chunk uma vez e vai acrescentando.
    """
    postings: dict[str, list] = defaultdict(list)
    chunks: dict[int, list] = {}
    doc_len: dict[int, int] = {}
    counter = 0

    for path, text in documents.items():
        if path.endswith(".py"):
            counter, spans = chunk_python_file(text, counter)
        elif path.endswith(".md"):
            counter, spans = chunk_markdown_file(text, counter)
        else:
            continue
        for cid, start, end in spans:
            tokens = tokenizer(text[start:end])
            chunks[cid] = [PATH_PREFIX + path, start, end]
            doc_len[cid] = len(tokens)
            # o Counter junta as repeticoes dentro do chunk, por isso
            # cada chunk entra uma unica vez na lista de cada palavra
            for token, freq in Counter(tokens).items():
                postings[token].append([cid, freq])

    total = len(doc_len)
    avgdl = sum(doc_len.values()) / total if total else 0.0
    return {
        "postings": dict(postings),
        "chunks": chunks,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "n_chunks": total,
    }


def save_index(index: dict, path: str = INDEX_PATH) -> None:
    """Persiste o indice em JSON."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(index, handle)


def load_index(path: str = INDEX_PATH) -> dict:
    """Recarrega o indice; o JSON devolve as chaves como strings."""

    index: dict = {}
    with open(path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    index["chunks"] = {int(k): v for k, v in index["chunks"].items()}
    index["doc_len"] = {int(k): v for k, v in index["doc_len"].items()}
    return index


def bm25_scores(query: str, index: dict) -> dict[int, float]:
    """Pontua apenas os chunks que contem algum token da pergunta."""
    postings = index["postings"]
    doc_len = index["doc_len"]
    total = index["n_chunks"]
    avgdl = index["avgdl"] or 1.0
    scores: dict[int, float] = defaultdict(float)

    for token in tokenizer(query):
        entries = postings.get(token)
        if not entries:
            continue
        df = len(entries)
        idf = math.log((total - df + 0.5) / (df + 0.5) + 1.0)
        for cid, freq in entries:
            norm = 1.0 - B + B * doc_len[cid] / avgdl
            scores[cid] += idf * freq * (K1 + 1.0) / (freq + K1 * norm)
    return scores


def search(query: str, index: dict,
           k: int = 10) -> list[tuple[str, int, int, float]]:
    """Devolve os k melhores chunks como (file_path, inicio, fim, score)."""
    if k <= 0 or not query.strip():
        return []
    scores = bm25_scores(query, index)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    results = []
    for cid, score in ranked[:k]:
        path, start, end = index["chunks"][cid]
        results.append((path, start, end, score))
    return results


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection over union de dois intervalos de caracteres."""
    inter = min(a_end, b_end) - max(a_start, b_start)
    if inter <= 0:
        return 0.0
    return inter / (max(a_end, b_end) - min(a_start, b_start))


def recall_at_k(dataset_path: str, index: dict, k: int = 5) -> tuple:
    """Recall@k: fatia das sources corretas que aparecem no top-k."""
    with open(dataset_path, "r", encoding="utf-8") as handle:
        questions = json.load(handle)["rag_questions"]

    found = 0
    total = 0
    for question in questions:
        sources = question.get("sources") or []
        if not sources:
            continue
        results = search(question["question"], index, k)
        for source in sources:
            total += 1
            for path, start, end, _score in results:
                if path != source["file_path"]:
                    continue
                if iou(start, end,
                       source["first_character_index"],
                       source["last_character_index"]) >= MIN_IOU:
                    found += 1
                    break
    return (found / total if total else 0.0), found, total


def cmd_index() -> None:
    """Le o corpus, corta, tokeniza, indexa e grava."""
    start = time.time()
    documents = data_processment.load_docs(CORPUS_ROOT)
    print(f"ficheiros lidos: {len(documents):,}")
    index = build_index(documents)
    save_index(index)
    size = os.path.getsize(INDEX_PATH) / 1e6
    print(f"chunks .........: {index['n_chunks']:,}")
    print(f"vocabulario ....: {len(index['postings']):,}")
    print(f"avgdl ..........: {index['avgdl']:.0f} tokens")
    print(f"indice .........: {INDEX_PATH} ({size:.1f} MB)")
    print(f"tempo ..........: {time.time() - start:.1f} s  (limite 300 s)")


def cmd_search(query: str, k: int = 10) -> None:
    """Mostra o top-k para uma pergunta."""
    index = load_index()
    start = time.time()
    results = search(query, index, k)
    elapsed = time.time() - start
    if not results:
        print("sem resultados")
        return
    for rank, (path, ini, fim, score) in enumerate(results, 1):
        print(f"{rank:>2}. {score:7.2f}  {path}  [{ini}:{fim}]")
    print(f"\n({elapsed * 1000:.0f} ms)")


def cmd_evaluate(dataset: str, k: int = 5) -> None:
    """Recall@k de um dataset contra o ground truth."""
    index = load_index()
    score, found, total = recall_at_k(dataset, index, k)
    print(f"recall@{k}: {score:.3f}  ({found}/{total})")


def cmd_evaluate_all() -> None:
    """Corre os datasets publicos em varios k."""
    index = load_index()
    base = "datasets_public/public/AnsweredQuestions"
    datasets = [("docs", f"{base}/dataset_docs_public.json", 0.80),
                ("code", f"{base}/dataset_code_public.json", 0.50)]
    for label, path, target in datasets:
        if not os.path.exists(path):
            print(f"{label}: dataset nao encontrado ({path})")
            continue
        start = time.time()
        line = []
        for k in (1, 3, 5, 10):
            score, _found, _total = recall_at_k(path, index, k)
            mark = ""
            if k == 5:
                mark = " OK" if score >= target else " FALHA"
            line.append(f"recall@{k}: {score:.3f}{mark}")
        elapsed = time.time() - start
        print(f"{label:<5} " + "  ".join(line))
        print(f"      ({elapsed / 4:.1f} s por passagem, limite 90 s)")


def main(argv: list[str]) -> int:
    """Despacha os subcomandos."""
    if len(argv) < 2:
        print(__doc__)
        return 1
    command = argv[1]
    try:
        if command == "index":
            cmd_index()
        elif command == "search":
            if len(argv) < 3:
                print("uso: main.py search <query> [k]")
                return 1
            k = int(argv[3]) if len(argv) > 3 else 10
            cmd_search(argv[2], k)
        elif command == "evaluate":
            if len(argv) < 3:
                print("uso: main.py evaluate <dataset.json> [k]")
                return 1
            k = int(argv[3]) if len(argv) > 3 else 5
            cmd_evaluate(argv[2], k)
        elif command == "evaluate-all":
            cmd_evaluate_all()
        else:
            print(__doc__)
            return 1
    except FileNotFoundError as err:
        print(f"ficheiro nao encontrado: {err.filename}")
        print("corre primeiro: python3 main.py index")
        return 1
    except (ValueError, json.JSONDecodeError) as err:
        print(f"entrada invalida: {err}")
        return 1
    return 0


if __name__ == "__main__":
    # fire.Fire("Hello")
    sys.exit(main(sys.argv))
