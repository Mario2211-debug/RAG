import os
import sys
import tqdm
import json
import math
import time
import fire
from typing import Any
from src.index import Index
from src.data_process import DataProcess
from collections import Counter, defaultdict
from src.models import CORPUS_ROOT, INDEX_PATH, MIN_IOU, K1, B
from src.utils.tokenizer import tokenizer

indexing = Any
data = DataProcess()


def bm25_scores(query: str, index: dict) -> dict[int, float]:
    """Pontua apenas os chunks que contem algum token da pergunta."""
    postings = index["postings"]
    doc_len = index["doc_len"]
    total = index["n_chunks"]
    avgdl = index["avgdl"] or 1.0
    scores: dict[int, float] = defaultdict(float)

    for token in tqdm(tokenizer(query)):
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
    documents = data.load_docs(CORPUS_ROOT)
    print(f"ficheiros lidos: {len(documents):,}")
    indexing = Index(documents)
    index = indexing.build_index()
    indexing.save_index(index)
    size = os.path.getsize(INDEX_PATH) / 1e6
    print(f"chunks .........: {index['n_chunks']:,}")
    print(f"vocabulario ....: {len(index['postings']):,}")
    print(f"avgdl ..........: {index['avgdl']:.0f} tokens")
    print(f"indice .........: {INDEX_PATH} ({size:.1f} MB)")
    print(f"tempo ..........: {time.time() - start:.1f} s  (limite 300 s)")


def cmd_search(query: str, k: int = 10) -> None:
    """Mostra o top-k para uma pergunta."""
    index = indexing.load_index()
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
    index = indexing.load_index()
    score, found, total = recall_at_k(dataset, index, k)
    print(f"recall@{k}: {score:.3f}  ({found}/{total})")


def cmd_evaluate_all() -> None:
    """Corre os datasets publicos em varios k."""
    index = indexing.load_index()
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
