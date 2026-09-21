import os
import json
from src.chunking import chunk_document
from src.utils.tokenizer import tokenizer
from collections import Counter, defaultdict
from src.models import PATH_PREFIX, INDEX_PATH


class Index():
    def __init__(self,  documents: dict[str, str]) -> None:
        self.documents = documents
        pass

    def build_index(self) -> dict:
        """Constroi o indice invertido numa unica passagem pelo corpus.

        Nao procura nada: percorre cada chunk uma vez e vai acrescentando.
        """
        postings: dict[str, list] = defaultdict(list)
        chunks: dict[int, list] = {}
        doc_len: dict[int, int] = {}
        counter = 0

        for path, text in self.documents.items():
            counter, spans = chunk_document(path, text, counter)
            for cid, start, end in spans:
                tokens = tokenizer(text[start:end])
                chunks[cid] = [PATH_PREFIX + path, start, end]
                doc_len[cid] = len(tokens)
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

    def save_index(self, index: dict, path: str = INDEX_PATH) -> None:
        """Persiste o indice em JSON."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(index, handle)

    def load_index(self, path: str = INDEX_PATH) -> dict:
        """Recarrega o indice; o JSON devolve as chaves como strings."""

        index: dict = {}
        with open(path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        index["chunks"] = {int(k): v for k, v in index["chunks"].items()}
        index["doc_len"] = {int(k): v for k, v in index["doc_len"].items()}
        return index
