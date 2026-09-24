"""Build, persist, and load the inverted index."""

import json
import os
from collections import Counter, defaultdict
from typing import Any

from tqdm import tqdm

from src.chunking import chunk_document
from src.config import (HEADING_CONTEXT, HEADING_TEXT_RE, INDEX_PATH,
                        MAX_CHUNK_SIZE, PATH_BOOST)
from src.utils.tokenizer import tokenizer


REQUIRED_KEYS = ("postings", "chunks", "doc_len", "avgdl", "n_chunks")


class Index():
    """Lexical inverted index over the corpus chunks."""

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents

    @staticmethod
    def headings(path: str, text: str) -> list[tuple[int, str]]:
        """Position and text of each heading in a text file."""
        # print(f"Headings:\nPath {path}\nText: {text}")
        if path.endswith(".py"):
            return []
        return [(m.start(), m.group(1))
                for m in HEADING_TEXT_RE.finditer(text)]

    @staticmethod
    def heading_context(headings: list[tuple[int, str]], start: int) -> str:
        """The most recent headings that were open before the chunk.

        A chunk in the middle of a section does not contain the section title,
        but that title often carries the words that match the question.
        """
        above = [title for position, title in headings if position <= start]
        return " ".join(above[-HEADING_CONTEXT:])

    def build_index(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ) -> dict[str, Any]:
        """Build the inverted index in a single pass over the corpus.

        It does not search: it walks each chunk once and accumulates postings.
        """
        postings: dict[str, list[list[int]]] = defaultdict(list)
        chunks: dict[int, list[Any]] = {}
        doc_len: dict[int, int] = {}
        counter = 0

        for path, text in tqdm(self.documents.items(), desc="indexing",
                               unit="file"):
            headings = self.headings(path, text)
            counter, spans = chunk_document(path, text, counter,
                                            max_chunk_size)
            for cid, start, end in spans:
                # the file name is also evidence: a question about "lora"
                # must be able to find docs/features/lora.md
                context = self.heading_context(headings, start)
                # print(f"{headings}: {start}")
                # print(f"CONTEXT: \n{context}")
                tokens = (tokenizer(text[start:end])
                          + tokenizer(path) * PATH_BOOST
                          + tokenizer(context))
                chunks[cid] = [path, start, end]
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
            "max_chunk_size": max_chunk_size,
        }

    def save_index(self, index: dict[str, Any],
                   path: str = INDEX_PATH) -> None:
        """Persist the index to JSON."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(index, handle)

    @staticmethod
    def load_index(path: str = INDEX_PATH) -> dict[str, Any]:
        """Reload the index; JSON returns string keys."""
        with open(path, "r", encoding="utf-8") as handle:
            index: dict[str, Any] = json.load(handle)
        missing = [key for key in REQUIRED_KEYS if key not in index]
        if missing:
            raise KeyError(f"missing keys: {', '.join(missing)}")
        index["chunks"] = {int(k): v for k, v in index["chunks"].items()}
        index["doc_len"] = {int(k): v for k, v in index["doc_len"].items()}
        return index
