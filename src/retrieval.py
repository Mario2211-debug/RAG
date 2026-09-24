"""BM25 search over the inverted index."""

import math
from collections import defaultdict
from typing import Any

from src.config import B, FILE_PRIOR, K1
from src.models import MinimalSource
from src.utils.tokenizer import tokenizer


class Retriever():
    """Score and rank chunks for a question."""

    def __init__(self, index: dict[str, Any]) -> None:
        self.index = index

    def bm25_scores(self, query: str) -> dict[int, float]:
        """Score only chunks that contain at least one query token."""
        postings = self.index["postings"]
        doc_len = self.index["doc_len"]
        total = self.index["n_chunks"]
        avgdl = self.index["avgdl"] or 1.0
        scores: dict[int, float] = defaultdict(float)

        for token in set(tokenizer(query)):
            entries = postings.get(token)
            if not entries:
                continue
            # a token present in almost every chunk contributes almost nothing;
            # the IDF is what removes its weight
            df = len(entries)
            idf = math.log((total - df + 0.5) / (df + 0.5) + 1.0)
            for cid, freq in entries:
                norm = 1.0 - B + B * doc_len[cid] / avgdl
                scores[cid] += idf * freq * (K1 + 1.0) / (freq + K1 * norm)
        return scores

    def apply_file_prior(self, scores: dict[int, float]) -> dict[int, float]:
        """Boost chunks from files that perform well as a group.

        A documentation question usually concerns a theme, and that theme lives
        in an entire file: if several chunks from the same file score well, the
        answer is likely there.
        """
        if not FILE_PRIOR or not scores:
            return scores
        per_file: dict[str, float] = defaultdict(float)
        for cid, score in scores.items():
            per_file[self.index["chunks"][cid][0]] += score
        best = max(per_file.values())
        if best <= 0:
            return scores
        for cid in scores:
            share = per_file[self.index["chunks"][cid][0]] / best
            scores[cid] += FILE_PRIOR * scores[cid] * share
        return scores

    def search(self, query: str,
               k: int) -> list[tuple[MinimalSource, float]]:
        """Return the k best chunks, each with its score."""
        if k <= 0 or not query.strip():
            return []
        scores = self.apply_file_prior(self.bm25_scores(query))
        # break ties by chunk id so the ordering is reproducible
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        results: list[tuple[MinimalSource, float]] = []
        for cid, score in ranked[:k]:
            path, start, end = self.index["chunks"][cid]
            source = MinimalSource(file_path=path,
                                   first_character_index=start,
                                   last_character_index=end)
            results.append((source, score))
        return results

    def sources(self, query: str, k: int) -> list[MinimalSource]:
        """Only the sources, without scores: what goes into the JSON output."""
        return [source for source, _score in self.search(query, k)]
