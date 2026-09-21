from collections import Counter, defaultdict
from utils.tokenizer import tokenizer
from models import OVERLAP, MAX_CHUNK_SIZE, PATH_PREFIX, HEADING_RE


class Indexing():
    def __init__(self) -> None:
        pass

    def chunk_python_file(self,
                          text: str,
                          index: int,
                          max_chunk_size: int = MAX_CHUNK_SIZE,
                          overlap: int = OVERLAP
                          ) -> tuple[int, list[tuple[int, int, int]]]:
        """Corta codigo em spans (id, inicio, fim),
        preferindo linhas em branco."""
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

    def chunk_markdown_file(self,
                            text: str,
                            index: int,
                            max_chunk_size: int = MAX_CHUNK_SIZE,
                            overlap: int = OVERLAP
                            ) -> tuple[int, list[tuple[int, int, int]]]:
        """Corta texto em spans, priorizando fronteiras de seccao (##)."""
        boundaries = [m.start() for m in HEADING_RE.finditer(text)]
        boundaries.append(len(text))
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

    def build_index(self, documents: dict[str, str]) -> dict:
        """Constroi o indice invertido numa unica passagem pelo corpus.

        Nao procura nada: percorre cada chunk uma vez e vai acrescentando.
        """
        postings: dict[str, list] = defaultdict(list)
        chunks: dict[int, list] = {}
        doc_len: dict[int, int] = {}
        counter = 0

        for path, text in documents.items():
            if path.endswith(".py"):
                counter, spans = self.chunk_python_file(text, counter)
            else:
                counter, spans = self.chunk_markdown_file(text, counter)
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
