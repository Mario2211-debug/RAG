"""Leitura do corpus a partir do disco."""

import os

from src.config import SKIP_DIRS, TEXT_EXT


class DataProcess():
    """Percorre uma arvore de ficheiros e devolve o texto de cada um."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    def load_docs(self, folder: str) -> dict[str, str]:
        """Le recursivamente os ficheiros de texto do corpus.

        Devolve {path_relativo: conteudo}. O path e guardado tal como
        aparece no disco (data/raw/vllm-0.10.1/...), porque o grader
        compara os paths verbatim.

        Salta diretorios de cache mas nao ficheiros comecados por "_":
        os __init__.py do vLLM sao codigo real e aparecem como sources
        de referencia.
        """
        try:
            items = sorted(os.listdir(folder))
        except OSError:
            return self.docs

        for item in items:
            if item.startswith("."):
                continue
            path = os.path.join(folder, item)
            if os.path.isdir(path):
                if item not in SKIP_DIRS:
                    self.docs.update(self.load_docs(path))
            elif item.endswith(TEXT_EXT):
                try:
                    with open(path, "r", encoding="utf-8",
                              errors="ignore") as handle:
                        self.docs[path] = handle.read()
                except OSError:
                    continue
        return self.docs
