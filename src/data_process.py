import os
import re

TEXT_EXT = (".py", ".md", ".txt")
SKIP_DIRS = {"__pycache__", "node_modules", ".git"}
_TOKEN_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z]+")


class DataProcess():

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    def tokenizer(self, text: str) -> list[str]:
        """Corta texto em tokens minusculos.
        A mesma funcao corre sobre os chunks (na indexacao) e sobre a
        pergunta (na pesquisa): os dois lados tem de falar o mesmo dialeto.
        """
        return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]

    def load_docs(self, folder: str) -> dict[str, str]:
        """Le recursivamente os ficheiros de texto do corpus.
        Devolve {path_relativo: conteudo}.
        Salta diretorios de cache mas nao ficheiros
        comecados por "_": os __init__.py do vLLM sao codigo
        real e aparecem como sources de referencia."""
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
