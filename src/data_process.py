"""Read the corpus from disk."""

import os

from src.config import SKIP_DIRS, TEXT_EXT


class DataProcess():
    """Walk a file tree and return the text of each file."""

    def __init__(self) -> None:
        self.docs: dict[str, str] = {}

    def load_docs(self, folder: str) -> dict[str, str]:
        """Recursively read the text files from the corpus.

        Returns {relative_path: content}. The path is stored exactly as it
        appears on disk (data/raw/vllm-0.10.1/...); the grader compares these
        paths verbatim.

        Skip cache directories but do not skip files starting with "_": the
        vLLM __init__.py files are real code and appear as reference sources.
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
