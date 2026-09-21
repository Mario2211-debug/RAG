import json
from pathlib import Path
from ..models import FunctionCall


class ResultWriterError(Exception):
    pass


class ResultWriter:
    def write(self, path: str | Path, results: list[FunctionCall]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.model_dump() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
