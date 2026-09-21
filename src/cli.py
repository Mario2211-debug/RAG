import os
import json
from typing import Any
from src.retieval import Retrieval
from pydantic import ValidationError
from models import (Augment, Evaluate, Indexing, Generate,
                    RagDataset, StudentSearchResults, AnsweredQuestion,
                    MinimalAnswer, MinimalSearchResults, MinimalSource,
                    StudentSearchResultsAndAnswer, UnansweredQuestion)


class CliError(Exception):
    pass


def as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CliError(f"--{name} must be an integer, get {value!r}")
    try:
        return int(value)
    except ValueError:
        raise CliError(f"--{name} must be an integer, get {value!r}")


def read_json(path: str) -> Any:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise CliError(f"file not found: {path}")
    except json.JSONDecodeError as err:
        raise CliError(f"Mal formed json in {path}: {err}")
    except OSError as err:
        raise CliError(f"cannot read file in {path}: {err}")


def write_json(payload: Any, save_directory: str, filename: str) -> str:
    try:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload.model_dump_json(ident=2))
    except OSError as err:
        raise CliError(f"cannot write file in {save_directory}: {err}")
    return path


def load_dataset(path: str) -> RagDataset:
    try:
        return RagDataset.model_validate(read_json(path))
    except ValidationError as err:
        raise CliError(f"{path} is not a valid rag dataset: {err}")


def load_search_result(path: Any) -> StudentSearchResults:
    try:
        return StudentSearchResults.model_validate(read_json(path))
    except ValidationError as err:
        raise CliError(f"{path} is not a valid StudentSearchResults: {err}")


def load_retrivier(index_path: str) -> Retrieval:
    try:
        return Retrieval(Indexing(index_path))
    except FileNotFoundError:
        raise CliError(f"index not found: {index_path}\n"
                       "build it first: uv run python -m src index")
    except (json.JSONDecodeError, KeyError) as err:
        raise CliError(f"corrupt index in {index_path}: {err}\n"
                       "rebuild it: uv run python -m src index")
