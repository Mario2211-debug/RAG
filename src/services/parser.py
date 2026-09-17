"""Reading and validating the two JSON input files.

Both files come from outside the program and may be missing, truncated or
simply wrong, so every failure here is turned into a :class:`ParserError`
carrying a message meant for the user rather than a traceback.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..models import FunctionDefinition

_FUNCTIONS_ADAPTER = TypeAdapter(list[FunctionDefinition])


class ParserError(Exception):
    """Raised when an input file is missing, malformed or off-schema."""


class Parser:
    """Loads function definitions and prompts from disk."""

    def load_functions(self, path: str | Path) -> list[FunctionDefinition]:
        """Loads ``functions_definition.json``.

        Args:
            path: Path to the function definition file.

        Returns:
            The declared functions, in file order.

        Raises:
            ParserError: If the file is unreadable, is not valid JSON, does
                not hold a list, is empty, declares duplicate names or uses
                an unsupported parameter type.
        """
        data = self._load_json(path)

        if not isinstance(data, list):
            raise ParserError(
                f"{path}: expected a JSON array of function definitions, "
                f"got {type(data).__name__}."
            )

        try:
            functions = _FUNCTIONS_ADAPTER.validate_python(data)
        except ValidationError as exc:
            raise ParserError(
                f"{path}: invalid function definition:\n{exc}"
            ) from exc

        if not functions:
            raise ParserError(f"{path}: no functions defined.")

        names = [function.name for function in functions]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ParserError(
                f"{path}: duplicate function names: {sorted(duplicates)}."
            )

        return functions

    def load_prompts(self, path: str | Path) -> list[str]:
        """Loads ``function_calling_tests.json``.

        Args:
            path: Path to the prompts file.

        Returns:
            The prompt strings, in file order.

        Raises:
            ParserError: If the file is unreadable, is not valid JSON, does
                not hold a list, or holds entries without a string ``prompt``.
        """
        data = self._load_json(path)

        if not isinstance(data, list):
            raise ParserError(
                f"{path}: expected a JSON array of prompts, "
                f"got {type(data).__name__}."
            )

        prompts: list[str] = []
        for position, entry in enumerate(data):
            prompts.append(self._extract_prompt(path, position, entry))
        return prompts

    @staticmethod
    def _extract_prompt(path: str | Path, position: int, entry: Any) -> str:
        """Pulls the prompt text out of one entry of the input file."""
        if isinstance(entry, str):
            return entry
        if not isinstance(entry, dict):
            raise ParserError(
                f"{path}: entry {position} is a {type(entry).__name__}, "
                "expected an object with a 'prompt' key."
            )
        prompt = entry.get("prompt")
        if not isinstance(prompt, str):
            raise ParserError(
                f"{path}: entry {position} has no string 'prompt' key."
            )
        return prompt

    @staticmethod
    def _load_json(path: str | Path) -> Any:
        """Reads and decodes a JSON file, reporting failures as ParserError."""
        try:
            with open(path, "r", encoding="utf-8") as input_file:
                return json.load(input_file)
        except FileNotFoundError as exc:
            raise ParserError(f"Input file not found: {path}") from exc
        except IsADirectoryError as exc:
            raise ParserError(f"Input path is a directory: {path}") from exc
        except PermissionError as exc:
            raise ParserError(f"Permission denied reading {path}") from exc
        except UnicodeDecodeError as exc:
            raise ParserError(
                f"{path}: file is not valid UTF-8: {exc}"
            ) from exc
        except OSError as exc:
            raise ParserError(f"Could not read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ParserError(f"{path}: invalid JSON: {exc}") from exc
