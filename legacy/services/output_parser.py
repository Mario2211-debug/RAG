"""Turns the generated text into a validated :class:`FunctionCall`.

Constrained decoding already makes the text parseable, so this module is a
safety net rather than a workhorse: it re-checks the invariants the grammar is
supposed to enforce, and coerces the values to the exact Python types the
declared schema asks for.
"""

import json
from typing import Any

from pydantic import ValidationError

from ..models import FunctionCall, FunctionDefinition, ParamType


class OutputParserError(Exception):
    """Raised when the generated text is not a schema-compliant call."""


class OutputParser:
    """Validates and normalises one generated function call."""

    def parse(
        self,
        prompt: str,
        generated_text: str,
        function_definition: FunctionDefinition | None = None,
    ) -> FunctionCall:
        """Builds a :class:`FunctionCall` from generated JSON.

        Args:
            prompt: The original natural language request.
            generated_text: The JSON document produced by the decoder.
            function_definition: Definition of the function the decoder
                settled on, used to coerce argument types.

        Returns:
            The validated call, ready to be written out.

        Raises:
            OutputParserError: If the text is not valid JSON, misses a
                required key, or does not match the declared parameters.
        """
        try:
            data = json.loads(generated_text)
        except json.JSONDecodeError as exc:
            raise OutputParserError(
                f"Generated text is not valid JSON ({exc}): {generated_text!r}"
            ) from exc

        if not isinstance(data, dict):
            raise OutputParserError(
                f"Generated JSON is not an object: {generated_text!r}"
            )

        try:
            name = data["name"]
            parameters = data["parameters"]
        except KeyError as exc:
            raise OutputParserError(
                f"Generated JSON is missing the key {exc}."
            ) from exc

        if not isinstance(parameters, dict):
            raise OutputParserError("Generated 'parameters' is not an object.")

        if function_definition is not None:
            self._check_parameter_names(parameters, function_definition)
            parameters = self._coerce(parameters, function_definition)

        try:
            return FunctionCall(prompt=prompt, name=name,
                                parameters=parameters)
        except ValidationError as exc:
            raise OutputParserError(
                f"Generated call does not match the output schema: {exc}"
            ) from exc

    @staticmethod
    def _check_parameter_names(
        parameters: dict[str, Any], definition: FunctionDefinition
    ) -> None:
        """Ensures the produced arguments are exactly the declared ones.

        Raises:
            OutputParserError: If an argument is missing or unexpected.
        """
        produced = set(parameters)
        declared = set(definition.parameters)

        if produced != declared:
            missing = sorted(declared - produced)
            unexpected = sorted(produced - declared)
            raise OutputParserError(
                f"Arguments for {definition.name} do not match the schema "
                f"(missing={missing}, unexpected={unexpected})."
            )

    @staticmethod
    def _coerce(
        parameters: dict[str, Any], definition: FunctionDefinition
    ) -> dict[str, Any]:
        """Coerces each argument to the Python type its schema declares."""
        coerced: dict[str, Any] = {}

        for name, value in parameters.items():
            param_type = definition.parameters[name].type
            numeric = (isinstance(value, (int, float))
                       and not isinstance(value, bool))

            if param_type == ParamType.NUMBER and numeric:
                coerced[name] = float(value)
            elif param_type == ParamType.INTEGER and numeric:
                coerced[name] = int(value)
            elif param_type == ParamType.STRING and isinstance(value, str):
                coerced[name] = value.strip()
            else:
                coerced[name] = value

        return coerced
