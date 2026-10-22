"""Mechanic C — schema-based structured output.

The v2.0 pipeline never does regex semantic extraction from raw traces. Every
structured phase is instructed to emit a JSON object wrapped in explicit
markers, which we then extract, parse, and validate against a Pydantic schema.

    FINAL_JSON_START
    { ... }
    FINAL_JSON_END
"""
from __future__ import annotations

import json
import typing
from enum import Enum
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

FINAL_JSON_START = "FINAL_JSON_START"
FINAL_JSON_END = "FINAL_JSON_END"

T = TypeVar("T", bound=BaseModel)


def _balanced_object(text: str) -> Optional[str]:
    """Return the first balanced top-level ``{...}`` object in ``text``."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_final_json_block(text: str) -> Optional[str]:
    """Extract the raw JSON text between the FINAL_JSON markers.

    Falls back to a balanced scan of the whole response when markers are
    absent, so a model that drops a marker still degrades gracefully.
    """
    if not text:
        return None
    s = text.find(FINAL_JSON_START)
    if s != -1:
        s += len(FINAL_JSON_START)
        e = text.find(FINAL_JSON_END, s)
        region = text[s:e] if e != -1 else text[s:]
        block = _balanced_object(region)
        if block is not None:
            return block
    return _balanced_object(text)


def _format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors()[:10]:
        loc = ".".join(str(x) for x in e.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {e.get('msg', 'invalid')}")
    return "schema validation failed — " + "; ".join(parts)


def parse_structured(
    text: str, schema: Type[T], default: Optional[T] = None
) -> tuple[Optional[T], Optional[str]]:
    """Extract, parse and validate a structured response.

    Returns ``(instance, None)`` on success or ``(default, error_message)`` on
    failure. The error message is safe to feed back to the model for a retry.
    """
    block = extract_final_json_block(text)
    if block is None:
        return (default, "no JSON object found in the response")
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        return (default, f"invalid JSON: {exc}")
    if not isinstance(data, dict):
        return (default, "JSON is not an object")
    try:
        return (schema.model_validate(data), None)
    except ValidationError as exc:
        return (default, _format_validation_error(exc))


def _clean_forward_arg(arg: str) -> str:
    """Clean an unresolved forward-ref string into a simple type name.

    Handles the common Pydantic case where an annotation is stored as a
    forward-ref string like ``"Optional['AnswerContract']"`` or ``"list[str]"``.
    """
    s = (arg or "").strip()
    for prefix, suffix, repl in (
        ("Optional[", "]", lambda inner: inner),
        ("list[", "]", lambda inner: f"array of {inner}"),
        ("List[", "]", lambda inner: f"array of {inner}"),
    ):
        if s.startswith(prefix) and s.endswith(suffix):
            s = repl(s[len(prefix) : -1].strip())
            break
    return s.strip("'\"") or "any"


def _simple_type(ann: Any) -> str:
    """Convert a Python type annotation into a simple, human-readable type name
    for a prompt.

    v2.6 guardrail: prompts must NOT leak raw Python type dumps
    (``typing.Optional[ForwardRef(...)]``, ``<enum 'Route'>``, ...). This maps
    annotations to plain JSON-style names (string / integer / array / object)
    and spells out enum members and nested-object fields.
    """
    if ann is None or ann is Any:
        return "any"
    if isinstance(ann, str):
        return _clean_forward_arg(ann)
    if isinstance(ann, typing.ForwardRef):
        return _clean_forward_arg(ann.__forward_arg__)
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _simple_type(non_none[0])
        return " | ".join(_simple_type(a) for a in non_none) or "any"
    if origin in (list, typing.List, set, typing.Set):
        inner = _simple_type(args[0]) if args else "any"
        return f"array of {inner}"
    if origin in (dict, typing.Dict):
        return "object"
    if origin in (tuple, typing.Tuple):
        return "array"
    if isinstance(ann, type):
        if ann is str:
            return "string"
        if ann is int:
            return "integer"
        if ann is float:
            return "number"
        if ann is bool:
            return "boolean"
        if issubclass(ann, Enum):
            values = ", ".join(f'"{m.value}"' for m in ann)
            return f"one of {values}"
        if issubclass(ann, BaseModel):
            # Spell out the nested object's fields so the model knows the shape.
            inner = _nested_fields(ann)
            return f"object {inner}" if inner else "object"
        return ann.__name__
    return str(ann).replace("typing.", "")


def _nested_fields(model: Type[BaseModel], depth: int = 0) -> str:
    """A compact ``{field: type, ...}`` description of a nested model (one level)."""
    if depth > 1:
        return ""
    parts = []
    for name, f in model.model_fields.items():
        parts.append(f"{name}: {_simple_type(f.annotation)}")
    if not parts:
        return ""
    return "{" + ", ".join(parts) + "}"


def describe_schema(schema: Type[BaseModel]) -> str:
    """Compact, human-readable field list for a prompt (no Python type dumps)."""
    lines = []
    for name, field in schema.model_fields.items():
        simple = _simple_type(field.annotation)
        if field.is_required():
            lines.append(f"- {name}: {simple} (required)")
        else:
            lines.append(f"- {name}: {simple} (optional)")
    return "\n".join(lines) if lines else "(no fields)"


def build_schema_instruction(
    schema: Type[BaseModel],
    purpose: str = "",
    exemplar: Optional[dict[str, Any]] = None,
) -> str:
    """Prompt block that forces marker-wrapped, schema-valid JSON."""
    example = (
        "\nOne example of the exact shape (values are illustrative only):\n"
        f"{json.dumps(exemplar, ensure_ascii=False)}\n"
        if exemplar
        else ""
    )
    return (
        f"{purpose}\n"
        "Respond with ONLY one JSON object. Wrap it exactly between these two "
        "marker lines and put nothing after the closing marker:\n"
        f"{FINAL_JSON_START}\n"
        "{ ... }\n"
        f"{FINAL_JSON_END}\n"
        f"Required/optional fields:\n{describe_schema(schema)}\n"
        f"{example}"
        "Honesty rules:\n"
        "- If you are unsure about a value, lower the relevant confidence and "
        "say so; never invent facts.\n"
        "- If a field does not apply, use an empty value, not a guess.\n"
    )
