"""Robust JSON extraction from LLM output.

LLMs frequently wrap JSON in code fences, add prose before/after, or emit
slightly malformed JSON. These helpers are deliberately lenient.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _balanced_scan(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced open_ch..close_ch block, respecting strings."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _loads_lenient(raw: str) -> Any:
    raw = _strip_code_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Repair common issues: trailing commas, single quotes.
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse JSON from: {raw[:200]}")


def extract_json(text: str, default: Any = None) -> Any:
    """Extract the first JSON object from text. Returns `default` on failure."""
    if text is None:
        return default
    candidate = _balanced_scan(text, "{", "}")
    if candidate is None:
        return default
    try:
        return _loads_lenient(candidate)
    except ValueError:
        return default


def extract_list(text: str, default: Any = None) -> Any:
    """Extract the first JSON array from text. Returns `default` on failure."""
    if text is None:
        return default
    candidate = _balanced_scan(text, "[", "]")
    if candidate is None:
        return default
    try:
        return _loads_lenient(candidate)
    except ValueError:
        return default


def parse_float(text: str, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    """Parse a float from text (e.g. '0.8', '80%', '7/10'), clamped to [lo, hi]."""
    if text is None:
        return default
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        value = float(m.group(1)) / 100.0
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", text)
        if m:
            denom = float(m.group(2))
            value = float(m.group(1)) / denom if denom else default
        else:
            m = re.search(r"-?\d+(?:\.\d+)?", text)
            if not m:
                return default
            value = float(m.group(0))
    return max(lo, min(hi, value))


def parse_score(text: str, lo: int = 0, hi: int = 10, default: int = 5) -> int:
    """Parse an integer score from text, clamped to [lo, hi]."""
    value = parse_float(text, lo=lo, hi=hi, default=float(default))
    return int(round(value))
