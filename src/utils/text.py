"""Text helpers shared across agents."""
from __future__ import annotations

import re

_ANSWER_MARKERS = (
    "FINAL ANSWER:",
    "ANSWER:",
    "ANSWER :",
    "Final Answer:",
)


def extract_answer(text: str) -> str:
    """Pull the final answer out of a response.

    Prefers an explicit ANSWER: marker (last occurrence wins); otherwise
    returns the last non-empty line, falling back to the full text.
    """
    if text is None:
        return ""
    text = text.strip()
    best: str | None = None
    for marker in _ANSWER_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            candidate = text[idx + len(marker) :].strip()
            if candidate:
                best = candidate
    if best is not None:
        # Keep only the first line/paragraph of the marked answer.
        return best.split("\n\n")[0].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_all(text: str, *needles: str) -> bool:
    """Case-insensitive check that `text` contains every needle."""
    norm = normalize_text(text)
    return all(normalize_text(n) in norm for n in needles)
