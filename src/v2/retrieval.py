"""Mechanic H — retrieval grounding + facts ledger, with safe downgrade.

If a retrieval backend is wired in, the pipeline grounds its facts in retrieved
evidence. If no backend is available (the common case here), it degrades
gracefully: retrieval is marked unavailable and any factual claims the model
makes are flagged as ungrounded rather than silently trusted.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Union

from src.v2.contracts import FactsLedgerEntry

# A retrieval backend is an (optionally async) callable: query -> list[str].
RetrievalBackend = Union[
    Callable[[str], list[str]],
    Callable[[str], Awaitable[list[str]]],
]


class RetrievalAdapter:
    def __init__(self, backend: Optional[RetrievalBackend] = None):
        self.backend = backend

    @property
    def available(self) -> bool:
        return self.backend is not None

    async def retrieve(
        self, task: str, task_id: str = "", emit=None
    ) -> tuple[bool, list[str]]:
        """Return ``(available, facts)``. Never raises — degrades to
        ``(False, [])`` when the backend is missing or errors."""
        if self.backend is None:
            return (False, [])
        try:
            result = self.backend(task)
            if hasattr(result, "__await__"):
                result = await result
            facts = [str(r).strip() for r in (result or []) if str(r).strip()]
            if emit is not None:
                await emit("phase0", "retrieval", available=True, n_facts=len(facts))
            return (True, facts)
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                await emit("phase0", "retrieval", available=False, error=str(exc))
            return (False, [])


def build_facts_ledger(
    retrieved: list[str], available: bool, source: str = "retrieval"
) -> list[FactsLedgerEntry]:
    """Turn raw retrieval results into grounded facts-ledger entries."""
    if not available:
        return []
    return [
        FactsLedgerEntry(fact=f, source=source, grounded=True, confidence=0.7)
        for f in retrieved
    ]


def mark_ungrounded(claims: list[str]) -> list[FactsLedgerEntry]:
    """Represent model-asserted claims that were never grounded in retrieval."""
    return [
        FactsLedgerEntry(fact=c, source="ungrounded", grounded=False, confidence=0.3)
        for c in claims
    ]
