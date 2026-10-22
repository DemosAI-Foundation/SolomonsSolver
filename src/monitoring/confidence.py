"""Confidence monitor: turns pipeline events into live confidence signals.

Subscribed to the :class:`EventBus`, it watches each task's ``progress``,
``critique``, and self-consistency events and emits a single ``confidence``
event per observation with:

- ``confidence``: 0.0-1.0 score,
- ``level``: qualitative ``high`` / ``medium`` / ``low``,
- ``converged``: whether the best answer has stabilized,
- ``finish_now_eligible``: whether the UI may enable the FINISH NOW button,
- ``steps_done`` / ``steps_total`` / ``phase``: progress for the button label.

The monitor only *suggests* eligibility — it never auto-triggers anything.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.monitoring.events import EventBus, TraceEvent


@dataclass
class _TaskConfidence:
    answers: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    steps_done: int = 0
    steps_total: int = 0
    current_phase: str = ""
    consensus_conf: float | None = None
    consensus_answer: str | None = None


class ConfidenceMonitor:
    """Watches trace events and emits live confidence signals per task."""

    def __init__(self, event_bus: EventBus | None = None, threshold: float = 0.7):
        self._bus = event_bus
        self._threshold = threshold
        self._tasks: dict[str, _TaskConfidence] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # event handling
    # ------------------------------------------------------------------ #
    async def observe(self, event: TraceEvent) -> None:
        """Handle one trace event. Safe to use as an EventBus subscriber."""
        if event.event_type == "confidence":
            return  # ignore our own emissions
        with self._lock:
            t = self._tasks.setdefault(event.task_id, _TaskConfidence())
        self._update(t, event)
        payload = self._compute(t)
        await self._emit(event.task_id, payload)

    def _update(self, t: _TaskConfidence, event: TraceEvent) -> None:
        if event.event_type == "progress":
            ans = event.data.get("best_answer")
            if ans:
                t.answers.append(ans)
            t.steps_done = int(event.data.get("steps_done", t.steps_done))
            t.steps_total = int(event.data.get("steps_total", t.steps_total))
            t.current_phase = event.data.get("current_phase", t.current_phase)
        elif event.event_type == "critique":
            verdict = event.data.get("verdict")
            if verdict:
                t.verdicts.append(verdict)
        elif event.phase == "self_consistency" and event.event_type == "phase_end":
            t.consensus_conf = event.data.get("confidence", t.consensus_conf)
            t.consensus_answer = event.data.get("consensus", t.consensus_answer)

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _converged(self, t: _TaskConfidence) -> bool:
        if len(t.answers) < 2:
            return False
        a, b = t.answers[-1], t.answers[-2]
        return bool(a) and a == b

    def _score(self, t: _TaskConfidence) -> float:
        score = 0.0
        if t.answers and t.answers[-1]:
            score = max(score, 0.4)
        if t.consensus_conf is not None:
            score = max(score, float(t.consensus_conf))
        if t.verdicts and t.verdicts[-1] == "pass":
            score = max(score, 0.85)
        if self._converged(t):
            score = max(score, 0.75)
        return round(min(1.0, score), 3)

    @staticmethod
    def _level(score: float) -> str:
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _compute(self, t: _TaskConfidence) -> dict:
        score = self._score(t)
        return {
            "confidence": score,
            "level": self._level(score),
            "converged": self._converged(t),
            "finish_now_eligible": bool(t.answers) and score >= self._threshold,
            "steps_done": t.steps_done,
            "steps_total": t.steps_total,
            "phase": t.current_phase,
        }

    async def _emit(self, task_id: str, payload: dict) -> None:
        if self._bus is not None:
            await self._bus.emit(
                TraceEvent(
                    task_id=task_id,
                    phase="confidence",
                    event_type="confidence",
                    data=payload,
                )
            )

    def snapshot(self, task_id: str) -> dict | None:
        with self._lock:
            t = self._tasks.get(task_id)
        return self._compute(t) if t else None

    def forget(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)
