"""Trace event bus: in-process fan-out of pipeline events to subscribers
(WebSocket broadcast, trace store, console logging)."""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TraceEvent:
    task_id: str
    phase: str
    event_type: str  # phase_start | phase_end | llm_call | critique | debate | info | error | result
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "event_type": self.event_type,
            "data": self.data,
            "ts": self.ts,
        }


Subscriber = Callable[["TraceEvent"], Any]


class EventBus:
    """Fan-out bus. Subscribers may be sync or async callables."""

    def __init__(self):
        self._subscribers: list[Subscriber] = []
        self._lock = asyncio.Lock()

    def subscribe(self, callback: Subscriber) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def emit(self, event: TraceEvent) -> None:
        for cb in list(self._subscribers):
            try:
                result = cb(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # A broken subscriber must never break the pipeline.
                pass

    def emit_nowait(self, event: TraceEvent) -> None:
        """Synchronous fire-and-forget for sync contexts."""
        for cb in list(self._subscribers):
            try:
                result = cb(event)
                if inspect.isawaitable(result):
                    asyncio.get_event_loop().create_task(result)
            except Exception:
                pass

    def make_emitter(self, task_id: str) -> Callable:
        """Return an async emitter bound to a task:
        ``await emit(phase, event_type, **data)``."""

        async def emit(phase: str, event_type: str, **data: Any) -> None:
            await self.emit(TraceEvent(task_id=task_id, phase=phase, event_type=event_type, data=data))

        return emit
