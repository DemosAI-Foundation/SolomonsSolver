"""Global FINISH NOW / STOP interrupt manager (v2.6).

Implements the requirement that ``FINISH NOW`` be a *real global interrupt*
that works at any moment — including mid-stream:

- a per-task ``finish_event`` that every phase checks before and after each
  LLM call (and the streaming transport checks per chunk),
- immediate side effects the moment the button is pressed: kill sandbox
  processes (by process group) and cancel the current pipeline task,
- capture of the :class:`ActiveContext` snapshot for the emergency finish,
- idempotent repeated presses (pressing twice == pressing once).

The single emergency synthesis call is *not* issued here — the pipeline runs
it in a separate, non-cancelled task (see ``V2Pipeline._finish``) so the
interrupt cannot starve the very call that produces the partial answer.
"""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from src.agents.control import SessionRegistry
    from src.v2.active_context import ActiveContext
    from src.v2.sandbox import SandboxAdapter


class FinishControl:
    """Per-task interrupt state. The ``finish_event`` is the global FINISH NOW
    flag for this task: once set, every gate check (phase boundary, in-flight
    LLM call, streaming chunk) sees the interrupt and unwinds immediately."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        # The global FINISH NOW / STOP events for this task.
        self.finish_event = threading.Event()
        self.stop_event = threading.Event()
        self.pipeline_task: Optional[asyncio.Task] = None
        self.sandbox: Optional["SandboxAdapter"] = None
        self.sessions: Optional["SessionRegistry"] = None
        self.last_snapshot: Optional[dict] = None
        # Counts presses so repeated presses are observable but idempotent.
        self.finish_presses = 0
        self.stop_presses = 0

    # -- requests (idempotent) ------------------------------------------ #
    def request_finish(self) -> bool:
        """Set ``finish_event`` and apply the immediate side effects.

        Returns True when this task is known and the interrupt is now pending
        (or was already pending — repeated presses are idempotent).
        """
        already = self.finish_event.is_set()
        self.finish_presses += 1
        self.finish_event.set()
        if not already:
            self._mirror_to_sessions("finish")
            self.kill_sandbox()
            self.cancel_pipeline()
        return True

    def request_stop(self) -> bool:
        already = self.stop_event.is_set() or self.finish_event.is_set()
        self.stop_presses += 1
        self.stop_event.set()
        if not already:
            self._mirror_to_sessions("stop")
            self.kill_sandbox()
            self.cancel_pipeline()
        return True

    def _mirror_to_sessions(self, signal: str) -> None:
        if self.sessions is None:
            return
        sess = self.sessions.get(self.task_id)
        if sess is None:
            return
        if signal == "finish":
            sess.request_finish_now()
        else:
            sess.request_stop()

    # -- state ----------------------------------------------------------- #
    def signal(self) -> str:
        """Highest-priority pending signal: ``"stop"`` > ``"finish"``."""
        if self.stop_event.is_set():
            return "stop"
        if self.finish_event.is_set():
            return "finish"
        return "continue"

    def is_finished(self) -> bool:
        return self.finish_event.is_set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    # -- side effects ----------------------------------------------------- #
    def bind_pipeline_task(self, task: "asyncio.Task") -> None:
        """Register the asyncio task running the pipeline so a FINISH NOW can
        cancel it directly (works even between gate checks, mid-stream)."""
        self.pipeline_task = task

    def set_sandbox(self, sandbox: Optional["SandboxAdapter"]) -> None:
        self.sandbox = sandbox

    def cancel_pipeline(self) -> bool:
        task = self.pipeline_task
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def kill_sandbox(self) -> bool:
        sandbox = self.sandbox
        if sandbox is None:
            return False
        try:
            return bool(sandbox.kill())
        except Exception:  # noqa: BLE001
            return False

    # -- snapshot --------------------------------------------------------- #
    def snapshot(self, context: Optional["ActiveContext"]) -> Optional[dict]:
        """Capture the ActiveContext snapshot for the emergency finish."""
        if context is None:
            return self.last_snapshot
        snap = context.to_dict()
        self.last_snapshot = snap
        return snap


class FinishManager:
    """Global registry of per-task FINISH NOW / STOP controls.

    The pipeline consults :meth:`signal` before and after every phase and every
    LLM call; the streaming LLM transport consults it per chunk. Pressing
    FINISH NOW sets the task's ``finish_event`` and immediately kills sandbox
    processes and cancels the pipeline task, so the interrupt works at any
    moment. Repeated presses are idempotent.
    """

    def __init__(self, sessions: Optional["SessionRegistry"] = None):
        self.sessions = sessions
        self._controls: dict[str, FinishControl] = {}
        self._lock = threading.Lock()

    # -- registry ---------------------------------------------------------- #
    def register(self, task_id: str) -> FinishControl:
        with self._lock:
            control = self._controls.get(task_id)
            if control is None:
                control = FinishControl(task_id)
                control.sessions = self.sessions
                self._controls[task_id] = control
            return control

    def get(self, task_id: str) -> Optional[FinishControl]:
        with self._lock:
            return self._controls.get(task_id)

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._controls.pop(task_id, None)

    def active(self) -> list[FinishControl]:
        with self._lock:
            return list(self._controls.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._controls)

    # -- user-facing requests (idempotent) --------------------------------- #
    def request_finish(self, task_id: str) -> bool:
        """Press FINISH NOW for a task. True if the interrupt is now pending.

        Unknown tasks (never registered and no session registry) are rejected.
        """
        control = self.get(task_id)
        if control is None:
            if self.sessions is None or self.sessions.get(task_id) is None:
                return False
            control = self.register(task_id)
        return control.request_finish()

    def request_stop(self, task_id: str) -> bool:
        control = self.get(task_id)
        if control is None:
            if self.sessions is None or self.sessions.get(task_id) is None:
                return False
            control = self.register(task_id)
        return control.request_stop()

    # -- pipeline-facing ---------------------------------------------------- #
    def signal(self, task_id: str) -> str:
        control = self.get(task_id)
        if control is None:
            return "continue"
        return control.signal()

    def snapshot(self, task_id: str, context: Optional["ActiveContext"]) -> Optional[dict]:
        control = self.get(task_id)
        if control is None:
            return None
        return control.snapshot(context)
