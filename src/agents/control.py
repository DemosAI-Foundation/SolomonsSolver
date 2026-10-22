"""Per-task control sessions and the registry that holds them.

A ``TaskSession`` carries the live :class:`AgentState` for a running task and
two user-initiated control signals:

- ``stop_requested``  -> the agent returns its best answer so far (aborted).
- ``finish_now_requested`` -> the agent runs one synthesis step and returns
  an early-completion result.

Signals are checked by agents at phase boundaries AND polled during in-flight
LLM calls (the orchestrator races the call against the signal and cancels it),
so a request takes effect immediately without waiting for the current step to
finish. The FINISH NOW synthesis call itself bypasses the gate so it can
complete.
"""
from __future__ import annotations

import threading

from src.agents.state import AgentState


class TaskSession:
    """Live control handle for a single running task."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.state: AgentState | None = None
        self.stop_requested = False
        self.finish_now_requested = False
        self.finished = False

    def request_stop(self) -> None:
        self.stop_requested = True

    def request_finish_now(self) -> None:
        self.finish_now_requested = True

    def signal(self) -> str:
        """Return the highest-priority pending control signal.

        ``"stop"`` beats ``"finish"``. Returns ``"continue"`` when nothing is
        pending.
        """
        if self.stop_requested:
            return "stop"
        if self.finish_now_requested:
            return "finish"
        return "continue"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "state": self.state.to_dict() if self.state else None,
            "stop_requested": self.stop_requested,
            "finish_now_requested": self.finish_now_requested,
            "finished": self.finished,
        }


class SessionRegistry:
    """Thread-safe registry of live task sessions, keyed by task id."""

    def __init__(self):
        self._sessions: dict[str, TaskSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, task_id: str) -> TaskSession:
        with self._lock:
            sess = self._sessions.get(task_id)
            if sess is None:
                sess = TaskSession(task_id)
                self._sessions[task_id] = sess
            return sess

    def get(self, task_id: str) -> TaskSession | None:
        with self._lock:
            return self._sessions.get(task_id)

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._sessions.pop(task_id, None)

    def active(self) -> list[TaskSession]:
        with self._lock:
            return list(self._sessions.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
