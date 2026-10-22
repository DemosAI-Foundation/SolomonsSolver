"""Context isolation — the technical implementation of Solomon's Paradox.

Every critique / verification call gets a completely fresh context window.
The critic sees only what is explicitly injected (task, trace, answer) and
never the generator's conversation history.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ContextWindow:
    window_id: str
    task_id: str
    purpose: str  # e.g. "critique", "cove_answer", "debate_logic"
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    isolated: bool = True

    def render(self) -> str:
        """Render the window contents for audit/inspection."""
        return "\n\n".join(
            f"[{m['role']}] {m['content']}" for m in self.messages
        )

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "task_id": self.task_id,
            "purpose": self.purpose,
            "isolated": self.isolated,
            "message_count": len(self.messages),
            "created_at": self.created_at,
        }


class ContextManager:
    """Creates and audits isolated context windows per task."""

    def __init__(self):
        self._windows: dict[str, ContextWindow] = {}

    def create_fresh_context(self, task_id: str, purpose: str = "generic") -> ContextWindow:
        """Create a new context window with zero shared state."""
        window = ContextWindow(
            window_id=uuid.uuid4().hex[:12],
            task_id=task_id,
            purpose=purpose,
        )
        self._windows[window.window_id] = window
        return window

    def inject_task_only(
        self,
        context: ContextWindow,
        task: str,
        trace: str | None = None,
        answer: str | None = None,
    ) -> ContextWindow:
        """Inject ONLY the specified content into a fresh context.

        The critic sees task + trace + answer, and nothing else: no prior
        turns, no generator scratchpad, no other agents' output.
        """
        parts = [f"TASK:\n{task}"]
        if trace:
            parts.append(f"REASONING TRACE:\n{trace}")
        if answer:
            parts.append(f"ANSWER:\n{answer}")
        context.messages.append({"role": "user", "content": "\n\n".join(parts)})
        return context

    def get_window(self, window_id: str) -> ContextWindow | None:
        return self._windows.get(window_id)

    def windows_for_task(self, task_id: str) -> list[ContextWindow]:
        return [w for w in self._windows.values() if w.task_id == task_id]

    def verify_isolation(self, task_id: str) -> dict:
        """Audit: every window for this task must be fresh (isolated) and
        contain only explicitly injected content (no assistant history)."""
        windows = self.windows_for_task(task_id)
        violations = []
        for w in windows:
            if not w.isolated:
                violations.append(f"{w.window_id}: not marked isolated")
            roles = [m["role"] for m in w.messages]
            if "assistant" in roles or "system" in roles:
                violations.append(f"{w.window_id}: contains non-user history {roles}")
        return {
            "task_id": task_id,
            "window_count": len(windows),
            "isolated": not violations,
            "violations": violations,
        }

    def clear(self, task_id: str | None = None) -> None:
        if task_id is None:
            self._windows.clear()
        else:
            self._windows = {
                wid: w for wid, w in self._windows.items() if w.task_id != task_id
            }
