"""Application wiring: builds and connects all system components."""
from __future__ import annotations

import asyncio
import time

from config.settings import Settings, get_settings
from src.agents.control import SessionRegistry
from src.agents.synthesis import SynthesisStep
from src.llm.orchestrator import LLMOrchestrator
from src.models.dataclasses import AgentResult, Task, new_task_id
from src.monitoring.confidence import ConfidenceMonitor
from src.monitoring.dashboard import MonitoringDashboard
from src.monitoring.events import EventBus, TraceEvent
from src.monitoring.trace_store import TraceStore
from src.router import TaskRouter
from src.v2 import (
    FinishManager,
    RetrievalAdapter,
    SandboxAdapter,
    TokenBudget,
    V2Pipeline,
    make_subprocess_runner,
)


class SolomonApp:
    """Owns the full object graph: LLM, agents, router, stores, dashboard."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.event_bus = EventBus()
        self.trace_store = TraceStore(self.settings.db_path)
        self.dashboard = MonitoringDashboard()

        # FINISH NOW / STOP infrastructure (created before the LLM so the
        # orchestrator can race in-flight calls against pending control signals).
        self.sessions = SessionRegistry()
        # v2.6: the global FINISH NOW / STOP manager — the real global
        # interrupt (per-task finish_event + sandbox kill + task cancel).
        self.finish_manager = FinishManager(sessions=self.sessions)
        self.llm = LLMOrchestrator(
            self.settings.llm, event_bus=self.event_bus, gate=self._control_gate
        )
        self.synthesis = SynthesisStep(self.llm)
        self.confidence_monitor = ConfidenceMonitor(
            event_bus=self.event_bus,
            threshold=self.settings.pipeline.confidence_threshold,
        )

        # v2.0 verifier-first pipeline (the only pipeline)
        v2 = self.settings.v2
        # No retrieval backend is wired in yet -> safe downgrade (ungrounded).
        self.v2_retrieval = RetrievalAdapter(backend=None)
        self.v2_sandbox = SandboxAdapter(
            runner=(
                make_subprocess_runner(timeout=v2.sandbox_timeout)
                if v2.sandbox_enabled
                else None
            ),
            timeout=v2.sandbox_timeout,
        )
        self.v2_pipeline = V2Pipeline(
            self.llm,
            self.event_bus,
            settings=self.settings,
            sessions=self.sessions,
            synthesis=self.synthesis,
            token_budget=TokenBudget(
                window=v2.max_context_window,
                reserves=dict(v2.phase_reserves),
            ),
            retrieval=self.v2_retrieval,
            sandbox=self.v2_sandbox,
            mode=v2.default_mode,
            finish_manager=self.finish_manager,
        )
        self.router = TaskRouter(
            self.llm,
            self.event_bus,
            self.settings,
            v2_pipeline=self.v2_pipeline,
        )

        self.event_bus.subscribe(self._persist_event)
        self.event_bus.subscribe(self.confidence_monitor.observe)
        self._ws_subscribers: dict[str, set] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}

    def _control_gate(self, task_id: str) -> str:
        """Return the pending STOP / FINISH NOW signal for a task (sync).

        Passed to the LLM orchestrator so in-flight calls are cancelled the
        moment a control signal appears, making FINISH NOW / STOP immediate.
        v2.6: the global finish_event (FinishManager) is checked first — it is
        also the signal the streaming transport checks on every chunk.
        """
        sig = self.finish_manager.signal(task_id)
        if sig != "continue":
            return sig
        sess = self.sessions.get(task_id)
        if sess is None:
            return "continue"
        return sess.signal()

    async def _persist_event(self, event: TraceEvent) -> None:
        self.trace_store.record_event(
            event.task_id, event.phase, event.event_type, event.data, ts=event.ts
        )

    # ---------- task lifecycle ----------

    def submit_task(
        self,
        prompt: str,
        task_id: str | None = None,
        mode: str = "auto",
        v2_mode: str | None = None,
    ) -> Task:
        task = Task(id=task_id or new_task_id(), original_prompt=prompt)
        self.trace_store.upsert_task(task)
        self._active_tasks[task.id] = asyncio.create_task(
            self._run_task(task, mode=mode, v2_mode=v2_mode)
        )
        return task

    async def _run_task(
        self, task: Task, mode: str = "auto", v2_mode: str | None = None
    ) -> None:
        try:
            result = await self.router.route(task, mode=mode, v2_mode=v2_mode)
        except Exception as exc:  # noqa: BLE001
            result = AgentResult(
                task_id=task.id,
                final_answer="",
                reasoning_trace="",
                tier=task.tier,
                failed=True,
                error=str(exc),
            )
        self.dashboard.record_task(result)
        self.trace_store.finish_task(result)
        await self.event_bus.emit(
            TraceEvent(
                task_id=task.id,
                phase="result",
                event_type="result",
                data=result.to_dict(),
            )
        )
        self._mark_finished(task.id)
        self.confidence_monitor.forget(task.id)
        self._active_tasks.pop(task.id, None)

    def _mark_finished(self, task_id: str) -> None:
        sess = self.sessions.get(task_id)
        if sess is not None:
            sess.finished = True

    # ---------- FINISH NOW / STOP control ----------

    def request_finish_now(self, task_id: str) -> bool:
        """Ask a running task to synthesize and stop. Returns True if pending.

        v2.6: routes through the global FinishManager, which sets the task's
        ``finish_event``, kills sandbox processes (by process group), and
        cancels the pipeline task — a real global interrupt that works at any
        moment, including mid-stream. Idempotent for repeated presses."""
        if self.sessions.get(task_id) is None:
            return False
        return self.finish_manager.request_finish(task_id)

    def request_stop(self, task_id: str) -> bool:
        """Ask a running task to stop and return its best answer so far.

        v2.6: routes through the global FinishManager (stop_event + sandbox
        kill + task cancel). Idempotent for repeated presses."""
        if self.sessions.get(task_id) is None:
            return False
        return self.finish_manager.request_stop(task_id)

    def task_state(self, task_id: str) -> dict | None:
        sess = self.sessions.get(task_id)
        if sess is None:
            return None
        return sess.to_dict()

    # ---------- websocket fan-out ----------

    def register_ws(self, task_id: str, ws) -> None:
        self._ws_subscribers.setdefault(task_id, set()).add(ws)

    def unregister_ws(self, task_id: str, ws) -> None:
        subs = self._ws_subscribers.get(task_id)
        if subs:
            subs.discard(ws)

    async def broadcast(self, task_id: str, payload: dict) -> None:
        for ws in list(self._ws_subscribers.get(task_id, ())):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001
                self.unregister_ws(task_id, ws)

    async def close(self) -> None:
        await self.llm.close()
        self.trace_store.close()
