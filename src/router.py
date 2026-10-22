"""Task Router: maps every user-facing mode to the V2 verifier-first pipeline.

- ``lightning`` -> ``lightning`` (the short L0-L3 pipeline: fast, labeled
  "Lightning answer: fast, not deeply verified.")
- ``deep``      -> ``standard`` (medium budget, full Phase 0-8)
- ``omega``     -> ``deep``     (slowest, most budget, full Phase 0-8)
- ``auto``      -> the configured default V2 mode
- ``v2``        -> an explicit ``v2_mode`` (or the pipeline default)
"""
from __future__ import annotations

from config.settings import Settings, get_settings
from src.llm.orchestrator import LLMOrchestrator
from src.models.dataclasses import (
    AgentResult,
    Task,
    MODE_V2,
    TIER_TO_V2_MODE,
)
from src.monitoring.events import EventBus

# Reverse of TIER_TO_V2_MODE: maps a V2 budget mode back to a tier name so the
# ``auto`` mode can report a concrete tier for trace/dashboard bookkeeping.
V2_MODE_TO_TIER = {v: k for k, v in TIER_TO_V2_MODE.items()}


class TaskRouter:
    """Dispatches tasks to the V2 pipeline with a mode-mapped budget."""

    def __init__(
        self,
        llm: LLMOrchestrator,
        event_bus: EventBus | None = None,
        settings: Settings | None = None,
        v2_pipeline=None,
    ):
        self.llm = llm
        self.event_bus = event_bus
        self.settings = settings or get_settings()
        self.v2_pipeline = v2_pipeline

    def _emitter(self, task_id: str):
        if self.event_bus is None:
            async def _noop(phase: str, event_type: str, **data) -> None:
                pass

            return _noop
        return self.event_bus.make_emitter(task_id)

    def _map_mode(self, mode: str, v2_mode: str | None) -> tuple[str, str | None]:
        """Return ``(tier, v2_mode)`` for a user-facing mode."""
        if mode == MODE_V2:
            return "v2", v2_mode
        if mode in TIER_TO_V2_MODE:
            return mode, TIER_TO_V2_MODE[mode]
        # auto -> the configured default V2 mode, reported as its tier name.
        default_mode = self.settings.v2.default_mode
        tier = V2_MODE_TO_TIER.get(default_mode, "deep")
        return tier, default_mode

    async def route(
        self, task: Task, mode: str = "auto", v2_mode: str | None = None
    ) -> AgentResult:
        """Route a task through the V2 verifier-first pipeline."""
        emit = self._emitter(task.id)
        await emit("router", "phase_start", mode=mode)

        tier, v2m = self._map_mode(mode, v2_mode)
        task.tier = tier
        await emit("router", "phase_end", mode=mode, tier=tier, v2_mode=v2m)

        if self.v2_pipeline is None:
            return self._create_failure_result(task, "v2 pipeline is not configured")

        try:
            return await self.v2_pipeline.run(task, mode=v2m)
        except Exception as exc:  # noqa: BLE001
            await emit("router", "error", tier=tier, error=str(exc))
            return self._create_failure_result(task, f"v2: {exc}")

    @staticmethod
    def _create_failure_result(task: Task, error: str) -> AgentResult:
        return AgentResult(
            task_id=task.id,
            final_answer="",
            reasoning_trace="",
            tier=task.tier,
            iterations=0,
            total_latency=0.0,
            techniques_used=[],
            failed=True,
            error=error,
        )
