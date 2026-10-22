"""Tests for the Task Router: mode -> V2 budget mapping and dispatch."""
import pytest

from src.models.dataclasses import AgentResult, Task, new_task_id
from src.router import TaskRouter, V2_MODE_TO_TIER


class StubV2Pipeline:
    """Records the budget mode the router dispatches to the V2 pipeline."""

    def __init__(self, answer="v2 answer"):
        self.answer = answer
        self.calls: list[str] = []

    async def run(self, task: Task, mode: str | None = None) -> AgentResult:
        self.calls.append(mode)
        return AgentResult(
            task_id=task.id,
            final_answer=self.answer,
            reasoning_trace="v2 trace",
            tier=task.tier,
        )


class ExplodingV2:
    async def run(self, task, mode=None):
        raise RuntimeError("pipeline exploded")


def _make_router(settings, v2=None):
    return TaskRouter(None, event_bus=None, settings=settings, v2_pipeline=v2)


class TestModeMapping:
    def test_tier_to_v2_mode_map(self):
        # v2.6: every tier runs the v2 pipeline; lightning maps to the short
        # L0-L3 "lightning" budget (not the pre-v2.6 "quick" mode).
        assert V2_MODE_TO_TIER == {
            "lightning": "lightning",
            "standard": "deep",
            "deep": "omega",
        }

    async def test_lightning_routes_to_lightning(self, settings):
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        result = await router.route(task, mode="lightning")
        assert v2.calls == ["lightning"]
        assert task.tier == "lightning"
        assert result.final_answer == "v2 answer"

    async def test_deep_routes_to_standard(self, settings):
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        await router.route(task, mode="deep")
        assert v2.calls == ["standard"]
        assert task.tier == "deep"

    async def test_omega_routes_to_deep(self, settings):
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        await router.route(task, mode="omega")
        assert v2.calls == ["deep"]
        assert task.tier == "omega"

    async def test_v2_mode_passes_explicit_v2_mode(self, settings):
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        await router.route(task, mode="v2", v2_mode="deep")
        assert v2.calls == ["deep"]
        assert task.tier == "v2"

    async def test_v2_mode_defaults_to_none(self, settings):
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        await router.route(task, mode="v2")
        assert v2.calls == [None]
        assert task.tier == "v2"

    async def test_auto_routes_to_default_mode(self, settings):
        settings.v2.default_mode = "standard"
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        await router.route(task, mode="auto")
        assert v2.calls == ["standard"]
        assert task.tier == "deep"

    async def test_auto_uses_configured_default_mode(self, settings):
        # v2.6: the configured default is a v2 budget mode (lightning here).
        settings.v2.default_mode = "lightning"
        v2 = StubV2Pipeline()
        router = _make_router(settings, v2)
        task = Task(id=new_task_id(), original_prompt="x")
        await router.route(task, mode="auto")
        assert v2.calls == ["lightning"]
        assert task.tier == "lightning"


class TestDispatch:
    async def test_pipeline_failure_returns_failure_result(self, settings):
        router = _make_router(settings, ExplodingV2())
        task = Task(id=new_task_id(), original_prompt="x")
        result = await router.route(task, mode="lightning")
        assert result.failed is True
        assert "pipeline exploded" in result.error

    async def test_no_pipeline_returns_failure(self, settings):
        router = _make_router(settings, None)
        task = Task(id=new_task_id(), original_prompt="x")
        result = await router.route(task, mode="lightning")
        assert result.failed is True
        assert "not configured" in result.error
