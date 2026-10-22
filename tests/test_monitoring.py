"""Tests for monitoring: event bus, SQLite trace store, dashboard metrics."""
import pytest

from src.models.dataclasses import AgentResult, Task, new_task_id
from src.monitoring.dashboard import MonitoringDashboard
from src.monitoring.events import EventBus, TraceEvent
from src.monitoring.trace_store import TraceStore


class TestEventBus:
    async def test_sync_subscriber_receives_events(self):
        bus = EventBus()
        seen = []
        bus.subscribe(seen.append)
        await bus.emit(TraceEvent(task_id="t1", phase="p", event_type="info"))
        assert len(seen) == 1
        assert seen[0].task_id == "t1"

    async def test_async_subscriber_is_awaited(self):
        bus = EventBus()
        seen = []

        async def cb(event):
            seen.append(event)

        bus.subscribe(cb)
        await bus.emit(TraceEvent(task_id="t1", phase="p", event_type="info"))
        assert len(seen) == 1

    async def test_broken_subscriber_does_not_break_others(self):
        bus = EventBus()
        seen = []

        def broken(event):
            raise RuntimeError("subscriber exploded")

        bus.subscribe(broken)
        bus.subscribe(seen.append)
        await bus.emit(TraceEvent(task_id="t1", phase="p", event_type="info"))
        assert len(seen) == 1

    async def test_unsubscribe(self):
        bus = EventBus()
        seen = []
        bus.subscribe(seen.append)
        await bus.emit(TraceEvent(task_id="t1", phase="p", event_type="info"))
        bus.unsubscribe(seen.append)
        await bus.emit(TraceEvent(task_id="t1", phase="p", event_type="info"))
        assert len(seen) == 1

    async def test_make_emitter_binds_task(self):
        bus = EventBus()
        seen = []
        bus.subscribe(seen.append)
        emit = bus.make_emitter("bound-task")
        await emit("reframe", "phase_start", detail="x")
        assert seen[0].task_id == "bound-task"
        assert seen[0].phase == "reframe"
        assert seen[0].event_type == "phase_start"
        assert seen[0].data == {"detail": "x"}


class TestTraceStore:
    def test_upsert_and_finish_task(self, tmp_path):
        store = TraceStore(tmp_path / "test.db")
        task = Task(id="abc123", original_prompt="What is 2+2?")
        store.upsert_task(task)
        row = store.get_task("abc123")
        assert row is not None
        assert row["status"] == "running"
        assert row["prompt"] == "What is 2+2?"

        result = AgentResult(
            task_id="abc123",
            final_answer="4",
            reasoning_trace="1+1+1+1",
            tier="lightning",
            iterations=1,
            total_latency=0.5,
            total_cost=0.0,
            tokens_prompt=100,
            tokens_completion=50,
            techniques_used=["third_person_reframe"],
        )
        store.finish_task(result)
        row = store.get_task("abc123")
        assert row["status"] == "done"
        assert row["final_answer"] == "4"
        assert row["iterations"] == 1
        assert row["tokens_prompt"] == 100
        assert row["techniques"] == ["third_person_reframe"]
        assert row["failed"] == 0
        store.close()

    def test_upsert_conflict_updates_fields(self, tmp_path):
        store = TraceStore(tmp_path / "test.db")
        task = Task(id="x1", original_prompt="p", tier="lightning")
        store.upsert_task(task)
        task.tier = "deep"
        task.complexity_score = 0.55
        store.upsert_task(task)
        rows = [r for r in store.list_tasks() if r["id"] == "x1"]
        assert len(rows) == 1
        assert rows[0]["tier"] == "deep"
        assert rows[0]["complexity"] == pytest.approx(0.55)
        store.close()

    def test_list_tasks_ordered_newest_first(self, tmp_path):
        store = TraceStore(tmp_path / "test.db")
        t1 = Task(id="old", original_prompt="p1", created_at=100.0)
        t2 = Task(id="new", original_prompt="p2", created_at=200.0)
        store.upsert_task(t1)
        store.upsert_task(t2)
        ids = [r["id"] for r in store.list_tasks()]
        assert ids == ["new", "old"]
        store.close()

    def test_get_missing_task_returns_none(self, tmp_path):
        store = TraceStore(tmp_path / "test.db")
        assert store.get_task("nope") is None
        store.close()

    def test_record_and_fetch_events(self, tmp_path):
        store = TraceStore(tmp_path / "test.db")
        store.record_event("t1", "reframe", "phase_start", {"a": 1}, ts=1.0)
        store.record_event("t1", "solve", "llm_call", {"b": 2}, ts=2.0)
        store.record_event("t2", "reframe", "phase_start", ts=3.0)
        events = store.get_task_events("t1")
        assert len(events) == 2
        assert events[0]["phase"] == "reframe"
        assert events[0]["data"] == {"a": 1}
        assert events[1]["event_type"] == "llm_call"
        assert store.get_task_events("t2")[0]["phase"] == "reframe"
        assert store.get_task_events("t3") == []
        store.close()

    def test_failed_task_stored(self, tmp_path):
        store = TraceStore(tmp_path / "test.db")
        task = Task(id="f1", original_prompt="p")
        store.upsert_task(task)
        result = AgentResult(
            task_id="f1",
            final_answer="",
            reasoning_trace="",
            tier="omega",
            failed=True,
            error="omega: boom",
            escalated_from="deep",
        )
        store.finish_task(result)
        row = store.get_task("f1")
        assert row["failed"] == 1
        assert row["error"] == "omega: boom"
        assert row["escalated_from"] == "deep"
        store.close()


class _FakeCritique:
    def __init__(self, verdict):
        self.verdict = verdict


class TestMonitoringDashboard:
    def _result(self, tier="lightning", verdicts=None, failed=False, error="", escalated_from=None, techs=None):
        history = [_FakeCritique(verdict=v) for v in (verdicts or [])]
        return AgentResult(
            task_id=new_task_id(),
            final_answer="a",
            reasoning_trace="t",
            tier=tier,
            critique_history=history,
            iterations=2,
            total_latency=1.5,
            total_cost=0.25,
            tokens_prompt=10,
            tokens_completion=20,
            techniques_used=techs or ["x"],
            failed=failed,
            error=error,
            escalated_from=escalated_from,
        )

    def test_counts_and_averages(self):
        d = MonitoringDashboard()
        d.record_task(self._result(tier="lightning", verdicts=["pass"]))
        d.record_task(self._result(tier="deep", verdicts=["revise", "pass"]))
        snap = d.get_dashboard()
        assert snap["tasks_processed"] == 2
        assert snap["tasks_by_tier"]["lightning"] == 1
        assert snap["tasks_by_tier"]["deep"] == 1
        assert snap["critique_pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert snap["critique_verdicts"] == {"pass": 2, "revise": 1}
        assert snap["refine_iterations_avg"] == pytest.approx(2.0)
        assert snap["avg_latency_by_tier"]["lightning"] == pytest.approx(1.5)
        assert snap["tokens_by_tier"]["deep"] == {"prompt": 10, "completion": 20}

    def test_escalation_rate(self):
        d = MonitoringDashboard()
        d.record_task(self._result(escalated_from="lightning"))
        d.record_task(self._result())
        snap = d.get_dashboard()
        assert snap["escalation_rate"] == pytest.approx(0.5)
        assert d.escalations == 1

    def test_failure_modes(self):
        d = MonitoringDashboard()
        d.record_task(self._result(failed=True, error="omega: boom"))
        d.record_task(self._result(failed=True, error="omega: other"))
        snap = d.get_dashboard()
        assert snap["failure_modes"] == {"omega": 2}

    def test_technique_usage(self):
        d = MonitoringDashboard()
        d.record_task(self._result(techs=["a", "b"]))
        d.record_task(self._result(techs=["a"]))
        snap = d.get_dashboard()
        assert snap["technique_usage"] == {"a": 2, "b": 1}

    def test_empty_dashboard(self):
        snap = MonitoringDashboard().get_dashboard()
        assert snap["tasks_processed"] == 0
        assert snap["critique_pass_rate"] == 0.0
        assert snap["escalation_rate"] == 0.0

    def test_unknown_tier_falls_back_to_lightning(self):
        d = MonitoringDashboard()
        d.record_task(self._result(tier="quantum"))
        snap = d.get_dashboard()
        assert snap["tasks_by_tier"]["lightning"] == 1
