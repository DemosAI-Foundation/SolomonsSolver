"""Tests for the FINISH NOW / STOP feature surface.

Covers:
- AgentState (intermediate state + completeness classification)
- TaskSession / SessionRegistry (control signals + registry)
- SynthesisStep (isolated FINISH NOW synthesis call)
- ConfidenceMonitor (live confidence scoring + eligibility)
- In-flight interruption (STOP / FINISH NOW cancels a running LLM call)
- SolomonApp control methods (the logic the HTTP endpoints delegate to)
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from src.agents.control import SessionRegistry, TaskSession
from src.agents.state import AgentState, SolvedSubProblem
from src.agents.synthesis import SynthesisStep
from src.app import SolomonApp
from src.llm.orchestrator import (
    ControlInterrupted,
    LLMConfig,
    LLMOrchestrator,
    LLMResponse,
)
from src.monitoring.confidence import ConfidenceMonitor
from src.monitoring.events import EventBus, TraceEvent
from src.prompts.builder import build_synthesis_prompt
from tests.conftest import ScriptedTransport, smart_handler


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_app(settings, tmp_path):
    """Build a full SolomonApp with a temp DB and a mock LLM transport."""
    settings.db_path = tmp_path / "test.db"
    settings.log_dir = tmp_path / "logs"
    app = SolomonApp(settings)
    app.llm.transport = ScriptedTransport(smart_handler)
    return app


# --------------------------------------------------------------------------- #
# AgentState
# --------------------------------------------------------------------------- #
class TestAgentState:
    def test_record_answer_keeps_last_nonempty(self):
        s = AgentState(task="t")
        s.record_answer("first", "trace1")
        s.record_answer("", "trace2")  # empty answer -> keep 'first'
        assert s.best_answer == "first"
        assert s.best_trace == "trace2"

    def test_completion_pct(self):
        s = AgentState()
        assert s.completion_pct == 0.0
        s.total_steps = 4
        s.mark_step()
        s.mark_step()
        assert s.completion_pct == 0.5

    def test_completeness_level_lightning_flow(self):
        s = AgentState()
        assert s.completeness_level() == "barely_started"
        s.record_answer("42")
        assert s.completeness_level() == "mostly_complete"

    def test_completeness_level_partial(self):
        s = AgentState()
        s.sub_problems = ["a", "b", "c"]
        s.solved = [SolvedSubProblem("a", "1")]
        assert s.completeness_level() == "partial"
        s.solved.append(SolvedSubProblem("b", "2"))
        assert s.completeness_level() == "mostly_complete"

    def test_completeness_level_consistency(self):
        s = AgentState()
        s.consensus_answer = "42"
        assert s.completeness_level() == "consistency"

    def test_to_dict_json_serializable(self):
        s = AgentState(task="t", tier="lightning")
        s.sub_problems = ["a"]
        s.solved = [SolvedSubProblem("a", "1", "trace")]
        s.record_answer("1")
        d = s.to_dict()
        json.dumps(d)  # must not raise
        assert d["best_answer"] == "1"
        assert d["solved"][0]["answer"] == "1"
        assert d["completion_pct"] == 0.0


# --------------------------------------------------------------------------- #
# TaskSession / SessionRegistry
# --------------------------------------------------------------------------- #
class TestControl:
    def test_signal_continue(self):
        assert TaskSession("t").signal() == "continue"

    def test_signal_stop_beats_finish(self):
        s = TaskSession("t")
        s.request_finish_now()
        assert s.signal() == "finish"
        s.request_stop()
        assert s.signal() == "stop"

    def test_registry_get_or_create(self):
        r = SessionRegistry()
        assert len(r) == 0
        s1 = r.get_or_create("a")
        assert r.get_or_create("a") is s1
        assert len(r) == 1
        r.get_or_create("b")
        assert len(r) == 2
        assert r.get("a") is s1
        assert r.get("zzz") is None
        r.remove("a")
        assert r.get("a") is None
        assert [x.task_id for x in r.active()] == ["b"]

    def test_session_to_dict(self):
        s = TaskSession("t")
        s.state = AgentState(task="x")
        d = s.to_dict()
        assert d["task_id"] == "t"
        assert d["state"]["task"] == "x"
        assert d["stop_requested"] is False
        assert d["finish_now_requested"] is False
        assert d["finished"] is False


# --------------------------------------------------------------------------- #
# SynthesisStep
# --------------------------------------------------------------------------- #
class TestSynthesis:
    async def test_synthesize_returns_triple(self, settings):
        orch = LLMOrchestrator(settings.llm, transport=ScriptedTransport(smart_handler))
        step = SynthesisStep(orch)
        state = AgentState(task="What is 2+2?")
        state.record_answer("42", "some trace")
        events = []

        async def emit(phase, etype, **data):
            events.append((phase, etype, data))

        answer, trace, meta = await step.synthesize(state, "t1", emit=emit)
        assert answer == "SYNTHESIZED ANSWER"
        assert trace
        assert meta["level"] == "mostly_complete"
        assert meta["solved_sub_problems"] == 0
        phases = [(p, e) for p, e, _ in events]
        assert ("synthesis", "phase_start") in phases
        assert ("synthesis", "phase_end") in phases

    async def test_synthesize_uses_isolation(self, settings):
        orch = LLMOrchestrator(settings.llm, transport=ScriptedTransport(smart_handler))
        step = SynthesisStep(orch)
        state = AgentState(task="t")
        state.record_answer("x")
        await step.synthesize(state, "t1")
        call = orch.transport.calls[0]
        assert call["config"].context_isolation is True
        assert "Synthesize the single best final answer" in call["prompt"]

    def test_build_synthesis_prompt_contains_state(self):
        state = AgentState(task="What is 2+2?")
        state.record_answer("42")
        prompt = build_synthesis_prompt(state, "mostly_complete")
        assert "What is 2+2?" in prompt
        assert "42" in prompt


# --------------------------------------------------------------------------- #
# ConfidenceMonitor
# --------------------------------------------------------------------------- #
class TestConfidenceMonitor:
    async def test_score_and_eligibility(self):
        bus = EventBus()
        seen = []
        bus.subscribe(
            lambda ev: seen.append(ev) if ev.event_type == "confidence" else None
        )
        mon = ConfidenceMonitor(event_bus=bus, threshold=0.7)

        await mon.observe(
            TraceEvent(
                task_id="t",
                phase="solve",
                event_type="progress",
                data={"best_answer": "42", "steps_done": 1, "steps_total": 3},
            )
        )
        snap = mon.snapshot("t")
        assert snap["confidence"] == 0.4
        assert snap["level"] == "medium"
        assert snap["finish_now_eligible"] is False  # 0.4 < 0.7

        await mon.observe(
            TraceEvent(
                task_id="t",
                phase="critique",
                event_type="critique",
                data={"verdict": "pass"},
            )
        )
        snap = mon.snapshot("t")
        assert snap["confidence"] == 0.85
        assert snap["level"] == "high"
        assert snap["finish_now_eligible"] is True
        # a confidence event was emitted back onto the bus
        assert any(e.event_type == "confidence" for e in seen)

    async def test_convergence(self):
        mon = ConfidenceMonitor(threshold=0.7)
        await mon.observe(
            TraceEvent(task_id="t", phase="a", event_type="progress", data={"best_answer": "X"})
        )
        await mon.observe(
            TraceEvent(task_id="t", phase="b", event_type="progress", data={"best_answer": "X"})
        )
        snap = mon.snapshot("t")
        assert snap["converged"] is True
        assert snap["confidence"] >= 0.75

    async def test_forget(self):
        mon = ConfidenceMonitor()
        await mon.observe(
            TraceEvent(task_id="t", phase="a", event_type="progress", data={"best_answer": "X"})
        )
        assert mon.snapshot("t") is not None
        mon.forget("t")
        assert mon.snapshot("t") is None

    async def test_ignores_own_events(self):
        mon = ConfidenceMonitor()
        await mon.observe(
            TraceEvent(
                task_id="t",
                phase="confidence",
                event_type="confidence",
                data={"confidence": 0.9},
            )
        )
        assert mon.snapshot("t") is None  # no task state created


# --------------------------------------------------------------------------- #
# In-flight interruption (STOP / FINISH NOW cancels a running LLM call)
# --------------------------------------------------------------------------- #
class TestInFlightInterruption:
    async def test_generate_interrupted_mid_call(self, settings):
        async def slow_transport(messages, config):
            await asyncio.sleep(5)
            return LLMResponse(text="done", trace="")

        orch = LLMOrchestrator(settings.llm, transport=slow_transport)
        orch.gate = lambda tid: "stop"
        t0 = time.monotonic()
        with pytest.raises(ControlInterrupted) as exc:
            await orch.generate("p", LLMConfig(), "t1", "solve")
        assert exc.value.signal == "stop"
        assert time.monotonic() - t0 < 2  # interrupted well before the 5s call

    async def test_generate_gate_false_runs_to_completion(self, settings):
        async def slow_transport(messages, config):
            await asyncio.sleep(0.2)
            return LLMResponse(text="done", trace="")

        orch = LLMOrchestrator(settings.llm, transport=slow_transport)
        orch.gate = lambda tid: "stop"
        resp = await orch.generate("p", LLMConfig(), "t1", "solve", gate=False)
        assert resp.text == "done"


# --------------------------------------------------------------------------- #
# SolomonApp control methods (logic behind the HTTP endpoints)
# --------------------------------------------------------------------------- #
class TestAppControl:
    async def test_control_methods(self, settings, tmp_path):
        app = _make_app(settings, tmp_path)
        try:
            # unknown task -> no session
            assert app.request_finish_now("nope") is False
            assert app.request_stop("nope") is False
            assert app.task_state("nope") is None

            # simulate a running task
            sess = app.sessions.get_or_create("t1")
            sess.state = AgentState(task="x")
            assert app.request_finish_now("t1") is True
            assert app.request_stop("t1") is True
            st = app.task_state("t1")
            assert st is not None
            assert st["task_id"] == "t1"
            assert st["state"]["task"] == "x"

            # after completion the session is removed -> signals rejected
            app.sessions.remove("t1")
            assert app.request_finish_now("t1") is False
            assert app.request_stop("t1") is False
            assert app.task_state("t1") is None
        finally:
            await app.close()
