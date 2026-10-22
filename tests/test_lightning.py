"""Acceptance tests for the v2.6 Lightning pipeline fixes (L0-L3).

Covers the Lightning Fix Plan acceptance criteria:
  1. The full L1 answer is stored before delivery (source-of-truth artifact).
  2. The delivered answer is not truncated.
  3. The Lightning label is metadata, not part of the answer body.
  4. The L2 fast gate runs before delivery.
  5. Missing sections trigger exactly one targeted repair.
  6. Forbidden overclaim patterns are detected (contract-level base set).
  7. Token usage is no longer 0/0 (local estimate fallback).
  8. Truncation is recorded when stop_reason == length.
  9. Finish Now returns a partial answer immediately.
  10. Lightning still avoids Deep-mode phases.
"""
from __future__ import annotations

import asyncio
import json

from config.settings import Settings
from src.agents.control import SessionRegistry
from src.agents.synthesis import SynthesisStep
from src.llm.orchestrator import LLMOrchestrator, LLMResponse
from src.models.dataclasses import Task, new_task_id
from src.v2 import (
    FinishManager,
    LIGHTNING_BASE_FORBIDDEN_PATTERNS,
    LIGHTNING_LABEL,
    RetrievalAdapter,
    SandboxAdapter,
    V2Pipeline,
)
from src.v2.prompts import build_lightning_answer_prompt, build_lightning_intake_prompt
from src.v2.contracts import LightningIntake


def _wrap(obj) -> str:
    return f"FINAL_JSON_START\n{json.dumps(obj)}\nFINAL_JSON_END"


_LONG_OK = (
    "## Answer\n"
    "The answer is 42. This is a sufficiently long, self-contained answer that "
    "passes the minimum length check and contains no forbidden patterns."
)


class LightningTransport:
    """Deterministic transport for the Lightning stages, configurable per test."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.calls: list[str] = []
        self.blocked = False
        self.block_on = cfg.get("block_on")
        self.block_delay = cfg.get("block_delay", 5)

    async def __call__(self, messages, config):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if self.block_on and not self.blocked and self.block_on in prompt:
            self.blocked = True
            await asyncio.sleep(self.block_delay)

        if "LIGHTNING INTAKE stage" in prompt:
            text = _wrap(self.cfg.get("intake", {
                "route": "non_executable",
                "answer_type": "factual",
                "required_sections": ["Answer"],
                "forbidden_patterns": [],
            }))
            finish = "stop"
        elif "LIGHTNING REASONING stage" in prompt:
            text = self.cfg.get("reasoning", "- one\n- two\n- three")
            finish = "stop"
        elif "LIGHTNING ANSWER stage" in prompt:
            text = self.cfg.get("answer", _LONG_OK)
            finish = self.cfg.get("answer_finish", "stop")
        elif "LIGHTNING REPAIR stage" in prompt:
            text = self.cfg.get("repaired", _LONG_OK + " (repaired)")
            finish = "stop"
        else:
            text = "ANSWER: default"
            finish = "stop"

        return LLMResponse(
            text=text, trace="",
            prompt_tokens=self.cfg.get("prompt_tokens", 10),
            completion_tokens=self.cfg.get("completion_tokens", 20),
            latency=0.01, finish_reason=finish,
        )


def _pipeline(transport) -> V2Pipeline:
    s = Settings()
    s.llm.max_concurrency = 16
    llm = LLMOrchestrator(s.llm, transport=transport)
    sessions = SessionRegistry()
    return V2Pipeline(
        llm, None, settings=s, sessions=sessions, synthesis=SynthesisStep(llm),
        retrieval=RetrievalAdapter(), sandbox=SandboxAdapter(),
        mode="lightning",
    )


def _task(prompt: str = "What is 2+2?") -> Task:
    return Task(id=new_task_id(), original_prompt=prompt)


async def _run(cfg: dict):
    transport = LightningTransport(cfg)
    pipeline = _pipeline(transport)
    task = _task()
    result = await pipeline.run(task)
    return result, pipeline, transport


class TestLightningDelivery:
    async def test_full_l1_answer_stored_as_artifact(self):
        """1. The full L1 answer is stored before delivery (source of truth)."""
        result, pipeline, transport = await _run({"answer": _LONG_OK})
        ctx = pipeline.active_context
        assert ctx.artifacts.get("lightning_answer") == _LONG_OK
        # Delivery reads from the stored artifact.
        assert result.final_answer == _LONG_OK

    async def test_delivered_answer_not_truncated(self):
        """2. A complete answer is delivered intact (not cut off)."""
        result, _pipeline, _transport = await _run({"answer": _LONG_OK})
        assert result.final_answer == _LONG_OK
        assert not V2Pipeline._looks_truncated(result.final_answer)
        assert result.metadata.get("truncated") is False

    async def test_label_is_metadata_not_body(self):
        """3. The Lightning label is delivery metadata, never in the answer body."""
        result, _pipeline, _transport = await _run({"answer": _LONG_OK})
        assert LIGHTNING_LABEL not in result.final_answer
        assert result.metadata.get("mode") == "lightning"
        assert result.metadata.get("verified") is False
        assert result.metadata.get("label") == LIGHTNING_LABEL


class TestLightningGate:
    async def test_missing_section_triggers_one_repair(self):
        """4+5. The gate runs before delivery; a missing section triggers exactly
        one targeted repair call."""
        cfg = {
            "intake": {
                "route": "non_executable", "answer_type": "factual",
                "required_sections": ["Answer", "Summary"],
                "forbidden_patterns": [],
            },
            "answer": "## Answer\n" + _LONG_OK[len("## Answer\n"):],
            "repaired": "## Answer\n" + _LONG_OK[len("## Answer\n"):] + "\n## Summary\nA short summary.",
        }
        result, _pipeline, transport = await _run(cfg)
        repair_calls = [c for c in transport.calls if "LIGHTNING REPAIR stage" in c]
        assert len(repair_calls) == 1, "exactly one targeted repair expected"
        assert "lightning_fast_gate_failed" in result.techniques_used
        assert "lightning_repair" in result.techniques_used
        # The repaired (gate-passing) answer is delivered.
        assert "Summary" in result.final_answer

    async def test_forbidden_overclaim_pattern_detected(self):
        """6. Contract-level forbidden overclaim patterns are detected."""
        cfg = {
            "intake": {
                "route": "non_executable", "answer_type": "factual",
                "required_sections": ["Answer"], "forbidden_patterns": [],
            },
            "answer": "## Answer\nThis system creates real life in a digital "
                      "environment and is fully operational and safe.",
            "repaired": _LONG_OK,
        }
        result, _pipeline, transport = await _run(cfg)
        assert "creates real life" in LIGHTNING_BASE_FORBIDDEN_PATTERNS
        # The base pattern is checked even though the intake emitted none.
        assert "lightning_fast_gate_failed" in result.techniques_used
        repair_calls = [c for c in transport.calls if "LIGHTNING REPAIR stage" in c]
        assert len(repair_calls) == 1
        # The repaired answer (clean) is delivered.
        assert "creates real life" not in result.final_answer

    async def test_gate_rejects_truncated_answer(self):
        """2 (gate side). An answer ending mid-sentence is flagged as truncated."""
        intake = LightningIntake(required_sections=["Answer"])
        issues = V2Pipeline._lightning_gate("The system and", intake)
        assert any("truncation" in i for i in issues)

    async def test_actionability_required_for_how_to(self):
        """P1-6. A how-to task must cover its minimal actionability requirements."""
        intake = LightningIntake(
            required_sections=["Answer"],
            actionability_requirements=["minimal_path", "success_check", "limitations"],
        )
        # Covers "minimal path" + "success check" but not "limitations".
        partial = (
            "## Answer\nUse the minimal path: install the deps and run the "
            "build. The success check is that the tests pass. "
            "This is long enough to clear the minimum length bar."
        )
        issues = V2Pipeline._lightning_gate(partial, intake)
        assert any("actionability" in i and "limitations" in i for i in issues)
        # Adding the missing requirement clears the issue.
        full = partial + " Limitations: it only works on POSIX systems."
        issues2 = V2Pipeline._lightning_gate(full, intake)
        assert not any("actionability" in i for i in issues2)


class TestLightningTelemetry:
    async def test_token_usage_not_zero(self):
        """7. Token usage is no longer 0/0 (local estimate fallback)."""
        result, _pipeline, _transport = await _run({
            "answer": _LONG_OK,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        })
        assert result.tokens_prompt > 0, "prompt tokens must be estimated"
        assert result.tokens_completion > 0, "completion tokens must be estimated"

    async def test_truncation_recorded_on_length(self):
        """8. Truncation is recorded when stop_reason == length."""
        result, pipeline, _transport = await _run({
            "answer": _LONG_OK,
            "answer_finish": "length",
        })
        assert result.metadata.get("truncated") is True
        assert pipeline.active_context.truncations >= 1


class TestLightningFinishNow:
    async def test_finish_now_returns_partial_answer_immediately(self):
        """9. FINISH NOW during the repair returns the stored L1 partial answer
        immediately (no deep phases, no extra LLM call)."""
        cfg = {
            "intake": {
                "route": "non_executable", "answer_type": "factual",
                "required_sections": ["Answer", "Summary"],
                "forbidden_patterns": [],
            },
            # L1 answer is missing "Summary" -> gate fails -> repair triggered.
            "answer": "## Answer\n" + _LONG_OK[len("## Answer\n"):],
            "block_on": "LIGHTNING REPAIR stage",
            "block_delay": 5,
        }
        fm = FinishManager()
        transport = LightningTransport(cfg)
        s = Settings()
        s.llm.max_concurrency = 16
        llm = LLMOrchestrator(s.llm, transport=transport, gate=fm.signal)
        pipeline = V2Pipeline(
            llm, None, settings=s, sessions=SessionRegistry(),
            synthesis=SynthesisStep(llm), retrieval=RetrievalAdapter(),
            sandbox=SandboxAdapter(), mode="lightning", finish_manager=fm,
        )
        task = _task()
        run_task = asyncio.create_task(pipeline.run(task))
        for _ in range(300):
            await asyncio.sleep(0.01)
            if transport.blocked:
                break
        assert transport.blocked, "pipeline never reached the repair call"
        fm.request_finish(task.id)
        result = await asyncio.wait_for(run_task, timeout=10)
        assert result.early_completion is True
        # The partial L1 answer (stored artifact) is returned, labeled incomplete.
        assert "## Answer" in result.final_answer
        assert "Summary" not in result.final_answer
        assert result.synthesis["incomplete_parts"]
        assert result.metadata.get("mode") == "lightning"


class TestLightningIsolation:
    async def test_lightning_avoids_deep_phases(self):
        """10. Lightning never runs the Deep-mode (Phase 0-8) prompts."""
        result, _pipeline, transport = await _run({"answer": _LONG_OK})
        deep_markers = [
            "INTAKE & ROUTING stage", "PLANNING stage", "DECOMPOSITION stage",
            "WRITING CODE", "SOLVING one sub-problem", "VERIFICATION stage",
            "producing DRAFT", "INDEPENDENT VERIFIER", "FINAL SYNTHESIS stage",
            "COVERAGE VALIDATION stage",
        ]
        seen = "\n".join(transport.calls)
        for marker in deep_markers:
            assert marker not in seen, f"Deep-mode prompt leaked into Lightning: {marker}"
        # Only Lightning stages were invoked.
        assert any("LIGHTNING INTAKE stage" in c for c in transport.calls)
        assert any("LIGHTNING ANSWER stage" in c for c in transport.calls)
        for tech in result.techniques_used:
            assert not tech.startswith("phase"), f"Deep-mode technique in Lightning: {tech}"


class TestLightningReasoning:
    async def test_short_reasoning_used_when_requested(self):
        """P2-7. A logic-heavy task gets a bounded reasoning step before the
        one-shot answer."""
        cfg = {
            "intake": {
                "route": "non_executable", "answer_type": "analysis",
                "required_sections": ["Answer"], "forbidden_patterns": [],
                "needs_reasoning": True,
            },
            "reasoning": "- step one\n- step two\n- step three",
            "answer": _LONG_OK,
        }
        result, _pipeline, transport = await _run(cfg)
        reasoning_calls = [c for c in transport.calls if "LIGHTNING REASONING stage" in c]
        assert len(reasoning_calls) == 1
        assert "lightning_short_reasoning" in result.techniques_used

    async def test_no_reasoning_for_simple_task(self):
        """P2-7. A simple factual task gets NO reasoning step."""
        result, _pipeline, transport = await _run({"answer": _LONG_OK})
        reasoning_calls = [c for c in transport.calls if "LIGHTNING REASONING stage" in c]
        assert reasoning_calls == []
        assert "lightning_short_reasoning" not in result.techniques_used
