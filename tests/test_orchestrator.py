"""Tests for the LLM orchestrator: retries, cost tracking, parallelism,
and the reasoning-model empty-answer retry."""
import pytest

from src.llm.orchestrator import (
    LLMConfig,
    LLMError,
    LLMOrchestrator,
    LLMResponse,
    SYSTEM_PROMPT,
)
from tests.conftest import ScriptedTransport


class TestBuildMessages:
    def test_system_plus_user(self, settings):
        orch = LLMOrchestrator(settings.llm, transport=ScriptedTransport(lambda p: "x"))
        msgs = orch.build_messages("hello", LLMConfig())
        assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_custom_system(self, settings):
        orch = LLMOrchestrator(settings.llm, transport=ScriptedTransport(lambda p: "x"))
        msgs = orch.build_messages("hello", LLMConfig(), system="custom sys")
        assert msgs[0]["content"] == "custom sys"

    def test_config_system_used_when_no_explicit(self, settings):
        orch = LLMOrchestrator(settings.llm, transport=ScriptedTransport(lambda p: "x"))
        cfg = LLMConfig(system="from config")
        msgs = orch.build_messages("hello", cfg)
        assert msgs[0]["content"] == "from config"


class TestGenerate:
    async def test_basic_generate(self, orchestrator):
        # "Solve the task below" maps to the mock's "ANSWER: 42"
        resp = await orchestrator.generate(
            "Solve the task below: what is 2+2?", LLMConfig(), task_id="t1", phase="solve"
        )
        assert "42" in resp.text
        assert resp.trace  # mock trace present

    async def test_cost_tracked_per_task(self, orchestrator, settings):
        await orchestrator.generate("p1", LLMConfig(), task_id="t1", phase="a")
        await orchestrator.generate("p2", LLMConfig(), task_id="t1", phase="b")
        await orchestrator.generate("p3", LLMConfig(), task_id="t2", phase="c")
        # mock returns 10 prompt / 20 completion tokens per call
        p, c = orchestrator.cost_tracker.get_task_tokens("t1")
        assert (p, c) == (20, 40)
        p2, c2 = orchestrator.cost_tracker.get_task_tokens("t2")
        assert (p2, c2) == (10, 20)
        # prices: 0.01/1k prompt, 0.03/1k completion
        expected = 20 / 1000 * 0.01 + 40 / 1000 * 0.03
        assert orchestrator.cost_tracker.get_task_cost("t1") == pytest.approx(expected)

    async def test_collect_trace_false_clears_trace(self, orchestrator):
        resp = await orchestrator.generate(
            "p", LLMConfig(), task_id="t1", phase="a", collect_trace=False
        )
        assert resp.trace == ""
        assert resp.text  # answer still present

    async def test_retry_on_transient_error(self, settings):
        calls = {"n": 0}

        async def flaky(messages, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("boom")
            return LLMResponse(text="ok", trace="t", finish_reason="stop")

        settings.llm.max_retries = 2
        orch = LLMOrchestrator(settings.llm, transport=flaky)
        resp = await orch.generate("p", LLMConfig(), task_id="t", phase="a")
        assert resp.text == "ok"
        assert calls["n"] == 2

    async def test_raises_after_exhausting_retries(self, settings):
        async def always_fail(messages, config):
            raise ConnectionError("down")

        settings.llm.max_retries = 1
        orch = LLMOrchestrator(settings.llm, transport=always_fail)
        with pytest.raises(LLMError):
            await orch.generate("p", LLMConfig(), task_id="t", phase="a")

    async def test_llm_error_not_retried(self, settings):
        calls = {"n": 0}

        async def fatal(messages, config):
            calls["n"] += 1
            raise LLMError("fatal")

        settings.llm.max_retries = 3
        orch = LLMOrchestrator(settings.llm, transport=fatal)
        with pytest.raises(LLMError):
            await orch.generate("p", LLMConfig(), task_id="t", phase="a")
        assert calls["n"] == 1

    async def test_empty_answer_length_retry_doubles_max_tokens(self, settings):
        seen_max_tokens = []

        async def transport(messages, config):
            seen_max_tokens.append(config.max_tokens)
            if len(seen_max_tokens) == 1:
                # reasoning model burned all tokens thinking, no answer
                return LLMResponse(
                    text="", trace="long thinking...", finish_reason="length"
                )
            return LLMResponse(text="ANSWER: done", trace="thinking", finish_reason="stop")

        orch = LLMOrchestrator(settings.llm, transport=transport)
        resp = await orch.generate(
            "p", LLMConfig(max_tokens=1000), task_id="t", phase="a"
        )
        assert resp.text == "ANSWER: done"
        assert seen_max_tokens == [1000, 2000]

    async def test_empty_answer_no_retry_when_not_length(self, settings):
        seen = []

        async def transport(messages, config):
            seen.append(1)
            return LLMResponse(text="", trace="thinking", finish_reason="stop")

        orch = LLMOrchestrator(settings.llm, transport=transport)
        resp = await orch.generate("p", LLMConfig(max_tokens=100), task_id="t", phase="a")
        assert resp.text == ""
        assert len(seen) == 1  # no retry

    async def test_generate_parallel(self, orchestrator):
        resps = await orchestrator.generate_parallel(
            ["a", "b", "c"], LLMConfig(), task_id="t", phase="par"
        )
        assert len(resps) == 3
        assert all(r.text for r in resps)

    async def test_generate_parallel_empty(self, orchestrator):
        assert await orchestrator.generate_parallel([], LLMConfig(), "t", "p") == []

    async def test_full_text_format(self):
        r = LLMResponse(text="the answer", trace="the thinking")
        assert "[REASONING]" in r.full_text and "[OUTPUT]" in r.full_text
        assert LLMResponse(text="", trace="only trace").full_text == "only trace"
        assert LLMResponse(text="only text", trace="").full_text == "only text"


class TestCostTracker:
    def test_snapshot(self):
        from src.llm.orchestrator import CostTracker

        ct = CostTracker(price_per_1k_prompt=1.0, price_per_1k_completion=2.0)
        ct.record("t1", 1000, 1000)  # $1 + $2 = $3
        ct.record("t2", 500, 0)  # $0.5
        snap = ct.snapshot()
        assert snap["global"]["calls"] == 2
        assert snap["global"]["cost"] == pytest.approx(3.5)
        assert snap["tasks"]["t1"]["cost"] == pytest.approx(3.0)
        assert snap["tasks"]["t2"]["cost"] == pytest.approx(0.5)
