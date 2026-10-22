"""Repro: FINISH NOW (real FinishManager + created task, like server submit_task)
must interrupt the pipeline at every phase and in every mode."""
from __future__ import annotations

import asyncio
import time

from config.settings import Settings
from src.agents.control import SessionRegistry
from src.agents.synthesis import SynthesisStep
from src.llm.orchestrator import LLMOrchestrator, LLMResponse
from src.models.dataclasses import Task, new_task_id
from src.v2 import (
    FinishManager,
    RetrievalAdapter,
    SandboxAdapter,
    V2Pipeline,
)
from tests.test_v2 import v2_handler


def _task(prompt="What is 2+2?"):
    return Task(id=new_task_id(), original_prompt=prompt)


class SlowTransport:
    """Sleeps 5s on the first call matching a marker (a long in-flight call)."""

    def __init__(self, marker):
        self.marker = marker
        self.calls = []
        self.blocked = False

    async def __call__(self, messages, config):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if not self.blocked and self.marker in prompt:
            self.blocked = True
            await asyncio.sleep(5)
        return LLMResponse(
            text=v2_handler(prompt), trace="",
            prompt_tokens=10, completion_tokens=20,
            latency=0.01, finish_reason="stop",
        )


async def _scenario(mode, marker):
    fm = FinishManager()
    transport = SlowTransport(marker)
    s = Settings()
    s.llm.max_concurrency = 16
    llm = LLMOrchestrator(s.llm, transport=transport, gate=fm.signal)
    sessions = SessionRegistry()
    synthesis = SynthesisStep(llm)
    pipeline = V2Pipeline(
        llm, None, settings=s, sessions=sessions, synthesis=synthesis,
        retrieval=RetrievalAdapter(), sandbox=SandboxAdapter(),
        mode=mode, finish_manager=fm,
    )
    task = _task()
    t0 = time.monotonic()

    # Start the pipeline in its own task (like the server's submit_task).
    run_task = asyncio.create_task(pipeline.run(task))

    # Wait until the pipeline is blocked on the marker call.
    for _ in range(300):
        await asyncio.sleep(0.01)
        if transport.blocked:
            break
    assert transport.blocked, f"pipeline never reached {marker!r}"

    # Press FINISH NOW (the real server mechanism).
    pressed = fm.request_finish(task.id)
    result = await asyncio.wait_for(run_task, timeout=10)
    elapsed = time.monotonic() - t0
    print(f"\n{mode.upper()} (marker={marker!r}): pressed={pressed} "
          f"early_completion={result.early_completion} aborted={result.aborted} "
          f"failed={result.failed} elapsed={elapsed:.2f}s n_calls={len(transport.calls)}")
    print(f"  answer={(result.final_answer or '')[:70]!r}")
    return result, elapsed


class TestFinishNowEverywhere:
    async def test_standard(self):
        result, elapsed = await _scenario("standard", "producing DRAFT")
        assert result.early_completion is True, "FINISH NOW did not stop standard"
        assert elapsed < 2, f"standard took {elapsed:.2f}s (should stop fast)"

    async def test_lightning(self):
        result, elapsed = await _scenario("lightning", "LIGHTNING")
        assert result.early_completion is True, "FINISH NOW did not stop lightning"
        assert elapsed < 2, f"lightning took {elapsed:.2f}s (should stop fast)"

    async def test_deep(self):
        result, elapsed = await _scenario("deep", "producing DRAFT")
        assert result.early_completion is True, "FINISH NOW did not stop deep"
        assert elapsed < 2, f"deep took {elapsed:.2f}s (should stop fast)"
