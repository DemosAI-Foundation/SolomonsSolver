"""Shared fixtures: a scripted mock LLM transport for deterministic unit tests."""
from __future__ import annotations

import pytest

from config.settings import LLMSettings, Settings
from src.llm.orchestrator import LLMOrchestrator, LLMResponse
from src.models.dataclasses import Task, new_task_id


class ScriptedTransport:
    """Transport that routes prompts through a handler function.

    handler(prompt: str) -> str | LLMResponse
    """

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    async def __call__(self, messages, config):
        prompt = messages[-1]["content"]
        self.calls.append({"messages": messages, "config": config, "prompt": prompt})
        result = self.handler(prompt)
        if isinstance(result, LLMResponse):
            return result
        return LLMResponse(
            text=result,
            trace=f"mock reasoning for: {result[:80]}",
            prompt_tokens=10,
            completion_tokens=20,
            latency=0.01,
            finish_reason="stop",
        )


def smart_handler(prompt: str) -> str:
    """Route a prompt to a deterministic mock response based on its content."""
    # --- FINISH NOW synthesis ---
    if "Synthesize the single best final answer" in prompt:
        return "Combining the intermediate results.\nANSWER: SYNTHESIZED ANSWER"

    # --- generic solve ---
    if "Solve the task below" in prompt or "Solve the sub-problem below" in prompt:
        return "Some step-by-step reasoning here.\nANSWER: 42"

    return "ANSWER: default"


@pytest.fixture
def settings():
    s = Settings()
    s.llm.max_concurrency = 16
    s.llm.price_per_1k_prompt = 0.01
    s.llm.price_per_1k_completion = 0.03
    return s


@pytest.fixture
def transport():
    return ScriptedTransport(smart_handler)


@pytest.fixture
def orchestrator(settings, transport):
    return LLMOrchestrator(settings.llm, transport=transport)


@pytest.fixture
def make_task():
    def _make(prompt: str = "What is 2+2? Answer with just the number.") -> Task:
        return Task(id=new_task_id(), original_prompt=prompt)

    return _make
