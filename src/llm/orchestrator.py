"""LLM orchestration: config, responses, cost tracking, retries, parallelism.

The orchestrator is transport-agnostic: it accepts any async callable
``transport(messages, config) -> LLMResponse``. Production uses the
llama.cpp OpenAI-compatible client; tests inject a scripted mock.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from config.settings import LLMSettings
from src.llm.context_manager import ContextManager

SYSTEM_PROMPT = (
    "Work step by step, then state the result on its own final line beginning "
    "with 'ANSWER:'. Be concise. When asked to output JSON, output ONLY valid JSON."
)

Transport = Callable[[list[dict], "LLMConfig"], Awaitable["LLMResponse"]]


class LLMError(RuntimeError):
    """Raised when an LLM call ultimately fails."""


class ControlInterrupted(Exception):
    """Raised when a pending STOP / FINISH NOW interrupts an in-flight LLM call.

    ``signal`` is the control signal that fired: ``"stop"`` or ``"finish"``.
    """

    def __init__(self, signal: str):
        self.signal = signal
        super().__init__(signal)


_CONTROL_POLL_INTERVAL = 0.1

# v2.6: the control gate + task id are exposed to the *transport* via context
# variables so a streaming transport can check the gate on EVERY chunk and
# abort a FINISH NOW mid-stream immediately (not just at the next poll or
# phase boundary). They are only set while a gated call is in flight.
CONTROL_GATE: contextvars.ContextVar[Optional[Callable[[str], str]]] = contextvars.ContextVar(
    "solomon_control_gate", default=None
)
CURRENT_TASK_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "solomon_current_task_id", default=None
)


async def _poll_control_signal(gate, task_id: str, poll_interval: float) -> str:
    """Poll ``gate(task_id)`` until it returns a non-continue signal."""
    while True:
        await asyncio.sleep(poll_interval)
        sig = gate(task_id)
        if sig != "continue":
            return sig


@dataclass
class LLMConfig:
    provider: str = "local"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    context_isolation: bool = False  # force a fresh context window
    system: Optional[str] = None


@dataclass
class LLMResponse:
    text: str  # final answer (message content)
    trace: str  # internal reasoning (reasoning_content)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def full_text(self) -> str:
        """Trace + answer, for display and trace storage."""
        if self.trace and self.text:
            return f"[REASONING]\n{self.trace}\n\n[OUTPUT]\n{self.text}"
        return self.trace or self.text


@dataclass
class _TaskUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    calls: int = 0


class CostTracker:
    """Tracks token usage and estimated cost per task and globally."""

    def __init__(self, price_per_1k_prompt: float = 0.0, price_per_1k_completion: float = 0.0):
        self.price_per_1k_prompt = price_per_1k_prompt
        self.price_per_1k_completion = price_per_1k_completion
        self._tasks: dict[str, _TaskUsage] = {}
        self._global = _TaskUsage()

    def record(
        self,
        task_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        cost = (
            prompt_tokens / 1000.0 * self.price_per_1k_prompt
            + completion_tokens / 1000.0 * self.price_per_1k_completion
        )
        task = self._tasks.setdefault(task_id, _TaskUsage())
        task.prompt_tokens += prompt_tokens
        task.completion_tokens += completion_tokens
        task.cost += cost
        task.calls += 1
        self._global.prompt_tokens += prompt_tokens
        self._global.completion_tokens += completion_tokens
        self._global.cost += cost
        self._global.calls += 1

    def get_task_cost(self, task_id: str) -> float:
        return self._tasks.get(task_id, _TaskUsage()).cost

    def get_task_tokens(self, task_id: str) -> tuple[int, int]:
        t = self._tasks.get(task_id, _TaskUsage())
        return t.prompt_tokens, t.completion_tokens

    def get_task_calls(self, task_id: str) -> int:
        return self._tasks.get(task_id, _TaskUsage()).calls

    def snapshot(self) -> dict:
        return {
            "global": {
                "calls": self._global.calls,
                "prompt_tokens": self._global.prompt_tokens,
                "completion_tokens": self._global.completion_tokens,
                "cost": self._global.cost,
            },
            "tasks": {
                tid: {
                    "calls": t.calls,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "cost": t.cost,
                }
                for tid, t in self._tasks.items()
            },
        }


class LLMOrchestrator:
    """Manages all LLM calls with retries, cost tracking, and parallelism."""

    def __init__(
        self,
        settings: LLMSettings,
        transport: Optional[Transport] = None,
        event_bus=None,
        gate=None,
    ):
        self.settings = settings
        if transport is None:
            from src.llm.llama_client import LlamaCppTransport

            transport = LlamaCppTransport(settings)
        self.transport = transport
        self.event_bus = event_bus
        self.cost_tracker = CostTracker(
            price_per_1k_prompt=settings.price_per_1k_prompt,
            price_per_1k_completion=settings.price_per_1k_completion,
        )
        self.context_manager = ContextManager()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        # Optional control gate: a sync callable ``task_id -> "continue"|"stop"|"finish"``.
        # When set, in-flight LLM calls are raced against it so STOP / FINISH NOW
        # interrupt work immediately instead of only at phase boundaries.
        self.gate = gate

    async def _run_with_gate(
        self, coro, task_id: str, enabled: bool = True
    ) -> "LLMResponse":
        """Run ``coro`` while racing it against the control gate.

        When ``enabled`` and a gate is configured, the moment the gate returns a
        non-continue signal for ``task_id`` the in-flight call is cancelled and
        :class:`ControlInterrupted` is raised. Otherwise the coroutine is simply
        awaited (its result or exception propagates normally).

        While the call is in flight the gate + task id are also exposed to the
        transport via context variables, so streaming transports check the gate
        on every chunk and abort immediately.
        """
        if not enabled or self.gate is None or not task_id:
            return await coro
        tokens = [
            (CONTROL_GATE, CONTROL_GATE.set(self.gate)),
            (CURRENT_TASK_ID, CURRENT_TASK_ID.set(task_id)),
        ]
        try:
            return await self._race_with_gate(coro, task_id)
        finally:
            for var, tok in tokens:
                var.reset(tok)

    async def _race_with_gate(self, coro, task_id: str) -> "LLMResponse":
        coro_task = asyncio.ensure_future(coro)
        poll_task = asyncio.ensure_future(
            _poll_control_signal(self.gate, task_id, _CONTROL_POLL_INTERVAL)
        )
        try:
            done, _ = await asyncio.wait(
                {coro_task, poll_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if coro_task in done:
                poll_task.cancel()
                return await coro_task
            sig = poll_task.result()
            coro_task.cancel()
            with contextlib.suppress(BaseException):
                await coro_task
            raise ControlInterrupted(sig)
        finally:
            for t in (poll_task, coro_task):
                if not t.done():
                    t.cancel()

    def build_messages(
        self,
        prompt: str,
        config: LLMConfig,
        system: Optional[str] = None,
    ) -> list[dict]:
        """Build a fresh message list — never reuses another call's history."""
        messages: list[dict] = []
        sys_text = system or config.system or SYSTEM_PROMPT
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        config: LLMConfig,
        task_id: str,
        phase: str,
        collect_trace: bool = True,
        system: Optional[str] = None,
        gate: bool = True,
    ) -> LLMResponse:
        """Generate one LLM response with retry and cost accounting.

        When ``gate`` is True and a control gate is configured, the in-flight
        call is cancelled and :class:`ControlInterrupted` is raised as soon as a
        pending STOP / FINISH NOW signal appears for ``task_id``. Pass
        ``gate=False`` for calls that must run to completion (e.g. the FINISH
        NOW synthesis itself).
        """
        messages = self.build_messages(prompt, config, system=system)
        max_tokens = config.max_tokens or self.settings.default_max_tokens

        async with self._semaphore:
            response = await self._run_with_gate(
                self._call_with_retry(messages, config, max_tokens), task_id, gate
            )

        # Reasoning models can exhaust max_tokens inside their thinking block,
        # leaving an empty answer. Retry once with more headroom.
        if not response.text and response.trace and response.finish_reason == "length":
            response = await self._run_with_gate(
                self._call_with_retry(messages, config, min(max_tokens * 2, 8192)),
                task_id,
                gate,
            )

        # v2.6 P1-4: never report 0/0 for a non-empty response. If the backend
        # did not report usage, estimate locally (chars // 4) so token telemetry
        # and cost accounting stay meaningful.
        if response.completion_tokens <= 0 and response.text:
            response.completion_tokens = max(1, len(response.text) // 4)
        if response.prompt_tokens <= 0:
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            response.prompt_tokens = max(1, prompt_chars // 4)

        self.cost_tracker.record(
            task_id, response.prompt_tokens, response.completion_tokens
        )
        await self._emit_llm_call(messages, prompt, response, task_id, phase, collect_trace)
        if not collect_trace:
            response.trace = ""
        return response

    async def _emit_llm_call(
        self, messages, prompt, response, task_id, phase, collect_trace
    ) -> None:
        """Surface each LLM call's system prompt, prompt, and reasoning trace."""
        if self.event_bus is None:
            return
        from src.monitoring.events import TraceEvent

        system = ""
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
                break
        await self.event_bus.emit(
            TraceEvent(
                task_id=task_id,
                phase=phase,
                event_type="llm_call",
                data={
                    "system": system[:12000],
                    "prompt": prompt[:12000],
                    "trace": response.trace[:16000] if collect_trace else "",
                    "answer": response.text,
                    "finish_reason": response.finish_reason,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                },
            )
        )

    async def _call_with_retry(
        self, messages: list[dict], config: LLMConfig, max_tokens: int
    ) -> LLMResponse:
        call_config = LLMConfig(
            provider=config.provider,
            model=config.model or self.settings.model,
            temperature=config.temperature,
            max_tokens=max_tokens,
            context_isolation=config.context_isolation,
            system=config.system,
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return await self.transport(messages, call_config)
            except LLMError:
                raise
            except Exception as exc:  # network / parse / timeout
                last_error = exc
                if attempt < self.settings.max_retries:
                    await asyncio.sleep(min(2.0 ** attempt, 8.0))
        raise LLMError(f"LLM call failed after {self.settings.max_retries + 1} attempts: {last_error}")

    async def generate_parallel(
        self,
        prompts: list[str],
        config: LLMConfig,
        task_id: str,
        phase: str,
        collect_trace: bool = True,
        gate: bool = True,
    ) -> list[LLMResponse]:
        """Run multiple LLM calls concurrently (debate, consistency sampling)."""
        if not prompts:
            return []
        return await asyncio.gather(
            *(
                self.generate(
                    p, config, task_id, f"{phase}_{i}", collect_trace, gate=gate
                )
                for i, p in enumerate(prompts)
            )
        )

    async def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close is not None:
            await close()
