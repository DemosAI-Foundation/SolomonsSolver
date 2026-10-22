"""Async client for the local llama.cpp OpenAI-compatible server.

The model is a Qwen3-style reasoning model: the assistant message carries
``reasoning_content`` (internal thinking) and ``content`` (final answer).

v2.6: responses are streamed by default so FINISH NOW can be a real global
interrupt. The streaming path checks the control gate (exposed by the
orchestrator via ``CONTROL_GATE`` / ``CURRENT_TASK_ID`` context variables) on
EVERY chunk and aborts immediately with ``ControlInterrupted`` when a
STOP / FINISH NOW signal is pending — no waiting for the poll interval.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import aiohttp

from config.settings import LLMSettings
from src.llm.orchestrator import (
    CONTROL_GATE,
    CURRENT_TASK_ID,
    ControlInterrupted,
    LLMConfig,
    LLMError,
    LLMResponse,
)


def parse_sse_data(raw: bytes) -> Optional[dict]:
    """Parse one ``data: {...}`` SSE frame into a dict (None for keep-alives,
    comments, or the terminal ``[DONE]`` marker)."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    payload = text[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _chunk_gate_signal() -> Optional[str]:
    """Check the per-chunk control gate. Returns the pending signal or None."""
    gate = CONTROL_GATE.get()
    if gate is None:
        return None
    task_id = CURRENT_TASK_ID.get()
    if not task_id:
        return None
    try:
        return gate(task_id)
    except Exception:  # noqa: BLE001
        return None


class LlamaCppTransport:
    """Transport implementing ``async (messages, config) -> LLMResponse``."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.settings.request_timeout)
            )
        return self._session

    async def __call__(
        self, messages: list[dict], config: LLMConfig
    ) -> LLMResponse:
        if self.settings.stream:
            return await self._stream(messages, config)
        return await self._non_stream(messages, config)

    # ------------------------------------------------------------------ #
    # streaming path (per-chunk FINISH NOW / STOP abort)
    # ------------------------------------------------------------------ #
    async def _stream(self, messages: list[dict], config: LLMConfig) -> LLMResponse:
        payload = {
            "model": config.model or self.settings.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
        }
        start = time.monotonic()
        async with self._semaphore:
            content, trace, usage, finish_reason = await self._post_stream_with_retry(
                payload
            )
        latency = time.monotonic() - start
        return LLMResponse(
            text=(content or "").strip(),
            trace=(trace or "").strip(),
            model=config.model or self.settings.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency=latency,
            finish_reason=finish_reason,
        )

    async def _post_stream_with_retry(
        self, payload: dict
    ) -> tuple[str, str, dict, str]:
        """POST a streaming request and accumulate chunks until ``[DONE]``.

        The control gate is checked before processing every chunk; a pending
        STOP / FINISH NOW aborts the stream immediately (``ControlInterrupted``)
        and is NOT retried.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return await self._consume_stream(payload)
            except ControlInterrupted:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, LLMError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    await asyncio.sleep(min(2.0 ** attempt, 8.0))
        raise LLMError(f"llama.cpp unreachable after retries: {last_error}")

    async def _consume_stream(
        self, payload: dict
    ) -> tuple[str, str, dict, str]:
        from src.llm.orchestrator import ControlInterrupted

        content: list[str] = []
        trace: list[str] = []
        usage: dict = {}
        finish_reason = ""
        session = await self._get_session()
        async with session.post(self.settings.chat_url, json=payload) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    raise LLMError(f"HTTP {resp.status} from llama.cpp: {body}")
                buf = b""
                async for raw in resp.content.iter_any():
                    # Per-chunk FINISH NOW / STOP check: abort immediately.
                    sig = _chunk_gate_signal()
                    if sig is not None and sig != "continue":
                        raise ControlInterrupted(sig)
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        data = parse_sse_data(line)
                        if data is None:
                            continue
                        if data.get("usage"):
                            usage.update(data["usage"])
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        message = choice.get("delta", choice.get("message", {})) or {}
                        piece = message.get("content")
                        if piece:
                            content.append(piece)
                        think = message.get("reasoning_content")
                        if think:
                            trace.append(think)
        return "".join(content), "".join(trace), usage, finish_reason

    # ------------------------------------------------------------------ #
    # non-streaming path (SOLOMON_LLM_STREAM=0)
    # ------------------------------------------------------------------ #
    async def _non_stream(self, messages: list[dict], config: LLMConfig) -> LLMResponse:
        payload = {
            "model": config.model or self.settings.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        start = time.monotonic()
        async with self._semaphore:
            data = await self._post_with_retry(payload)
        latency = time.monotonic() - start

        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed LLM response: {exc}") from exc

        return LLMResponse(
            text=(message.get("content") or "").strip(),
            trace=(message.get("reasoning_content") or "").strip(),
            model=data.get("model", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency=latency,
            finish_reason=choice.get("finish_reason", ""),
            raw=data,
        )

    async def _post_with_retry(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(
                    self.settings.chat_url, json=payload
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:300]
                        raise LLMError(f"HTTP {resp.status} from llama.cpp: {body}")
                    return await resp.json()
            except LLMError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    await asyncio.sleep(min(2.0 ** attempt, 8.0))
        raise LLMError(f"llama.cpp unreachable after retries: {last_error}")

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
