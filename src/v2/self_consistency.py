"""Mechanic G — self-consistency / majority voting for non-executable work.

For reasoning sub-problems we sample several independent reasoning paths,
extract each answer, cluster by normalized answer, and return the majority
answer. The cluster fraction becomes a confidence signal.

v2.6 hard rules (fixes the ``FINAL_JSON_END``-as-answer bug):
1. Each sample is parsed as STRUCTURED JSON (marker block -> schema).
2. Only the parsed object's ``answer`` field is used — never raw text.
3. Samples whose answer is empty, marker-only, or unparseable are REJECTED.
4. A rejected sample is retried exactly once (with the error fed back).
5. If it is still invalid, it is discarded (counted, never clustered).
6. Raw marker text (``FINAL_JSON_START`` / ``FINAL_JSON_END``) is NEVER
   clustered: :func:`cluster_answers` defensively filters it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.llm.orchestrator import ControlInterrupted, LLMConfig, LLMOrchestrator
from src.utils.text import extract_answer
from src.v2.contracts import SubProblemResult
from src.v2.structured import FINAL_JSON_END, FINAL_JSON_START, parse_structured

# Marker tokens that must never survive as an answer (hard validation).
_MARKER_TOKENS = {"final_json_start", "final_json_end"}


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    s = str(answer).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".!?,;:")
    return s


def is_rejected_answer(answer: Optional[str]) -> bool:
    """Hard validation: True when ``answer`` must be rejected.

    Rejects empty answers, the raw FINAL_JSON markers, and answers that are
    nothing but marker text (e.g. "FINAL_JSON_START FINAL_JSON_END").
    """
    if answer is None:
        return True
    s = str(answer).strip()
    if not s:
        return True
    norm = s.lower()
    if norm in _MARKER_TOKENS:
        return True
    # Strip every marker occurrence; if nothing of substance remains, reject.
    stripped = norm
    for tok in _MARKER_TOKENS:
        stripped = stripped.replace(tok, " ")
    stripped = re.sub(r"\s+", " ", stripped).strip(" \t.:;,-")
    return not stripped


def cluster_answers(answers: list[str]) -> tuple[str, float, int]:
    """Group answers by normalized form; return ``(representative, fraction, count)``.

    v2.6: marker/empty answers are filtered out before clustering, so raw
    marker text can never be selected as the answer.
    """
    valid = [a for a in answers if not is_rejected_answer(a)]
    if not valid:
        return ("", 0.0, 0)
    groups: dict[str, list[str]] = {}
    for a in valid:
        groups.setdefault(normalize_answer(a), []).append(a)
    best_key, best_group = max(groups.items(), key=lambda kv: len(kv[1]))
    representative = best_group[0]
    fraction = len(best_group) / len(valid)
    return (representative, fraction, len(best_group))


@dataclass
class SelfConsistencyResult:
    answer: str
    confidence: float
    count: int
    total: int
    answers: list[str] = field(default_factory=list)
    # v2.6 bookkeeping for honest reporting.
    rejected: int = 0
    retries: int = 0


async def _sample_once(
    llm: LLMOrchestrator,
    prompt: str,
    config: LLMConfig,
    task_id: str,
    phase: str,
    schema: type,
    extract: Callable[[object], str],
    error: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Run one sample. Returns ``(answer, None)`` or ``(None, error)``."""
    call_prompt = prompt
    if error:
        call_prompt = (
            prompt
            + "\n\nYour previous response was invalid: "
            f"{error}. Respond again with ONLY valid JSON between the markers."
        )
    resp = await llm.generate(
        prompt=call_prompt, config=config, task_id=task_id, phase=phase
    )
    obj, err = parse_structured(resp.text, schema)
    if obj is None:
        return (None, err or "unparseable response")
    ans = str(extract(obj) or "").strip()
    if is_rejected_answer(ans):
        return (None, "empty or marker-only answer (rejected)")
    return (ans, None)


async def self_consistency(
    llm: LLMOrchestrator,
    prompt: str,
    config: LLMConfig,
    task_id: str,
    phase: str,
    n_samples: int,
    schema: type = SubProblemResult,
    extract: Optional[Callable[[object], str]] = None,
    emit=None,
) -> SelfConsistencyResult:
    """Sample ``n_samples`` independent answers and return the majority.

    v2.6: every sample is parsed as structured JSON; the ``answer`` field of
    the parsed object is the only thing that may enter the cluster. Invalid
    samples (unparseable, empty, or marker-only) are retried once and then
    discarded. The returned ``total`` counts only valid samples, and
    ``rejected`` / ``retries`` record how many were dropped.
    """
    if extract is None:
        extract = (
            lambda obj: getattr(obj, "answer", "") or ""  # noqa: E731
        )

    n = max(1, int(n_samples))
    answers: list[str] = []
    rejected = 0
    retries = 0

    # First round: one call per sample (parallel where possible).
    first_round: list[tuple[Optional[str], Optional[str]]] = []
    try:
        responses = await llm.generate_parallel(
            [prompt] * n, config, task_id, f"{phase}_sc"
        )
    except ControlInterrupted:
        raise
    for resp in responses:
        obj, err = parse_structured(resp.text, schema)
        if obj is None:
            first_round.append((None, err or "unparseable response"))
        else:
            ans = str(extract(obj) or "").strip()
            if is_rejected_answer(ans):
                first_round.append((None, "empty or marker-only answer (rejected)"))
            else:
                first_round.append((ans, None))

    for i, (ans, err) in enumerate(first_round):
        if ans is not None:
            answers.append(ans)
            if emit is not None:
                await emit(f"{phase}_sc", "sc_sample", index=i, answer=ans[:200])
            continue
        # Retry the invalid sample exactly once, feeding the error back.
        retries += 1
        if emit is not None:
            await emit(
                f"{phase}_sc", "sc_sample_retry", index=i, error=str(err)[:200]
            )
        retried, err2 = await _sample_once(
            llm, prompt, config, task_id, f"{phase}_sc", schema, extract, error=err
        )
        if retried is not None:
            answers.append(retried)
            if emit is not None:
                await emit(
                    f"{phase}_sc", "sc_sample", index=i, answer=retried[:200], retry=True
                )
        else:
            rejected += 1
            if emit is not None:
                await emit(
                    f"{phase}_sc",
                    "sc_sample_dropped",
                    index=i,
                    error=str(err2 or err)[:200],
                )

    representative, fraction, count = cluster_answers(answers)
    if emit is not None:
        await emit(
            f"{phase}_sc",
            "sc_cluster",
            answer=representative[:200],
            confidence=round(fraction, 3),
            count=count,
            total=len(answers),
            rejected=rejected,
        )
    return SelfConsistencyResult(
        answer=representative,
        confidence=fraction,
        count=count,
        total=len(answers),
        answers=answers,
        rejected=rejected,
        retries=retries,
    )


async def merge_samples(
    llm: LLMOrchestrator,
    prompt: str,
    samples: list[str],
    config: LLMConfig,
    task_id: str,
    phase: str,
) -> Optional[str]:
    """v2.6 low-agreement recovery: one merge call over the valid samples.

    Combines the disagreeing samples into the best single answer. Returns the
    merged answer text, or None when the merge call itself fails (the caller
    then keeps the raw majority answer and stays partial/unverified).
    """
    if not samples:
        return None
    from src.v2.prompts import build_sc_merge_prompt

    merge_prompt = build_sc_merge_prompt(prompt, samples)
    cfg = LLMConfig(
        temperature=0.0,
        max_tokens=config.max_tokens,
        context_isolation=config.context_isolation,
        system=config.system,
    )
    resp = await llm.generate(
        prompt=merge_prompt, config=cfg, task_id=task_id, phase=f"{phase}_sc_merge"
    )
    obj, err = parse_structured(resp.text, SubProblemResult)
    if obj is not None:
        ans = str(getattr(obj, "answer", "") or "").strip()
        if not is_rejected_answer(ans):
            return ans
    # Fall back to a plain text merge: last non-empty, non-marker line.
    text = (resp.text or "").strip()
    if is_rejected_answer(text):
        return None
    return extract_answer(text) or None
