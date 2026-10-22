"""Mechanic F — order-swapped pairwise judging.

Position bias is a real failure mode of LLM judges, so every comparison is run
twice with the two answers swapped. A definitive winner only emerges when both
orderings agree; any disagreement (including a tie in either ordering) is
reported as a tie. This makes the judge conservative and bias-resistant.
"""
from __future__ import annotations

from typing import Optional

from src.llm.orchestrator import LLMConfig, LLMOrchestrator
from src.v2.contracts import PairwiseJudgment
from src.v2.prompts import build_phase6_pairwise_prompt, phase_system_prompt
from src.v2.structured import parse_structured


async def _judge_once(
    llm: LLMOrchestrator, task: str, a: str, b: str, task_id: str, phase: str, order: str
) -> PairwiseJudgment:
    prompt = build_phase6_pairwise_prompt(task, a, b)
    resp = await llm.generate(
        prompt=prompt,
        config=LLMConfig(temperature=0.0, context_isolation=True, system=phase_system_prompt("phase6")),
        task_id=task_id,
        phase=f"{phase}_{order}",
    )
    obj, _err = parse_structured(resp.text, PairwiseJudgment)
    if obj is None:
        return PairwiseJudgment(winner="tie", reason="unparseable judgment")
    return obj


async def pairwise_judge(
    llm: LLMOrchestrator,
    task: str,
    answer_a: str,
    answer_b: str,
    task_id: str,
    phase: str,
    emit=None,
) -> PairwiseJudgment:
    """Order-swapped comparison. ``winner`` is in the ORIGINAL terms:
    'A' = answer_a, 'B' = answer_b, 'tie' = no robust preference."""
    j1 = await _judge_once(llm, task, answer_a, answer_b, task_id, phase, "ab")
    j2 = await _judge_once(llm, task, answer_b, answer_a, task_id, phase, "ba")

    w1 = j1.winner  # already in original terms (order A,B)
    if j2.winner == "A":
        w2 = "B"  # swapped: position A was the original answer_b
    elif j2.winner == "B":
        w2 = "A"
    else:
        w2 = "tie"

    if w1 in ("A", "B") and w1 == w2:
        winner = w1
        reason = f"both orderings agree ({w1})"
    else:
        winner = "tie"
        reason = f"orderings disagree (ab={j1.winner}, ba={j2.winner}) -> tie"

    if emit is not None:
        await emit(
            phase,
            "pairwise",
            winner=winner,
            ab=j1.winner,
            ba=j2.winner,
            reason=reason,
        )
    return PairwiseJudgment(winner=winner, reason=reason)


async def pairwise_tournament(
    llm: LLMOrchestrator,
    task: str,
    answers: list[str],
    task_id: str,
    phase: str,
    emit=None,
) -> int:
    """Round-robin order-swapped tournament. Returns the winning index, or -1
    when there is no robust single winner (all ties / tied win counts)."""
    n = len(answers)
    if n == 0:
        return -1
    if n == 1:
        return 0
    wins = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            jdg = await pairwise_judge(
                llm, task, answers[i], answers[j], task_id, phase, emit=emit
            )
            if jdg.winner == "A":
                wins[i] += 1
            elif jdg.winner == "B":
                wins[j] += 1
    best = max(wins)
    if best == 0:
        return -1
    leaders = [i for i in range(n) if wins[i] == best]
    return leaders[0] if len(leaders) == 1 else -1
