"""v2.6 Reasoning Policy — allocate reasoning tokens deliberately.

Small local models benefit from reasoning, but reasoning is a limited resource
that can truncate final answers, corrupt JSON, and drift off-task. This module
implements the v2.6 rule:

    Reason deeply for decisions.
    Use constrained output for formatting.
    Verify after every step.
    Never let reasoning eat the final-answer budget.

Public surface:
- :class:`ReasoningLevel` — OFF / LOW / MEDIUM / HIGH.
- :class:`ReasoningPolicy` — the per-phase reasoning allocation.
- :func:`reasoning_policy_for` — the controller that picks a level + token caps.
- :func:`sub_problem_reasoning_level` — the adaptive Phase 3 level.
- :func:`reasoning_directive` — the prompt line for a level.
- :func:`build_reasoning_prompt` / :func:`build_extraction_prompt` — the two
  reason-then-extract call prompts.
- :func:`reason_then_extract` — the two-call reason-then-extract helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from src.llm.orchestrator import LLMConfig, LLMOrchestrator
from src.v2.structured import build_schema_instruction, parse_structured

T = TypeVar("T", bound=BaseModel)


class ReasoningLevel(str, Enum):
    """How much reasoning a phase is allowed to spend (v2.6)."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Fraction of the mode's reasoning budget a level may spend on its reasoning
# call. OFF spends nothing; HIGH spends the full reasoning budget.
_LEVEL_REASON_FRACTION = {
    ReasoningLevel.OFF: 0.0,
    ReasoningLevel.LOW: 0.25,
    ReasoningLevel.MEDIUM: 0.6,
    ReasoningLevel.HIGH: 1.0,
}

# Base reasoning level per phase (Section 2 of the v2.6 plan). Phase 3 is
# adaptive and is overridden per sub-problem by the pipeline.
_PHASE_BASE_LEVEL = {
    "phase0": ReasoningLevel.LOW,
    "phase1": ReasoningLevel.MEDIUM,
    "phase2": ReasoningLevel.MEDIUM,
    "phase3": ReasoningLevel.MEDIUM,
    "phase4": ReasoningLevel.MEDIUM,
    "phase5": ReasoningLevel.MEDIUM,
    "phase6": ReasoningLevel.MEDIUM,
    "phase7": ReasoningLevel.MEDIUM,
    "phase8": ReasoningLevel.LOW,
    "phase8_repair": ReasoningLevel.MEDIUM,
    "finish_now": ReasoningLevel.LOW,
    "lightning_intake": ReasoningLevel.OFF,
    "lightning_answer": ReasoningLevel.OFF,
    "lightning_repair": ReasoningLevel.OFF,
}

# Phases that make high-stakes DECISIONS and therefore use the two-call
# reason-then-extract pattern (Section 4 of the v2.6 plan). These are the
# single-structured-call decision phases: planning, decomposition, and
# critique. Phase 6 selection is computed from evaluations by the pipeline and
# Phase 7/8 repair produces the (protected) final answer, so they apply the
# reasoning level as a directive instead of the two-call pattern.
_REASON_THEN_EXTRACT_PHASES = {
    "phase1",
    "phase2",
    "phase4",
}

# Reason-then-extract prompt styles (Section 4 of the v2.6 plan).
_REASONING_SYSTEM = (
    "Produce concise reasoning notes only. Do not produce the final answer and "
    "do not wrap anything in markers."
)
_EXTRACTION_SYSTEM = (
    "Produce only the required structured output. Do not re-derive the "
    "reasoning and do not discuss the output format."
)


@dataclass
class ReasoningPolicy:
    """Per-phase reasoning allocation (v2.6).

    ``max_reasoning_tokens`` bounds the reasoning call; ``max_completion_tokens``
    bounds the structured output call. ``use_reason_then_extract`` requests the
    two-call pattern (reason briefly, then emit strict structured output).
    """

    phase: str
    level: ReasoningLevel = ReasoningLevel.LOW
    max_reasoning_tokens: int = 0
    max_completion_tokens: int = 0
    use_reason_then_extract: bool = False

    @property
    def enabled(self) -> bool:
        return self.level is not ReasoningLevel.OFF


def _canonical(phase: str) -> str:
    return (phase or "").strip().lower()


def reasoning_policy_for(
    phase: str,
    mode: str = "standard",
    reasoning_budget: int = 2000,
    completion_budget: int = 4000,
    level: Optional[ReasoningLevel] = None,
    use_reason_then_extract: Optional[bool] = None,
) -> ReasoningPolicy:
    """The v2.6 controller: pick a reasoning level + token caps for a phase.

    ``level`` overrides the base level (used for the adaptive Phase 3 level and
    for truncation recovery, which *reduces* the level). The reasoning cap is a
    fraction of the mode's ``reasoning_budget``; the completion cap is the
    mode's ``completion_budget``.
    """
    p = _canonical(phase)
    if level is None:
        level = _PHASE_BASE_LEVEL.get(p, ReasoningLevel.LOW)
    fraction = _LEVEL_REASON_FRACTION.get(level, 0.0)
    max_reasoning = int(max(0, reasoning_budget) * fraction)
    max_completion = int(max(0, completion_budget)) if level is not ReasoningLevel.OFF else 0
    if use_reason_then_extract is None:
        use_reason_then_extract = p in _REASON_THEN_EXTRACT_PHASES
    return ReasoningPolicy(
        phase=p,
        level=level,
        max_reasoning_tokens=max_reasoning,
        max_completion_tokens=max_completion,
        use_reason_then_extract=bool(use_reason_then_extract),
    )


def sub_problem_reasoning_level(sp) -> ReasoningLevel:
    """v2.6 adaptive Phase 3 level: reasoning is allocated by sub-problem type.

    Code, debugging, and high-risk logic get HIGH; simple factual / conceptual
    work gets LOW/MEDIUM (Section 3 of the v2.6 plan).
    """
    route = getattr(sp, "type", None)
    risk = getattr(sp, "risk", None)
    route_value = route.value if hasattr(route, "value") else str(route or "").lower()
    risk_value = risk.value if hasattr(risk, "value") else str(risk or "").lower()
    if route_value in ("executable", "hybrid"):
        return ReasoningLevel.HIGH
    if risk_value == "high":
        return ReasoningLevel.HIGH
    if risk_value == "medium":
        return ReasoningLevel.MEDIUM
    return ReasoningLevel.LOW


def reasoning_directive(level: ReasoningLevel) -> str:
    """The prompt line for a reasoning level (v2.6 prompt style).

    Returns an empty string for OFF so the caller can append it unconditionally.
    """
    if level is ReasoningLevel.OFF:
        return ""
    if level is ReasoningLevel.LOW:
        return (
            "REASONING: low — return only the required output; do not explain."
        )
    if level is ReasoningLevel.MEDIUM:
        return (
            "REASONING: medium — think briefly, then return only the required "
            "output."
        )
    return (
        "REASONING: high — analyze the constraints, identify the key decision, "
        "consider failure modes, then produce the required output."
    )


def build_reasoning_prompt(task_context: str, level: ReasoningLevel) -> str:
    """Call 1 of reason-then-extract: brief reasoning notes (no final answer).

    ``task_context`` is the full phase prompt (task + compact context) so the
    reasoning call sees exactly what the structured call will see.
    """
    depth = {
        ReasoningLevel.LOW: "3 to 4 short bullets",
        ReasoningLevel.MEDIUM: "5 to 7 short bullets",
        ReasoningLevel.HIGH: "6 to 9 short bullets",
    }.get(level, "5 to 7 short bullets")
    return (
        "REASONING NOTES (reason-then-extract, step 1 of 2).\n"
        f"Produce {depth} of concise reasoning notes for the task below. Focus "
        "on the decision that must be made, the constraints, and the most "
        "likely failure modes. Do NOT produce the final structured output yet "
        "and do NOT wrap anything in markers.\n\n"
        f"TASK:\n{task_context}\n"
    )


def build_extraction_prompt(
    task_context: str,
    reasoning_notes: str,
    schema: Type[BaseModel],
    purpose: str = "",
) -> str:
    """Call 2 of reason-then-extract: strict structured output derived from the
    reasoning notes. The full phase prompt is embedded so the structured call
    sees the same task + context as the reasoning call."""
    return (
        "STRUCTURED EXTRACTION (reason-then-extract, step 2 of 2).\n"
        "Using ONLY the reasoning notes below, produce the required structured "
        "output. Do not re-derive the reasoning and do not discuss the format.\n\n"
        f"TASK:\n{task_context}\n\n"
        f"REASONING NOTES:\n{reasoning_notes or '(none)'}\n\n"
        + build_schema_instruction(schema, purpose=purpose)
    )


async def reason_then_extract(
    llm: LLMOrchestrator,
    task_context: str,
    schema: Type[T],
    policy: ReasoningPolicy,
    *,
    task_id: str = "",
    phase: str = "reason",
    config: Optional[LLMConfig] = None,
    emit: Optional[Callable] = None,
    gate: bool = True,
    record_usage: Optional[Callable] = None,
    system: Optional[str] = None,
) -> tuple[Optional[T], Optional[str]]:
    """Two-call reason-then-extract for high-stakes decisions (v2.6).

    Call 1 produces brief reasoning notes (free text, no final answer); call 2
    produces the strict structured output derived from those notes. Returns
    ``(instance, None)`` on success or ``(None, error)`` on failure. A control
    interruption (FINISH NOW / STOP) propagates :class:`ControlInterrupted`.

    ``record_usage`` (if given) is called with each :class:`LLMResponse` so the
    caller can feed the budget governor. ``system`` (the phase system prompt) is
    applied to BOTH calls so the reasoning call stays inside the phase's
    adaptive system prompt; the reasoning instruction lives in the user prompt.
    """
    base_temp = config.temperature if config is not None else 0.2
    base_system = config.system if config is not None else None
    call_system = system or base_system

    # Call 1 — reasoning notes.
    rprompt = build_reasoning_prompt(task_context, policy.level)
    rcfg = LLMConfig(
        temperature=base_temp,
        max_tokens=max(1, policy.max_reasoning_tokens),
        context_isolation=True,
        system=call_system or _REASONING_SYSTEM,
    )
    rresp = await llm.generate(
        prompt=rprompt, config=rcfg, task_id=task_id, phase=f"{phase}_reason", gate=gate
    )
    if record_usage is not None:
        record_usage(rresp)
    notes = (rresp.text or "").strip()
    if emit is not None:
        await emit(f"{phase}_reason", "reasoning_notes", notes=notes[:200])

    # Call 2 — structured extraction from the notes.
    eprompt = build_extraction_prompt(task_context, notes, schema)
    ecfg = LLMConfig(
        temperature=0.0,
        max_tokens=max(1, policy.max_completion_tokens),
        context_isolation=True,
        system=call_system or _EXTRACTION_SYSTEM,
    )
    eresp = await llm.generate(
        prompt=eprompt, config=ecfg, task_id=task_id, phase=phase, gate=gate
    )
    if record_usage is not None:
        record_usage(eresp)
    obj, err = parse_structured(eresp.text, schema)
    if obj is not None:
        return obj, None

    # One structured retry, feeding the validation error back.
    retry_prompt = (
        eprompt
        + f"\n\nYour previous response was invalid: {err}. Respond again with "
        "ONLY valid JSON between the markers."
    )
    eresp2 = await llm.generate(
        prompt=retry_prompt, config=ecfg, task_id=task_id, phase=f"{phase}_retry", gate=gate
    )
    if record_usage is not None:
        record_usage(eresp2)
    obj2, err2 = parse_structured(eresp2.text, schema)
    if obj2 is not None:
        return obj2, None
    return None, err2 or err
