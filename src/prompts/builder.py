"""Prompt builders for the V2 pipeline (FINISH NOW synthesis)."""
from __future__ import annotations

from src.prompts.templates import AUX_TEMPLATES


def _fmt(template: str, **kwargs) -> str:
    return template.format(**kwargs)


_LEVEL_DESC = {
    "consistency": "High — independent sampling reached a consensus.",
    "mostly_complete": "Mostly complete — nearly all sub-steps are solved.",
    "partial": "Partial — some sub-steps are solved, others are missing.",
    "barely_started": "Barely started — little or no sub-step work is complete.",
}


def _format_intermediate(state) -> str:
    """Render an AgentState (duck-typed) into a readable intermediate summary."""
    lines: list[str] = []
    # v2.0: lead with the compact verified-artifact context when present.
    if getattr(state, "context_text", ""):
        lines.append("VERIFIED ARTIFACTS (v2.0 active context):")
        lines.append(state.context_text)
    if getattr(state, "reframed_task", None):
        lines.append(f"Reframed task: {state.reframed_task}")

    sub_problems = getattr(state, "sub_problems", []) or []
    solved = getattr(state, "solved", []) or []
    if sub_problems:
        lines.append(f"Sub-problems ({len(solved)}/{len(sub_problems)} solved):")
        for i, sp in enumerate(sub_problems):
            if i < len(solved):
                s = solved[i]
                lines.append(f"  {i+1}. [SOLVED] {sp}")
                lines.append(f"     answer: {s.answer}")
            else:
                lines.append(f"  {i+1}. [UNSOLVED] {sp}")

    if getattr(state, "debate_synthesis", ""):
        lines.append(f"Debate synthesis: {state.debate_synthesis}")
    if getattr(state, "debate_revised_answer", None):
        lines.append(f"Debate-revised answer: {state.debate_revised_answer}")

    samples = getattr(state, "consistency_samples", []) or []
    if samples:
        lines.append(
            "Self-consistency samples: "
            + ", ".join(f'"{s}"' for s in samples)
        )
    if getattr(state, "consensus_answer", None) is not None:
        conf = getattr(state, "consensus_confidence", None)
        lines.append(
            f"Consensus answer: {state.consensus_answer}"
            + (f" (agreement {conf:.2f})" if conf is not None else "")
        )

    if getattr(state, "verification_revised_answer", None):
        lines.append(
            "Verification-revised answer: "
            f"{state.verification_revised_answer}"
        )

    if getattr(state, "best_answer", ""):
        lines.append(f"Best answer so far: {state.best_answer}")

    return "\n".join(lines) or "(no intermediate results captured)"


def build_synthesis_prompt(state, level: str | None = None) -> str:
    level = level or state.completeness_level()
    return _fmt(
        AUX_TEMPLATES["synthesis"],
        task=getattr(state, "task", "") or "(task not recorded)",
        intermediate=_format_intermediate(state),
        level_desc=_LEVEL_DESC.get(level, level),
    )
