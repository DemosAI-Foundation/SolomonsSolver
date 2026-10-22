"""Prompt templates for the V2 pipeline (FINISH NOW synthesis)."""
from __future__ import annotations

SYNTHESIS_TEMPLATE = """The multi-step reasoning for the task below was STOPPED before it finished. Below are the intermediate results gathered so far. Synthesize the single best final answer from them.

Task:
{task}

Intermediate results gathered so far:
{intermediate}

Completeness of the partial work: {level_desc}

MANDATORY HONESTY RULE: Be explicit about what you do and do not know. If the intermediate results are insufficient to answer confidently, say so plainly and state exactly which information is missing. Do NOT fabricate missing facts, invent numbers, or pretend a gap does not exist. If you can only answer partially, give the best partial answer and clearly mark what remains unresolved.

Produce a clear final answer. State the result on its own final line beginning with "ANSWER:"."""


AUX_TEMPLATES = {
    "synthesis": SYNTHESIS_TEMPLATE,
}
