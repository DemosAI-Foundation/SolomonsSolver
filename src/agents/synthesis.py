"""FINISH NOW synthesis step.

When the user presses FINISH NOW, the agent stops its normal pipeline and runs
this single, isolated LLM call to synthesize a final answer from whatever
intermediate results exist. The call:

- uses a fresh context (``context_isolation=True``) so it never sees the
  original conversation,
- is handed only the partial :class:`AgentState`,
- is instructed to be honest about gaps and never fabricate.
"""
from __future__ import annotations

from src.agents.state import AgentState
from src.llm.orchestrator import LLMConfig
from src.prompts.builder import build_synthesis_prompt
from src.utils.text import extract_answer


class SynthesisStep:
    """Runs one isolated synthesis LLM call over a partial AgentState."""

    def __init__(self, llm):
        self.llm = llm

    async def synthesize(
        self, state: AgentState, task_id: str, emit=None
    ) -> tuple[str, str, dict]:
        """Return ``(answer, trace, meta)`` for the partial state."""
        level = state.completeness_level()
        prompt = build_synthesis_prompt(state, level)
        if emit:
            await emit("synthesis", "phase_start", level=level)
        # gate=False: this call IS the FINISH NOW synthesis, so it must run to
        # completion and must not itself be interrupted by the pending signal.
        response = await self.llm.generate(
            prompt=prompt,
            config=LLMConfig(temperature=0.2, max_tokens=2048, context_isolation=True),
            task_id=task_id,
            phase="synthesis",
            gate=False,
        )
        answer = extract_answer(response.text)
        trace = response.full_text
        meta = {
            "level": level,
            "solved_sub_problems": len(state.solved),
            "total_sub_problems": len(state.sub_problems),
            "consensus_answer": state.consensus_answer,
            "consensus_confidence": state.consensus_confidence,
            "techniques": list(state.techniques),
        }
        if emit:
            await emit("synthesis", "phase_end", answer=answer, level=level)
        return answer, trace, meta
