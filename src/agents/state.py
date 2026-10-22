"""Intermediate reasoning state accumulated during an agent's pipeline.

Each agent builds an ``AgentState`` as it progresses through its phases.
The state is:

- emitted as ``progress`` events so the UI can show live completion,
- read by the ``ConfidenceMonitor`` to detect convergence,
- handed to the synthesis step when the user presses FINISH NOW,
- persisted on the final ``AgentResult`` (as a JSON-able dict).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolvedSubProblem:
    """A sub-problem that has been solved, with its reasoning trace."""

    sub_problem: str
    answer: str
    trace: str = ""

    def to_dict(self) -> dict:
        return {
            "sub_problem": self.sub_problem,
            "answer": self.answer,
            "trace": self.trace,
        }


@dataclass
class AgentState:
    """Mutable snapshot of everything an agent has produced so far."""

    task: str = ""
    tier: str = ""
    reframed_task: str | None = None

    # v2.0: compact verified-artifact context (ActiveContext.compact_text()).
    # When set, the FINISH NOW synthesis prompt leads with this instead of the
    # legacy per-sub-problem listing.
    context_text: str = ""

    # decomposition
    sub_problems: list[str] = field(default_factory=list)
    solved: list[SolvedSubProblem] = field(default_factory=list)

    # debate
    debate_challenges: list[str] = field(default_factory=list)
    debate_synthesis: str = ""
    debate_revised_trace: str | None = None
    debate_revised_answer: str | None = None

    # self-consistency (omega)
    consistency_samples: list[str] = field(default_factory=list)
    consensus_answer: str | None = None
    consensus_confidence: float | None = None

    # verification (omega)
    verification_revised_answer: str | None = None
    verification_inconsistencies: list[str] = field(default_factory=list)

    # critique
    critique_history: list = field(default_factory=list)

    # progress bookkeeping
    current_phase: str = ""
    total_steps: int = 0
    completed_steps: int = 0

    # best answer so far
    best_answer: str = ""
    best_trace: str = ""

    techniques: list[str] = field(default_factory=list)

    # control flags
    finish_now: bool = False
    aborted: bool = False

    # synthesis metadata (set when FINISH NOW runs)
    synthesis: dict | None = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def record_answer(self, answer: str, trace: str = "") -> None:
        """Record a candidate final answer, keeping the best one seen."""
        if answer:
            self.best_answer = answer
        if trace:
            self.best_trace = trace

    def mark_step(self) -> None:
        self.completed_steps += 1

    @property
    def completion_pct(self) -> float:
        if not self.total_steps:
            return 0.0
        return min(1.0, self.completed_steps / self.total_steps)

    @property
    def solved_count(self) -> int:
        return len(self.solved)

    def completeness_level(self) -> str:
        """Classify how complete the partial state is (for synthesis)."""
        if self.consensus_answer is not None:
            return "consistency"
        n = len(self.sub_problems)
        if n == 0:
            # single-shot pipeline (lightning): we either have a full
            # answer or we barely started.
            return "mostly_complete" if self.best_answer else "barely_started"
        solved = len(self.solved)
        if solved == 0:
            return "barely_started"
        if solved >= max(1, n - 1):
            return "mostly_complete"
        return "partial"

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "tier": self.tier,
            "reframed_task": self.reframed_task,
            "context_text": self.context_text,
            "sub_problems": list(self.sub_problems),
            "solved": [s.to_dict() for s in self.solved],
            "debate_challenges": list(self.debate_challenges),
            "debate_synthesis": self.debate_synthesis,
            "debate_revised_trace": self.debate_revised_trace,
            "debate_revised_answer": self.debate_revised_answer,
            "consistency_samples": list(self.consistency_samples),
            "consensus_answer": self.consensus_answer,
            "consensus_confidence": self.consensus_confidence,
            "verification_revised_answer": self.verification_revised_answer,
            "verification_inconsistencies": list(self.verification_inconsistencies),
            "critique_history": [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.critique_history
            ],
            "current_phase": self.current_phase,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "completion_pct": round(self.completion_pct, 3),
            "best_answer": self.best_answer,
            "best_trace": self.best_trace,
            "techniques": list(self.techniques),
            "finish_now": self.finish_now,
            "aborted": self.aborted,
            "synthesis": self.synthesis,
        }
