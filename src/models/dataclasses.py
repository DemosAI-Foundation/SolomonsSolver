"""Core data models shared by the V2 pipeline and infrastructure."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

Tier = str

TIER_LIGHTNING = "lightning"
TIER_DEEP = "deep"
TIER_OMEGA = "omega"
TIERS = (TIER_LIGHTNING, TIER_DEEP, TIER_OMEGA)

# v2.0 is a distinct mode (not a tier): the verifier-first pipeline.
MODE_V2 = "v2"
MODES = ("auto",) + TIERS + (MODE_V2,)

# v2.6: every user-facing tier runs the v2 verifier-first pipeline.
#   lightning -> lightning (the short L0-L3 pipeline: one intake, one
#                 one-shot answer, one fast rule gate, labeled delivery)
#   deep      -> standard  (medium budget)
#   omega     -> deep      (slowest, most budget)
TIER_TO_V2_MODE = {
    TIER_LIGHTNING: "lightning",
    TIER_DEEP: "standard",
    TIER_OMEGA: "deep",
}

TASK_TYPES = ("code", "math", "factual", "reasoning", "unknown")


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Task:
    id: str
    original_prompt: str
    reframed_prompt: Optional[str] = None
    complexity_score: float = 0.0
    task_type: str = "unknown"  # code | math | factual | reasoning | unknown
    tier: str = TIER_LIGHTNING  # lightning | deep | omega
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_prompt": self.original_prompt,
            "reframed_prompt": self.reframed_prompt,
            "complexity_score": self.complexity_score,
            "task_type": self.task_type,
            "tier": self.tier,
            "created_at": self.created_at,
        }


@dataclass
class AgentResult:
    task_id: str
    final_answer: str
    reasoning_trace: str
    tier: Tier
    critique_history: list = field(default_factory=list)
    iterations: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    techniques_used: list[str] = field(default_factory=list)
    tokens_prompt: int = 0
    tokens_completion: int = 0
    escalated_from: Optional[Tier] = None
    failed: bool = False
    # FINISH NOW / STOP support
    early_completion: bool = False
    aborted: bool = False
    confidence: float = 0.0
    completion_pct: float = 0.0
    state: dict = field(default_factory=dict)
    synthesis: Optional[dict] = None
    error: Optional[str] = None
    # v2.6 Lightning: delivery metadata (mode, verified flag, UI banner label,
    # truncation flag). Surfaced to the UI as a banner, never in the answer body.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "final_answer": self.final_answer,
            "reasoning_trace": self.reasoning_trace,
            "tier": self.tier,
            "critique_history": [
                c.to_dict() if hasattr(c, "to_dict") else c for c in self.critique_history
            ],
            "iterations": self.iterations,
            "total_cost": self.total_cost,
            "total_latency": self.total_latency,
            "techniques_used": self.techniques_used,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "escalated_from": self.escalated_from,
            "failed": self.failed,
            "early_completion": self.early_completion,
            "aborted": self.aborted,
            "confidence": self.confidence,
            "completion_pct": self.completion_pct,
            "synthesis": self.synthesis,
            "metadata": self.metadata,
        }
