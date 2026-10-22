"""Solomon v2.0 — typed data contracts.

The v2.0 architecture is verifier-first and schema-driven. Every phase emits
structured JSON (between FINAL_JSON_START / FINAL_JSON_END markers) that is
validated against these Pydantic models before it is allowed to flow into the
next phase. Raw reasoning traces never travel between phases — only these
compact, verified artifacts do.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Route(str, Enum):
    """How the task should be worked."""

    EXECUTABLE = "executable"
    NON_EXECUTABLE = "non_executable"
    HYBRID = "hybrid"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PairwiseMode(str, Enum):
    TIE_BREAK = "tie_break"
    TOP3 = "top3"


class CritiqueVerdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    FAIL = "fail"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnswerType(str, Enum):
    """What kind of final answer the task calls for (v2.2 answer contract)."""

    OPEN_ENDED_DESIGN = "open_ended_design"
    CODE = "code"
    FACTUAL = "factual"
    ANALYSIS = "analysis"
    HYBRID = "hybrid"


class LengthPolicy(str, Enum):
    """How long the final answer should be (v2.2 answer contract)."""

    CONCISE = "concise"
    STRUCTURED = "structured"
    COMPREHENSIVE = "comprehensive"


class ConfidenceLevel(str, Enum):
    """Coarse confidence label for the emergency FINISH NOW output (v2.2)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationStatus(str, Enum):
    """v2.4: how a piece of work was verified. Work is only 'done' when this is
    ``pass`` (or honestly labelled ``partial`` / ``unverified``)."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"


class CritiqueSeverity(str, Enum):
    """v2.4: severity of a structured critique issue."""

    CRITICAL = "critical"
    MINOR = "minor"


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class V2Model(BaseModel):
    """Base model: permissive of extra keys (the model may emit extras), but
    strict about the fields we actually rely on."""

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------- #
# Facts ledger (Mechanic H)
# --------------------------------------------------------------------------- #
class FactsLedgerEntry(V2Model):
    fact: str
    source: str = "ungrounded"
    grounded: bool = False
    confidence: float = Field(default=0.0)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# --------------------------------------------------------------------------- #
# Budget (Mechanics A/B)
# --------------------------------------------------------------------------- #
class BudgetProfile(V2Model):
    """Per-difficulty / per-mode budget gate. Drives how many branches, drafts,
    samples and repair rounds the pipeline is allowed to spend, plus the v2.2
    budget-governor reserves that protect the final answer and its repair."""

    difficulty: Difficulty = Difficulty.EASY
    mode: str = "standard"
    decomposition_branches: int = 1
    n_drafts: int = 2
    n_samples: int = 3
    n_repair_rounds: int = 1
    pairwise_mode: PairwiseMode = PairwiseMode.TIE_BREAK
    phase_reserves: dict[str, int] = Field(default_factory=dict)
    # v2.2 budget governor: protect the final answer and its repair.
    final_reserve_tokens: int = 6000
    final_repair_reserve_tokens: int = 2000
    intermediate_soft_cap: float = 0.7
    # v2.6 budget governor: separate pools for reasoning and structured
    # completion. Reasoning is intermediate work and can never consume the
    # protected final-answer reserve (enforced by the intermediate soft cap).
    reasoning_budget_tokens: int = 2000
    completion_budget_tokens: int = 4000

    def reserve_for(self, phase: str) -> int:
        """Answer-token reserve for a phase, with a sane fallback."""
        return int(self.phase_reserves.get(phase, 0) or 0)


# --------------------------------------------------------------------------- #
# Task profile (Phase 0 output)
# --------------------------------------------------------------------------- #
class TaskProfile(V2Model):
    route: Route = Route.NON_EXECUTABLE
    difficulty: Difficulty = Difficulty.EASY
    summary: str = ""
    key_requirements: list[str] = Field(default_factory=list)
    retrieval_available: bool = False
    facts: list[FactsLedgerEntry] = Field(default_factory=list)
    budget: BudgetProfile = Field(default_factory=BudgetProfile)
    answer_contract: Optional["AnswerContract"] = None  # v2.2

    def grounded_facts(self) -> list[FactsLedgerEntry]:
        return [f for f in self.facts if f.grounded]

    def ungrounded_facts(self) -> list[FactsLedgerEntry]:
        return [f for f in self.facts if not f.grounded]


# --------------------------------------------------------------------------- #
# Plan (Phase 1 output)
# --------------------------------------------------------------------------- #
class Plan(V2Model):
    approach: str = ""
    strategy: str = ""
    key_requirements: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    verification_focus: list[str] = Field(default_factory=list)
    # v2.5 Phase 1: acceptance criteria, the riskiest sub-problem, and the
    # planner's proposed budget reserves (merged into the run budget downstream).
    acceptance_criteria: list[str] = Field(default_factory=list)
    riskiest_subproblem: str = ""
    budget: Optional[BudgetProfile] = None


# --------------------------------------------------------------------------- #
# Decomposition (Phase 2 output)
# --------------------------------------------------------------------------- #
class SubProblemCandidate(V2Model):
    id: str
    description: str
    type: Route = Route.NON_EXECUTABLE
    depends_on: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW
    verification_hint: str = ""


class DecompositionBranch(V2Model):
    id: str
    rationale: str = ""
    sub_problems: list[SubProblemCandidate] = Field(default_factory=list)


class BranchSet(V2Model):
    """Wrapper for Phase 2 output: one or more decomposition branches."""

    branches: list[DecompositionBranch] = Field(default_factory=list)
    integration_requirement: str = ""  # v2.2: how sub-problems combine into the final answer


# --------------------------------------------------------------------------- #
# Execution (Phase 3)
# --------------------------------------------------------------------------- #
class ExecutionResult(V2Model):
    sub_problem_id: str = ""
    code: str = ""
    stdout: str = ""
    output: str = ""
    tests_passed: bool = False
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    verified: bool = False
    errors: list[str] = Field(default_factory=list)
    sandbox_available: bool = True


class VerificationReport(V2Model):
    """v2.4: the verification schema. A sub-problem or artifact is only 'done'
    when this reports ``pass`` (or is honestly labelled ``partial`` /
    ``unverified`` with a reason). Raw reasoning is never the evidence — only
    execution logs, test results, retrieved facts, or explicit checks are.

    v2.6 honest-verification rules (computed, never model-claimed):
    - a ``pass`` with NO evidence is downgraded to ``partial`` — a pass must
      always carry an evidence list;
    - ``unverified`` / ``fail`` / ``partial`` always carry a ``reason``.
    """

    status: VerificationStatus = VerificationStatus.UNVERIFIED
    evidence: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    reason: str = ""  # required when status == unverified

    @model_validator(mode="after")
    def _enforce_honesty(self) -> "VerificationReport":
        if self.status == VerificationStatus.PASS and not self.evidence:
            # A pass must be backed by evidence; without it, it is only partial.
            self.status = VerificationStatus.PARTIAL
            self.reason = "pass claimed without evidence (downgraded to partial)"
            if "no verification evidence" not in self.missing:
                self.missing.append("no verification evidence")
        if self.status in (
            VerificationStatus.UNVERIFIED,
            VerificationStatus.FAIL,
            VerificationStatus.PARTIAL,
        ) and not self.reason:
            self.reason = {
                VerificationStatus.UNVERIFIED: "not yet verified",
                VerificationStatus.FAIL: "verification failed",
                VerificationStatus.PARTIAL: "only partially verified",
            }[self.status]
        return self

    @property
    def is_pass(self) -> bool:
        return self.status == VerificationStatus.PASS


class SubProblemResult(V2Model):
    sub_problem_id: str
    answer: str = ""
    reasoning: str = ""  # compact, verified — NOT a raw trace
    execution: Optional[ExecutionResult] = None
    verified: bool = False
    verification_method: str = "none"
    confidence: float = Field(default=0.0)
    depends_on: list[str] = Field(default_factory=list)
    # v2.4: structured verification + honest bookkeeping.
    verification: Optional["VerificationReport"] = None
    # v2.6: the raw self-consistency agreement (cluster fraction) that the
    # pipeline computed for this result — Phase 4 sees it explicitly.
    sc_agreement: Optional[float] = None
    assumptions: list[str] = Field(default_factory=list)
    rejected_approaches: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def _verified_requires_pass_report(self) -> "SubProblemResult":
        """v2.6: ``verified`` is computed by the pipeline, never trusted from
        the model. A result is only ``verified`` when its verification report
        is a real ``pass`` (with evidence); a model-emitted ``verified: true``
        without a passing report is blocked."""
        if self.verified and not (
            self.verification is not None and self.verification.is_pass
        ):
            self.verified = False
        return self


# --------------------------------------------------------------------------- #
# Critique (Phase 4)
# --------------------------------------------------------------------------- #
class CritiqueIssue(V2Model):
    """v2.4: one structured critique issue. Every issue carries a concrete
    ``repair_action`` so it can drive a targeted repair (never a raw,
    unstructured blob of findings)."""

    issue_id: str = ""
    severity: CritiqueSeverity = CritiqueSeverity.MINOR
    description: str = ""
    evidence: str = ""
    repair_action: str = ""


class Critique(V2Model):
    target: str = ""
    verdict: CritiqueVerdict = CritiqueVerdict.REVISE
    falsifiable_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    # v2.2 structured self-refine: a REVISE must name concrete repair targets.
    broken_steps: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    scope_issues: list[str] = Field(default_factory=list)
    factual_issues: list[str] = Field(default_factory=list)
    logical_issues: list[str] = Field(default_factory=list)
    trace_fidelity_issues: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    # v2.4 structured critique (debate synthesis): severity-ranked issues and a
    # concrete repair plan. These are what Phase 5 repairs against — never raw
    # debate text.
    critical_issues: list[CritiqueIssue] = Field(default_factory=list)
    minor_issues: list[CritiqueIssue] = Field(default_factory=list)
    repair_plan: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def actionable_targets(self) -> list[str]:
        """Concrete repair targets (v2.2 + v2.4). A REVISE verdict is only
        actionable if it names at least one of: missing sections, broken steps,
        a suggested fix, a repair-plan step, or a structured issue's
        ``repair_action``. Generic 'revise' with none of these is not accepted."""
        out: list[str] = []
        out.extend(self.missing_sections or [])
        out.extend(self.broken_steps or [])
        out.extend(self.suggested_fixes or [])
        out.extend(self.repair_plan or [])
        for issue in list(self.critical_issues or []) + list(self.minor_issues or []):
            if issue.repair_action:
                out.append(issue.repair_action)
        return out

    def structured_issues(self) -> list[CritiqueIssue]:
        """v2.4: all structured issues, critical first (severity-ranked)."""
        return list(self.critical_issues or []) + list(self.minor_issues or [])


class DebateSynthesis(V2Model):
    """v2.4: the consolidated output of the debate-synthesis step. Raw debate
    findings are deduplicated, ranked by severity, and converted into this
    structured form *before* any repair happens (Rule 4)."""

    verdict: CritiqueVerdict = CritiqueVerdict.REVISE
    confidence: float = Field(default=0.0)
    critical_issues: list[CritiqueIssue] = Field(default_factory=list)
    minor_issues: list[CritiqueIssue] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    repair_plan: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def to_critique(self, target: str = "") -> "Critique":
        """Fold the synthesis into a :class:`Critique` for downstream repair."""
        return Critique(
            target=target,
            verdict=self.verdict,
            confidence=self.confidence,
            critical_issues=list(self.critical_issues),
            minor_issues=list(self.minor_issues),
            missing_sections=list(self.missing_sections),
            repair_plan=list(self.repair_plan),
        )


# --------------------------------------------------------------------------- #
# Drafts (Phase 5)
# --------------------------------------------------------------------------- #
class Draft(V2Model):
    id: str
    content: str
    approach: str = ""
    source: str = ""
    diversity_note: str = ""


# --------------------------------------------------------------------------- #
# Evaluation (Phase 6)
# --------------------------------------------------------------------------- #
class Evaluation(V2Model):
    draft_id: str
    score: float = Field(default=0.0)
    verification_passed: bool = False
    verification_notes: str = ""
    issues: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)  # v2.2

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        return max(0.0, min(10.0, float(v)))


# --------------------------------------------------------------------------- #
# Selection (Phase 6 output)
# --------------------------------------------------------------------------- #
class SelectionResult(V2Model):
    selected_draft_id: str = ""
    selected_content: str = ""
    evaluations: list[Evaluation] = Field(default_factory=list)
    pairwise_results: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    method: str = "verification"
    # v2.5 Phase 6: concrete fixes the selected draft still needs and any claims
    # it makes without supporting evidence (fed into final synthesis).
    required_fixes: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# --------------------------------------------------------------------------- #
# Pairwise judgment (Mechanic F)
# --------------------------------------------------------------------------- #
class PairwiseJudgment(V2Model):
    winner: str = "tie"  # "A" | "B" | "tie"
    reason: str = ""

    @field_validator("winner")
    @classmethod
    def _normalize_winner(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v in ("A", "B"):
            return v
        return "tie"


# --------------------------------------------------------------------------- #
# Final answer (Phase 7 output)
# --------------------------------------------------------------------------- #
class FinalAnswer(V2Model):
    answer: str = ""
    verification_passed: bool = False
    verification_notes: str = ""
    confidence: float = Field(default=0.0)
    ungrounded_claims: list[str] = Field(default_factory=list)
    # v2.6 Lightning: delivery metadata (mode, verified flag, UI banner label,
    # truncation flag). The label lives HERE, never in the answer body.
    metadata: dict = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# --------------------------------------------------------------------------- #
# v2.2: Answer Contract (Phase 0/1)
# --------------------------------------------------------------------------- #
class AnswerContract(V2Model):
    """The contract the final answer must satisfy (v2.2). Produced by Phase 0,
    refined by Phase 1, enforced by Phase 5 (drafts) and Phase 8 (coverage)."""

    answer_type: AnswerType = AnswerType.OPEN_ENDED_DESIGN
    length_policy: LengthPolicy = LengthPolicy.STRUCTURED
    required_sections: list[str] = Field(default_factory=list)
    must_include_subproblem_ids: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    min_completeness_score: float = Field(default=0.5)

    @field_validator("min_completeness_score")
    @classmethod
    def _clamp_min_completeness(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# --------------------------------------------------------------------------- #
# v2.6: Lightning intake (L0)
# --------------------------------------------------------------------------- #
class LightningIntake(V2Model):
    """v2.6 Lightning L0: one fast intake call that routes the task and defines
    the minimal answer contract (required sections + forbidden patterns) for
    the single one-shot answer. This is NOT the full Phase 0-8 profile."""

    route: Route = Route.NON_EXECUTABLE
    answer_type: AnswerType = AnswerType.FACTUAL
    required_sections: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    # v2.6: for how-to/build tasks, the minimal actionability the answer must
    # cover (e.g. ["minimal_path", "success_check", "limitations"]).
    actionability_requirements: list[str] = Field(default_factory=list)
    # v2.6: True when the task needs a tiny reasoning step (logic/planning/
    # tradeoffs) before the one-shot answer. Bounded to 3 bullets / 300 tokens.
    needs_reasoning: bool = False


# --------------------------------------------------------------------------- #
# v2.2: Coverage Report (Phase 8)
# --------------------------------------------------------------------------- #
class CoverageReport(V2Model):
    """Phase 8 final coverage gate result.

    v2.2 fields (``passed`` / ``forbidden_patterns_detected``) remain the
    canonical, pipeline-facing fields. v2.4 adds the LLM-emitted aliases
    (``coverage_pass`` / ``forbidden_patterns`` / ``verification_status``) and
    a post-init sync so a model that emits the v2.4 schema still drives the
    same gate. ``passed`` is a bool (not the reserved keyword ``pass``)."""

    passed: bool = True
    missing_sections: list[str] = Field(default_factory=list)
    weak_sections: list[str] = Field(default_factory=list)
    scope_issues: list[str] = Field(default_factory=list)
    forbidden_patterns_detected: list[str] = Field(default_factory=list)
    repair_instructions: str = ""
    completeness_score: float = Field(default=1.0)
    # v2.4: LLM-emitted coverage schema (accepted and synced into the fields
    # above). ``None`` means "the model did not emit this field".
    coverage_pass: Optional[bool] = None
    forbidden_patterns: Optional[list[str]] = None
    verification_status: str = "pass"
    # Clean-plan Fix 3: the LLM may emit the reserved-word key ``"pass"``; accept
    # it (via validation alias) and fold it into the canonical ``passed`` field.
    pass_key: Optional[bool] = Field(default=None, validation_alias="pass")
    # Clean-plan DoD #9: the gate must also confirm that every required fix from
    # the selector is addressed and that the actionability requirements hold.
    missing_required_fixes: list[str] = Field(default_factory=list)
    actionability_pass: bool = True

    @field_validator("repair_instructions", mode="before")
    @classmethod
    def _coerce_repair_instructions(cls, v: Any) -> str:
        """Clean-plan Fix 3: the LLM may emit ``repair_instructions`` as a list;
        coerce it into the single string the pipeline consumes."""
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return "; ".join(str(x) for x in v if str(x).strip())
        return v if isinstance(v, str) else str(v)

    @field_validator("completeness_score")
    @classmethod
    def _clamp_completeness(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def _sync_v24(self) -> "CoverageReport":
        """Fold the LLM-emitted coverage aliases into the canonical fields."""
        # ``"pass"`` (reserved word) is the most common LLM spelling of the gate.
        if self.pass_key is not None:
            self.passed = bool(self.pass_key)
        if self.coverage_pass is not None:
            self.passed = bool(self.coverage_pass)
        if self.forbidden_patterns is not None:
            merged = list(
                dict.fromkeys(
                    list(self.forbidden_patterns) + list(self.forbidden_patterns_detected)
                )
            )
            self.forbidden_patterns_detected = merged
            self.forbidden_patterns = merged
        if self.verification_status not in ("pass", "partial"):
            self.verification_status = "partial"
        return self

    @property
    def coverage_ok(self) -> bool:
        """The gate passes only when coverage passes, no forbidden pattern was
        detected, every required fix is addressed, and actionability holds
        (Clean-plan DoD #9: headings alone are not enough)."""
        return (
            bool(self.passed)
            and not self.forbidden_patterns_detected
            and not self.missing_required_fixes
            and bool(self.actionability_pass)
        )


# --------------------------------------------------------------------------- #
# v2.2: Emergency FINISH NOW (anytime synthesis)
# --------------------------------------------------------------------------- #
class EmergencyFinish(V2Model):
    """Structured output of the anytime FINISH NOW synthesis (v2.2)."""

    final_answer: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    completed_parts: list[str] = Field(default_factory=list)
    incomplete_parts: list[str] = Field(default_factory=list)
    verified_parts: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# v2.2: Budget State (Budget Governor)
# --------------------------------------------------------------------------- #
class BudgetState(V2Model):
    """Snapshot of the v2.2 budget governor, carried in ActiveContext."""

    mode: str = "standard"
    final_reserve_tokens: int = 6000
    final_repair_reserve_tokens: int = 2000
    intermediate_soft_cap: float = 0.7
    used_tokens: int = 0
    remaining_tokens: int = 0
    truncations: int = 0
    warnings: list[str] = Field(default_factory=list)
    final_answer_protected: bool = False
    partial_answer_allowed: bool = False
