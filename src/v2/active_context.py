"""ActiveContext — the only thing that travels between phases.

The v2.0 pipeline never passes raw reasoning traces from one phase to the
next. Each phase receives a fresh context window containing this compact,
verified artifact container and returns structured artifacts that are merged
back into it. This is what makes the architecture verifier-first and
context-isolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.v2.contracts import (
    AnswerContract,
    BranchSet,
    BudgetState,
    CoverageReport,
    Critique,
    DebateSynthesis,
    DecompositionBranch,
    Draft,
    Evaluation,
    FactsLedgerEntry,
    FinalAnswer,
    Plan,
    SelectionResult,
    SubProblemResult,
    TaskProfile,
)


@dataclass
class ActiveContext:
    task_id: str
    task: str
    current_phase: str = "phase0"
    progress_pct: float = 0.0
    task_profile: TaskProfile | None = None
    plan: Plan | None = None
    branches: list[DecompositionBranch] = field(default_factory=list)
    sub_problem_results: list[SubProblemResult] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    selection: SelectionResult | None = None
    final_answer: FinalAnswer | None = None
    notes: list[str] = field(default_factory=list)
    # v2.2 fields
    branch_set: BranchSet | None = None
    integration_requirement: str = ""
    coverage_report: CoverageReport | None = None
    budget_state: BudgetState | None = None
    partial: bool = False
    truncations: int = 0
    # v2.4 fields: structured debate synthesis (Rule 4) + dynamic FINISH NOW (Rule 7)
    debate_synthesis: DebateSynthesis | None = None
    finish_strategy: str = ""
    # v2.6 Lightning: keyed full-length artifacts (e.g. "lightning_answer") that
    # are the source of truth for delivery — never the live stream buffer.
    artifacts: dict = field(default_factory=dict)

    # -- v2.2 delegated accessors ----------------------------------------- #
    @property
    def facts(self) -> list[FactsLedgerEntry]:
        return self.task_profile.facts if self.task_profile else []

    @property
    def answer_contract(self) -> AnswerContract | None:
        return self.task_profile.answer_contract if self.task_profile else None

    @property
    def mode(self) -> str:
        if self.task_profile is not None:
            return self.task_profile.budget.mode
        return "standard"

    # -- artifact accessors ------------------------------------------------- #
    def verified_sub_problems(self) -> list[SubProblemResult]:
        return [r for r in self.sub_problem_results if r.verified]

    def result_by_id(self, sub_problem_id: str) -> SubProblemResult | None:
        for r in self.sub_problem_results:
            if r.sub_problem_id == sub_problem_id:
                return r
        return None

    def sub_problem_ids(self) -> list[str]:
        return [r.sub_problem_id for r in self.sub_problem_results]

    def all_sub_problems(self) -> list:
        out = []
        for branch in self.branches:
            out.extend(branch.sub_problems)
        return out

    def selected_draft(self) -> Draft | None:
        if self.selection is None or not self.selection.selected_draft_id:
            return None
        for d in self.drafts:
            if d.id == self.selection.selected_draft_id:
                return d
        return None

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "current_phase": self.current_phase,
            "progress_pct": round(self.progress_pct, 1),
            "task_profile": self.task_profile.model_dump() if self.task_profile else None,
            "plan": self.plan.model_dump() if self.plan else None,
            # Clean-plan Fix 4: serialize the Phase 2 branch set (wrapper that
            # also carries the integration requirement).
            "branch_set": self.branch_set.model_dump() if self.branch_set else None,
            "branches": [b.model_dump() for b in self.branches],
            "sub_problem_results": [r.model_dump() for r in self.sub_problem_results],
            "critiques": [c.model_dump() for c in self.critiques],
            "drafts": [d.model_dump() for d in self.drafts],
            "evaluations": [e.model_dump() for e in self.evaluations],
            "selection": self.selection.model_dump() if self.selection else None,
            "final_answer": self.final_answer.model_dump() if self.final_answer else None,
            "notes": list(self.notes),
            "answer_contract": (
                self.answer_contract.model_dump() if self.answer_contract else None
            ),
            "coverage_report": (
                self.coverage_report.model_dump() if self.coverage_report else None
            ),
            "budget_state": (
                self.budget_state.model_dump() if self.budget_state else None
            ),
            "integration_requirement": self.integration_requirement,
            "partial": self.partial,
            "truncations": self.truncations,
            "debate_synthesis": (
                self.debate_synthesis.model_dump() if self.debate_synthesis else None
            ),
            "finish_strategy": self.finish_strategy,
            "artifacts": dict(self.artifacts),
        }

    def compact_text(self) -> str:
        """Compact, verified-artifact-only text for prompts / synthesis / UI."""
        lines: list[str] = []
        lines.append(f"TASK: {self.task}")
        if self.task_profile is not None:
            p = self.task_profile
            lines.append(
                f"ROUTE: {p.route.value} | DIFFICULTY: {p.difficulty.value} | "
                f"RETRIEVAL: {'available' if p.retrieval_available else 'unavailable'}"
            )
            if p.summary:
                lines.append(f"SUMMARY: {p.summary}")
            if p.key_requirements:
                lines.append("REQUIREMENTS: " + "; ".join(p.key_requirements))
            grounded = p.grounded_facts()
            if grounded:
                lines.append("GROUNDED FACTS:")
                for f in grounded:
                    lines.append(f"  - {f.fact} [source: {f.source}]")
            ungrounded = p.ungrounded_facts()
            if ungrounded:
                lines.append("UNGROUNDED ASSUMPTIONS (verify or flag):")
                for f in ungrounded:
                    lines.append(f"  - {f.fact}")
        contract = self.answer_contract
        if contract is not None:
            secs = ", ".join(contract.required_sections) or "(none)"
            lines.append(
                f"ANSWER CONTRACT: type={contract.answer_type.value} "
                f"length={contract.length_policy.value} "
                f"required_sections=[{secs}]"
            )
        if self.integration_requirement:
            lines.append(f"INTEGRATION: {self.integration_requirement}")
        if self.plan is not None:
            pl = self.plan
            if pl.approach:
                lines.append(f"PLAN APPROACH: {pl.approach}")
            if pl.strategy:
                lines.append(f"PLAN STRATEGY: {pl.strategy}")
            if pl.risks:
                lines.append("PLAN RISKS: " + "; ".join(pl.risks))
        if self.branches:
            lines.append("DECOMPOSITION:")
            for b in self.branches:
                lines.append(f"  branch {b.id}: {b.rationale}")
                for sp in b.sub_problems:
                    lines.append(
                        f"    - {sp.id} [{sp.type.value}/{sp.risk.value}]: {sp.description}"
                    )
        if self.sub_problem_results:
            lines.append("VERIFIED SUB-PROBLEM RESULTS:")
            for r in self.sub_problem_results:
                status = "verified" if r.verified else "UNVERIFIED"
                vstatus = r.verification.status.value if r.verification else "none"
                lines.append(
                    f"  - {r.sub_problem_id} ({status}, {r.verification_method}, "
                    f"verification={vstatus}, conf={r.confidence:.2f}): {r.answer}"
                )
        if self.debate_synthesis is not None:
            ds = self.debate_synthesis
            lines.append(
                f"DEBATE SYNTHESIS: verdict={ds.verdict.value} "
                f"critical={len(ds.critical_issues)} minor={len(ds.minor_issues)} "
                f"repair_steps={len(ds.repair_plan)}"
            )
        if self.critiques:
            lines.append("CRITIQUES:")
            for c in self.critiques:
                lines.append(
                    f"  - [{c.verdict.value}] {c.target}: "
                    + ("; ".join(c.issues) if c.issues else "no issues")
                )
        if self.drafts:
            lines.append("DRAFTS:")
            for d in self.drafts:
                lines.append(f"  - {d.id} ({d.approach or d.source}): {d.content[:400]}")
        if self.evaluations:
            lines.append("EVALUATIONS:")
            for e in self.evaluations:
                lines.append(
                    f"  - {e.draft_id}: score={e.score:.1f} "
                    f"verified={e.verification_passed}"
                )
        if self.selection is not None and self.selection.selected_content:
            lines.append(
                f"SELECTED DRAFT ({self.selection.method}, "
                f"conf={self.selection.confidence:.2f}): "
                f"{self.selection.selected_content[:800]}"
            )
        if self.final_answer is not None and self.final_answer.answer:
            lines.append(
                f"FINAL ANSWER (verified={self.final_answer.verification_passed}): "
                f"{self.final_answer.answer}"
            )
        if self.coverage_report is not None:
            cr = self.coverage_report
            lines.append(
                f"COVERAGE: passed={cr.passed} "
                f"missing={'; '.join(cr.missing_sections) or 'none'}"
            )
        if self.partial:
            lines.append(f"PARTIAL ANSWER (truncations={self.truncations})")
        return "\n".join(lines)
