"""v2.2 phase prompts.

Every prompt is self-contained (no global state) and asks for a single JSON
object wrapped in explicit markers. The v2.2 additions:

- :data:`V22_SYSTEM_PROMPT` — a global system prompt enforcing no-fabrication,
  explicit assumptions, concise intermediates and comprehensive finals.
- Answer-contract-aware phase prompts: the contract (required sections, length
  policy, forbidden patterns) is injected into the phases that produce or check
  the final answer, so the final answer is never reduced to the last step.
- Phase 8 prompts: a final *coverage* validator and a *section-only* repair.
- An *emergency FINISH NOW* prompt for the anytime synthesis.

Existing v2.0 phase markers are preserved so the deterministic test transport
keeps working.
"""
from __future__ import annotations

import json
import re

from src.v2.active_context import ActiveContext
from src.v2.contracts import (
    AnswerContract,
    BranchSet,
    CoverageReport,
    Critique,
    DebateSynthesis,
    Draft,
    EmergencyFinish,
    Evaluation,
    FinalAnswer,
    LightningIntake,
    PairwiseJudgment,
    Plan,
    SubProblemCandidate,
    SubProblemResult,
    TaskProfile,
    VerificationReport,
)
from src.v2.structured import build_schema_instruction

# --------------------------------------------------------------------------- #
# v2.4 global system prompt (no persona)
# --------------------------------------------------------------------------- #
# v2.4 Rule 1: NO persona prompts. The prompt instructs the phase to complete,
# it never says "You are ...". ``V22_SYSTEM_PROMPT`` is kept as an alias so the
# existing test surface keeps working.
V24_SYSTEM_PROMPT = (
    "Complete the assigned phase. "
    "HARD RULES: "
    "(1) Never fabricate facts, results, code behavior, tests, or citations; "
    "if something is not verified, say so explicitly. "
    "(2) State assumptions and confidence when information is missing rather "
    "than guessing. "
    "(3) Prefer verified evidence: execution results, test results, retrieved "
    "facts, and explicit constraints. "
    "(4) Keep INTERMEDIATE outputs concise and structured — spend tokens on "
    "verification, not prose. "
    "(5) The FINAL answer must be complete and, unless the user explicitly asks "
    "for brevity, comprehensive: cover every required section, integrate every "
    "solved sub-problem, and never reduce it to a single last step. "
    "(6) Use step-by-step reasoning only if it is needed to reach the answer. "
    "(7) Return ONLY the output format requested by the phase — one output mode "
    "per call."
)
V22_SYSTEM_PROMPT = V24_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# v2.5 Rule 1: adaptive per-phase system prompts
# --------------------------------------------------------------------------- #
# v2.5 turns the single global system prompt into a set of phase-specific system
# prompts. Each one states the phase name, its functional stance, the constraints
# for that phase, and the output mode. The *user* prompt carries the data (task,
# prior verified results, retrieved facts, debate issues, drafts, verification
# reports); the *system* prompt carries the phase rules.
_PHASE_SYSTEM_PROMPTS: dict[str, str] = {
    "phase0": (
        "Phase: 0 Intake\n"
        "Stance: Task classifier\n"
        "Constraints:\n"
        "- Classify the task. Do not solve the task.\n"
        "- Identify the required final-answer sections and the answer contract.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase1": (
        "Phase: 1 Planning\n"
        "Stance: Verification-oriented planner\n"
        "Constraints:\n"
        "- Create a plan only. Do not solve the task.\n"
        "- Define acceptance criteria and name the riskiest sub-problem.\n"
        "- Reserve tokens for final synthesis and repair.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase2": (
        "Phase: 2 Decomposition\n"
        "Stance: Task decomposer\n"
        "Constraints:\n"
        "- Produce 2 to 6 sub-problems, each simpler than the original task.\n"
        "- Include an integration requirement. Do not solve the sub-problems.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase3": (
        "Phase: 3 Sub-problem solving\n"
        "Stance: Sub-problem solver\n"
        "Constraints:\n"
        "- Solve only the assigned sub-problem. Use prior verified results only.\n"
        "- If code is required, provide code and tests.\n"
        "- Do not claim execution results unless execution evidence exists.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase4": (
        "Phase: 4 Debate synthesis & critique\n"
        "Stance: Issue deduplicator, severity ranker, and falsifier\n"
        "Constraints:\n"
        "- Merge duplicate findings, rank issues by severity, and convert each into a repair action.\n"
        "- Falsify the target; a vague 'revise' with no structured issues is rejected.\n"
        "- Do not solve the original task.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase5": (
        "Phase: 5 Drafting & targeted repair\n"
        "Stance: Repair executor and draft writer\n"
        "Constraints:\n"
        "- Repair only the listed issues. Do not rewrite sections that already pass.\n"
        "- Integrate all solved sub-problems into one complete draft.\n"
        "- Do not discuss output format. Use verified information only.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase6": (
        "Phase: 6 Selection\n"
        "Stance: Evidence-based selector\n"
        "Constraints:\n"
        "- Prefer execution evidence for code; prefer retrieval grounding and self-consistency otherwise.\n"
        "- Score each draft, list missing sections, and name required fixes and unsupported claims.\n"
        "- Do not reward fluency alone.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase7": (
        "Phase: 7 Final synthesis\n"
        "Stance: Comprehensive synthesizer\n"
        "Constraints:\n"
        "- Integrate all verified sub-problem results. Do not return only the last sub-problem.\n"
        "- Do not omit major completed sections. Be comprehensive and organized.\n"
        "- Label incomplete or unverified parts clearly.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase7_verify": (
        "Phase: 7 Final verification\n"
        "Stance: Final answer verifier\n"
        "Constraints:\n"
        "- Verify the final answer against the task and the answer contract.\n"
        "- Emit a structured critique with concrete repair targets if defects exist.\n"
        "Output mode: MARKED_JSON"
    ),
    "phase8": (
        "Phase: 8 Coverage gate\n"
        "Stance: Final coverage validator\n"
        "Constraints:\n"
        "- Validate the final answer against the answer contract. Do not rewrite the answer.\n"
        "- Identify missing sections, weak sections, and forbidden patterns.\n"
        "Output mode: MARKED_JSON"
    ),
    "emergency_finish": (
        "Phase: Emergency Finish\n"
        "Stance: Best-effort synthesizer\n"
        "Constraints:\n"
        "- Use available partial state only. Do not invent missing work.\n"
        "- Do not wait for incomplete processes. Clearly mark incomplete parts.\n"
        "Output mode: MARKED_JSON"
    ),
    # v2.6 Lightning: the short pipeline (L0 intake, L1 one-shot answer,
    # L2 fast-gate repair). No persona, one output mode per call.
    "lightning_intake": (
        "Phase: L0 Lightning intake\n"
        "Stance: Fast task classifier\n"
        "Constraints:\n"
        "- Classify the task and define the minimal answer contract only.\n"
        "- Do not solve the task. Keep required sections to the minimum needed.\n"
        "Output mode: MARKED_JSON"
    ),
    "lightning_answer": (
        "Phase: L1 Lightning one-shot answer\n"
        "Stance: Concise direct answer writer\n"
        "Constraints:\n"
        "- Answer directly and concisely in plain prose; no JSON, no markers.\n"
        "- Include only the required sections; for how-to tasks give the minimal practical path.\n"
        "- Do not overclaim what is not verified. Do not discuss output format.\n"
        "Output mode: PLAIN_FINAL"
    ),
    "lightning_repair": (
        "Phase: L2 Lightning fast-gate repair\n"
        "Stance: Targeted answer repairer\n"
        "Constraints:\n"
        "- Repair ONLY the listed gate issues; preserve everything else intact.\n"
        "- Plain prose output; no JSON, no markers.\n"
        "Output mode: PLAIN_FINAL"
    ),
}


def _canonical_phase(phase: str) -> str:
    """Map a granular pipeline phase label to its canonical v2.5 phase key.

    The pipeline labels calls with granular names (``phase3_code``,
    ``phase6_eval``, ``phase8_repair``, ``finish_now``); this collapses them to
    the canonical keys used by :data:`_PHASE_SYSTEM_PROMPTS`.
    """
    p = (phase or "").strip().lower()
    if not p:
        return ""
    if p in ("finish_now", "emergency_finish", "emergency", "finish"):
        return "emergency_finish"
    if p == "phase7_verify":
        return "phase7_verify"
    # v2.6 Lightning stages (L0/L1/L2) map to their own canonical keys.
    if p.startswith("lightning"):
        key = p.split("_", 1)[1] if "_" in p else "lightning"
        if key in ("intake", "answer", "repair"):
            return f"lightning_{key}"
        return "lightning_intake"
    m = re.match(r"(phase\d+)", p)
    if m:
        return m.group(1)
    return ""


def phase_system_prompt(phase: str) -> str:
    """Return the v2.5 adaptive system prompt for a pipeline phase.

    Falls back to the global :data:`V24_SYSTEM_PROMPT` for unknown phases so an
    unrecognized label never leaves a call without a system prompt.
    """
    key = _canonical_phase(phase)
    return _PHASE_SYSTEM_PROMPTS.get(key, V24_SYSTEM_PROMPT)


# --------------------------------------------------------------------------- #
# v2.4 output modes (Rule 2: one output mode per call)
# --------------------------------------------------------------------------- #
# v2.6 (Bug 5): output modes are DISTINCT and never mixed:
# - JSON_ONLY   : raw JSON only — one JSON object, no markers, no prose.
# - MARKED_JSON : one JSON object wrapped in FINAL_JSON_START / FINAL_JSON_END
#                 markers and nothing else (the pipeline's default structured mode).
# - PLAIN_FINAL : prose/markdown only, no JSON at all.
# - SECTION_JSON: one JSON object of named sections.
# - ANSWER_LINE : a short answer whose final line begins with 'ANSWER:'.
OUTPUT_MODES = (
    "JSON_ONLY",
    "MARKED_JSON",
    "PLAIN_FINAL",
    "SECTION_JSON",
    "ANSWER_LINE",
)

_OUTPUT_MODE_TEXT = {
    "JSON_ONLY": (
        "respond with exactly one raw JSON object — no markers, no prose, "
        "nothing else."
    ),
    "MARKED_JSON": (
        "respond with exactly one JSON object between the FINAL_JSON_START and "
        "FINAL_JSON_END markers and nothing else — no prose before or after."
    ),
    "SECTION_JSON": (
        "respond with exactly one JSON object of named sections between the "
        "markers and nothing else."
    ),
    "PLAIN_FINAL": "respond with prose/markdown only, no JSON and no markers.",
    "ANSWER_LINE": (
        "respond with a short answer; the final line must begin with 'ANSWER:'."
    ),
}


def output_mode_directive(mode: str = "JSON_ONLY") -> str:
    """v2.4: an explicit single-output-mode line for a prompt (Rule 2)."""
    if mode not in _OUTPUT_MODE_TEXT:
        mode = "JSON_ONLY"
    return f"OUTPUT MODE: {mode} — {_OUTPUT_MODE_TEXT[mode]}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def first_nonempty(*values) -> str:
    for v in values:
        if v is not None and str(v).strip():
            return v
    return ""


def _sub_problem_block(sp: SubProblemCandidate) -> str:
    return (
        f"{sp.id}: {sp.description} "
        f"(type={sp.type.value}, risk={sp.risk.value}, "
        f"deps={', '.join(sp.depends_on) or 'none'})"
    )


def _facts_block(context: ActiveContext) -> str:
    if not context.facts:
        return ""
    lines = ["FACTS LEDGER (only these are grounded):"]
    for f in context.facts[:20]:
        lines.append(f"  - [{f.source}] {f.fact}")
    return "\n".join(lines) + "\n"


def _contract(context: ActiveContext) -> AnswerContract | None:
    """Best-effort access to the current answer contract."""
    return getattr(context, "answer_contract", None)


def _contract_block(contract: AnswerContract | None) -> str:
    if contract is None:
        return ""
    sections = "\n".join(f"  - {s}" for s in contract.required_sections) or "  (none specified)"
    must_ids = ", ".join(contract.must_include_subproblem_ids) or "(none)"
    forbidden = "\n".join(f"  - {p}" for p in contract.forbidden_patterns) or "  (none)"
    return (
        "\nANSWER CONTRACT (the final answer MUST satisfy this):\n"
        f"  answer_type: {contract.answer_type.value}\n"
        f"  length_policy: {contract.length_policy.value}\n"
        f"  required_sections:\n{sections}\n"
        f"  must_include_subproblem_ids: {must_ids}\n"
        f"  forbidden_patterns (must NOT appear):\n{forbidden}\n"
        f"  min_completeness_score: {contract.min_completeness_score:.2f}\n"
    )


# --------------------------------------------------------------------------- #
# Phase 0 — Intake, routing, retrieval, ANSWER CONTRACT
# --------------------------------------------------------------------------- #
def build_phase0_prompt(
    task_text: str,
    retrieval_available: bool,
    retrieved: list[str] | None = None,
) -> str:
    retrieved = retrieved or []
    if retrieval_available and retrieved:
        facts = (
            "RETRIEVAL AVAILABLE: yes\nFACTS:\n"
            + "\n".join(f"- {f}" for f in retrieved)
            + "\n"
        )
    elif retrieval_available:
        facts = "RETRIEVAL AVAILABLE: yes (no results returned)\n"
    else:
        facts = "RETRIEVAL AVAILABLE: no — no retrieval results available for this task.\n"
    return (
        "INTAKE & ROUTING stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Classify the task, estimate difficulty, decide which sub-tasks are "
        "executable vs non-executable, and produce the ANSWER CONTRACT that the "
        "final answer must satisfy.\n\n"
        f"TASK:\n{task_text}\n\n"
        f"{facts}\n"
        "In the JSON, set `answer_contract` with: answer_type "
        "(open_ended_design|code|factual|analysis|hybrid), length_policy "
        "(concise|structured|comprehensive), required_sections (list of section "
        "titles the final answer must contain), must_include_subproblem_ids "
        "(sub-problem ids that must appear in the final answer), "
        "forbidden_patterns (phrases the final answer must avoid), and "
        "min_completeness_score (0..1).\n"
        + build_schema_instruction(TaskProfile)
    )


# --------------------------------------------------------------------------- #
# Phase 1 — Meta-plan (refines the answer contract)
# --------------------------------------------------------------------------- #
def build_phase1_prompt(context: ActiveContext) -> str:
    profile = context.task_profile
    contract = _contract(context)
    return (
        "PLANNING stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Produce a concise verification-oriented meta-plan. State the approach, "
        "key requirements, and verification strategy. Define "
        "acceptance_criteria the final answer must satisfy, name the "
        "riskiest_subproblem, and set budget reserves (final_reserve_tokens and "
        "final_repair_reserve_tokens) to protect the final synthesis and its "
        "repair. Refine the ANSWER CONTRACT so its required_sections match the "
        "plan's key requirements.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"DIFFICULTY: {profile.difficulty.value if profile else 'medium'}\n"
        f"RETRIEVAL: {'available' if (profile and profile.retrieval_available) else 'unavailable'}\n\n"
        "CURRENT ANSWER CONTRACT:\n"
        f"{json.dumps(contract.model_dump() if contract else {}, indent=2)}\n\n"
        + _facts_block(context)
        + build_schema_instruction(Plan)
    )


# --------------------------------------------------------------------------- #
# Phase 2 — Decomposition (adds integration requirement)
# --------------------------------------------------------------------------- #
def build_phase2_prompt(context: ActiveContext, n_branches: int) -> str:
    plan = context.plan
    requirements = (
        "\n".join(f"  - {r}" for r in plan.key_requirements)
        if plan and plan.key_requirements
        else "  (none)"
    )
    return (
        "DECOMPOSITION stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        f"Break the task into {n_branches} independent, verifiable "
        "sub-problem branch(es). Set `integration_requirement` describing how "
        "the sub-problem results must be combined into the final answer (so the "
        "final answer is never just the last sub-problem).\n\n"
        f"TASK:\n{context.task}\n\n"
        f"KEY REQUIREMENTS:\n{requirements}\n\n"
        + _facts_block(context)
        + build_schema_instruction(BranchSet)
    )


# --------------------------------------------------------------------------- #
# Phase 3 — Sub-problem solving
# --------------------------------------------------------------------------- #
def build_phase3_solve_prompt(context: ActiveContext, sp: SubProblemCandidate) -> str:
    return (
        "SOLVING one sub-problem (non-executable).\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Solve ONLY this sub-problem. Be concise. State assumptions. Do not "
        "solve other sub-problems. Fill the `verification` object: status "
        "(pass|fail|partial|unverified), evidence (checks/facts that support "
        "the answer), and missing (what could not be verified). If you could "
        "not verify it, set status=unverified and a reason.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"SUB-PROBLEM:\n{_sub_problem_block(sp)}\n"
        f"VERIFICATION HINT: {sp.verification_hint or '(none)'}\n\n"
        + _facts_block(context)
        + build_schema_instruction(SubProblemResult)
    )


def build_phase3_code_prompt(context: ActiveContext, sp: SubProblemCandidate) -> str:
    return (
        "WRITING CODE to solve one executable sub-problem.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Write minimal, self-contained Python that solves ONLY this "
        "sub-problem. Include a few asserts that verify the result. Be "
        "concise. Put the code in the `answer` field (or `execution.code`). "
        "Fill the `verification` object: status (pass|fail|partial|unverified), "
        "evidence (the execution/test evidence), and missing (what was not "
        "run). If the sandbox could not run the code, set status=unverified "
        "with reason=sandbox_unavailable.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"SUB-PROBLEM:\n{_sub_problem_block(sp)}\n"
        f"VERIFICATION HINT: {sp.verification_hint or '(none)'}\n\n"
        + _facts_block(context)
        + build_schema_instruction(SubProblemResult)
    )


# --------------------------------------------------------------------------- #
# Phase 4 — Falsifiable critique (structured self-refine)
# --------------------------------------------------------------------------- #
def build_phase4_critique_prompt(
    context: ActiveContext,
    target: str,
    label: str,
    sp: "SubProblemCandidate | None" = None,
    result: "SubProblemResult | None" = None,
    agreement: float | None = None,
) -> str:
    """Phase 4 critique prompt (v2.6: full context).

    The critique must receive the original task, the sub-problem id, its
    statement/type/verification hint/prior dependencies, the result answer,
    the verification report, and the self-consistency agreement (with an
    explicit low-agreement flag) so it can falsify against what was actually
    checked — not just the bare answer text.
    """
    sp_block = ""
    if sp is not None:
        deps = ", ".join(sp.depends_on) or "none"
        sp_block = (
            "SUB-PROBLEM STATEMENT:\n"
            f"  id: {sp.id}\n"
            f"  statement: {sp.description}\n"
            f"  type: {sp.type.value}\n"
            f"  risk: {sp.risk.value}\n"
            f"  verification_hint: {sp.verification_hint or '(none)'}\n"
            f"  depends_on (prior results it builds on): {deps}\n\n"
        )
    ver_block = ""
    if result is not None and result.verification is not None:
        v: VerificationReport = result.verification
        evidence = "\n".join(f"  - {e}" for e in (v.evidence or [])) or "  (none)"
        missing = "\n".join(f"  - {m}" for m in (v.missing or [])) or "  (none)"
        ver_block = (
            "VERIFICATION REPORT (what was actually checked, by the pipeline):\n"
            f"  status: {v.status.value}\n"
            f"  evidence:\n{evidence}\n"
            f"  missing:\n{missing}\n"
            f"  reason: {v.reason or '(none)'}\n\n"
        )
    ag_block = ""
    if agreement is not None:
        ag_block = f"SELF-CONSISTENCY AGREEMENT: {agreement:.2f}\n"
        if agreement < 0.5:
            ag_block += (
                "LOW SELF-CONSISTENCY FLAG: agreement is below 0.5 — the sampled "
                "reasoning paths disagree, so the answer is UNRELIABLE. Treat "
                "this as a critical issue unless concrete evidence overrides it.\n"
            )
    return (
        "VERIFICATION stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Falsify this sub-problem result. If you cannot find a concrete "
        "defect, verdict=pass. If verdict=revise or fail you MUST produce a "
        "structured critique: fill `critical_issues` and/or `minor_issues` "
        "(each with issue_id, severity, description, evidence, and a concrete "
        "repair_action) and a `repair_plan` (ordered, concrete steps). A vague "
        "'revise' with no structured issues and no repair_plan is rejected and "
        "treated as pass. Check the answer against the sub-problem statement, "
        "its verification hint, and the verification report; do not trust the "
        "result's own verified flag.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"TARGET: {label}\n\n"
        f"{sp_block}"
        f"{ver_block}"
        f"{ag_block}"
        f"RESULT TO CHECK:\n{target}\n\n"
        + _facts_block(context)
        + build_schema_instruction(Critique)
    )


# --------------------------------------------------------------------------- #
# v2.4 — debate synthesis (Rule 4) + invalid-critique handling (Rule 5)
# --------------------------------------------------------------------------- #
def build_debate_synthesis_prompt(context: ActiveContext, raw_findings: list[str]) -> str:
    """v2.4 Rule 4: synthesize raw debate findings into ONE structured critique
    *before* any repair. The model must deduplicate, rank by severity, and turn
    each issue into a concrete repair action — never pass raw findings into a
    repair prompt."""
    findings = "\n".join(f"  - {f}" for f in raw_findings) or "  (none)"
    return (
        "DEBATE SYNTHESIS stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Merge the raw debate findings below into ONE structured critique. Do "
        "NOT solve the original task. Do NOT include reasoning in the output. "
        "Deduplicate issues, rank them by severity, select the most important "
        "critical issues, and convert each into a concrete repair_action. "
        "Output only valid JSON matching the schema.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"RAW DEBATE FINDINGS:\n{findings}\n\n"
        + _facts_block(context)
        + build_schema_instruction(DebateSynthesis)
    )


def build_invalid_critique_retry_prompt(
    context: ActiveContext, target: str, label: str, error: str
) -> str:
    """v2.4 Rule 5: first retry after an invalid / unparseable critique —
    stricter, JSON-only, no reasoning."""
    return (
        "VERIFICATION stage (retry).\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        f"Your previous output was invalid: {error}. Output ONLY valid JSON. "
        "Do not include reasoning. Re-emit the structured critique for this "
        f"target (verdict, critical_issues, minor_issues, repair_plan).\n\n"
        f"TASK:\n{context.task}\n\n"
        f"TARGET: {label}\n\n"
        f"RESULT TO CHECK:\n{target}\n\n"
        + build_schema_instruction(Critique)
    )


def build_invalid_critique_reduced_prompt(
    context: ActiveContext, target: str, label: str
) -> str:
    """v2.4 Rule 5: reduced-scope fallback — return only the top 3 critical
    issues so a struggling model can still emit a usable structured critique."""
    return (
        "VERIFICATION stage (reduced scope).\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Return only the top 3 critical issues for this target, as a "
        "structured critique (verdict, critical_issues, repair_plan). Output "
        "ONLY valid JSON. Do not include reasoning.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"TARGET: {label}\n\n"
        + build_schema_instruction(Critique)
    )


# --------------------------------------------------------------------------- #
# Phase 5 — Diverse draft generation (contract-aware)
# --------------------------------------------------------------------------- #
def build_phase5_draft_prompt(
    context: ActiveContext,
    draft_index: int,
    n_drafts: int,
    avoid: list[str],
) -> str:
    contract = _contract(context)
    length = contract.length_policy.value if contract else "structured"
    sections = (
        "\n".join(f"  - {s}" for s in contract.required_sections)
        if contract and contract.required_sections
        else "  (none specified)"
    )
    avoid_block = (
        "\n".join(f"  - {a}" for a in avoid) if avoid else "  (none yet)"
    )
    return (
        f"producing DRAFT {draft_index + 1}/{n_drafts}.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Write a complete draft answer to the TASK that integrates ALL solved "
        "sub-problems (do not reduce to a single step). Use the "
        f"{length} length policy and include every required section.\n\n"
        f"TASK:\n{context.task}\n\n"
        f"REQUIRED SECTIONS:\n{sections}\n\n"
        f"APPROACHES TO AVOID (be diverse):\n{avoid_block}\n\n"
        + _facts_block(context)
        + build_schema_instruction(Draft)
    )


# --------------------------------------------------------------------------- #
# Phase 6 — Verifier-first evaluation
# --------------------------------------------------------------------------- #
def build_phase6_eval_prompt(task: str, draft: Draft) -> str:
    return (
        "INDEPENDENT VERIFIER.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Evaluate this draft against the task. Score 0-10. List any "
        "missing_sections (required content that is absent or too weak) and "
        "other issues.\n\n"
        f"TASK:\n{task}\n\n"
        f"DRAFT ({draft.id}):\n{draft.content}\n\n"
        + build_schema_instruction(Evaluation)
    )


def build_phase6_pairwise_prompt(task: str, answer_a: str, answer_b: str) -> str:
    return (
        "Compare two draft answers to the same task as a neutral JUDGE.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Decide which is better overall (A, B, or tie). Be concise.\n\n"
        f"TASK:\n{task}\n\n"
        f"ANSWER A:\n{answer_a}\n\n"
        f"ANSWER B:\n{answer_b}\n\n"
        + build_schema_instruction(PairwiseJudgment)
    )


# --------------------------------------------------------------------------- #
# Phase 7 — Final synthesis (contract-aware, comprehensive)
# --------------------------------------------------------------------------- #
def build_phase7_synthesis_prompt(context: ActiveContext) -> str:
    contract = _contract(context)
    length = contract.length_policy.value if contract else "comprehensive"
    sections = (
        "\n".join(f"  - {s}" for s in contract.required_sections)
        if contract and contract.required_sections
        else "  (none specified)"
    )
    selected = (
        context.selection.selected_content
        if context.selection and context.selection.selected_content
        else first_nonempty(
            *[r.answer for r in context.sub_problem_results], context.task
        )
    )
    integration_req = getattr(context, "integration_requirement", "") or (
        context.branch_set.integration_requirement
        if context.branch_set is not None
        else ""
    )
    integration = (
        f"\nINTEGRATION REQUIREMENT: {integration_req}\n" if integration_req else ""
    )
    # v2.5 Phase 6 -> 7: carry the selector's required fixes and unsupported
    # claims into synthesis so the final answer addresses them explicitly.
    sel = context.selection
    fixes_block = ""
    if sel is not None and (sel.required_fixes or sel.unsupported_claims):
        fixes = "\n".join(f"  - {f}" for f in sel.required_fixes) or "  (none)"
        unsupported = (
            "\n".join(f"  - {c}" for c in sel.unsupported_claims) or "  (none)"
        )
        fixes_block = (
            f"\nREQUIRED FIXES (address each in the final answer):\n{fixes}\n"
            f"UNSUPPORTED CLAIMS (verify or remove each):\n{unsupported}\n"
        )
    return (
        "FINAL SYNTHESIS stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Produce the single best FINAL answer. It MUST integrate every solved "
        "sub-problem and satisfy the answer contract. Use the "
        f"{length} length policy. HARD: do NOT reduce the answer to the last "
        f"sub-problem; do NOT omit any major completed section; include "
        f"verified code when code was verified; label any incomplete parts "
        f"clearly.{integration}{fixes_block}\n\n"
        f"TASK:\n{context.task}\n\n"
        f"REQUIRED SECTIONS:\n{sections}\n\n"
        + _contract_block(contract)
        + f"SELECTED DRAFT:\n{selected}\n\n"
        + _facts_block(context)
        + build_schema_instruction(FinalAnswer)
    )


def build_phase7_verify_prompt(task: str, final_answer: str) -> str:
    return (
        "FINAL VERIFIER.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Verify the final answer against the task. If it is complete and "
        "grounded, verdict=pass. If it has concrete defects, verdict=revise "
        "with structured critical_issues/minor_issues and a repair_plan. Only "
        "verdict=fail if it is fundamentally wrong.\n\n"
        f"TASK:\n{task}\n\n"
        f"FINAL ANSWER:\n{final_answer}\n\n"
        + build_schema_instruction(Critique)
    )


def build_phase7_recover_prompt(
    task: str,
    partial_answer: str,
    missing_sections: list[str],
) -> str:
    """v2.6 truncation recovery (Phase 7): a *smaller-scope* retry after the
    full synthesis truncated. It uses reduced reasoning and generates ONLY the
    missing sections, never repeating the same full generation."""
    missing = (
        "\n".join(f"  - {s}" for s in missing_sections) or "  (complete the answer)"
    )
    return (
        "TRUNCATION RECOVERY stage (Phase 7, reduced scope).\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "The previous final-answer synthesis was truncated. Do NOT regenerate "
        "the whole answer. Using REDUCED reasoning, produce the complete final "
        "answer by keeping the partial answer below and adding ONLY the missing "
        "sections. Be concise.\n\n"
        f"TASK:\n{task}\n\n"
        f"MISSING SECTIONS (generate these):\n{missing}\n\n"
        f"PARTIAL ANSWER SO FAR:\n{partial_answer}\n\n"
        + build_schema_instruction(FinalAnswer)
    )


# --------------------------------------------------------------------------- #
# Phase 8 — Final coverage gate & repair (v2.2)
# --------------------------------------------------------------------------- #
def build_phase8_coverage_prompt(
    task: str,
    final_answer: str,
    contract: AnswerContract | None,
    required_fixes: list[str] | None = None,
) -> str:
    sections = (
        "\n".join(f"  - {s}" for s in contract.required_sections)
        if contract and contract.required_sections
        else "  (none specified)"
    )
    forbidden = (
        "\n".join(f"  - {p}" for p in contract.forbidden_patterns)
        if contract and contract.forbidden_patterns
        else "  (none)"
    )
    fixes = (
        "\n".join(f"  - {f}" for f in required_fixes)
        if required_fixes
        else "  (none)"
    )
    return (
        "COVERAGE VALIDATION stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "Check whether the FINAL ANSWER fully satisfies the answer contract. "
        "Validate only — do NOT rewrite the answer. Set coverage_pass=true "
        "(and passed=true) ONLY if: every required section is present AND "
        "substantive (not just a heading); every REQUIRED FIX below is actually "
        "addressed in the answer; the actionability requirements are met; no "
        "forbidden pattern appears; and the answer integrates all sub-problems. "
        "Otherwise set coverage_pass=false and list missing_sections, "
        "weak_sections, missing_required_fixes, and any forbidden_patterns "
        "detected. Set actionability_pass=true only if the answer is concrete "
        "and usable (for how-to/build tasks: it gives a minimal example, setup "
        "steps, success metrics, and a safety checklist). Set verification_status "
        "to 'pass' or 'partial'.\n\n"
        f"TASK:\n{task}\n\n"
        f"REQUIRED SECTIONS:\n{sections}\n\n"
        f"REQUIRED FIXES (must each be addressed):\n{fixes}\n\n"
        f"FORBIDDEN PATTERNS:\n{forbidden}\n\n"
        f"FINAL ANSWER:\n{final_answer}\n\n"
        + build_schema_instruction(CoverageReport)
    )


def build_phase8_repair_prompt(
    task: str,
    final_answer: str,
    contract: AnswerContract | None,
    coverage: CoverageReport,
    debate_synthesis: DebateSynthesis | None = None,
) -> str:
    missing = (
        "\n".join(f"  - {s}" for s in (coverage.missing_sections or []))
        or "  (none listed)"
    )
    weak = (
        "\n".join(f"  - {s}" for s in (coverage.weak_sections or []))
        or "  (none listed)"
    )
    # Clean-plan DoD #9: required fixes that are still unaddressed must be
    # repaired, not just the missing/weak sections.
    req_fixes = (
        "\n".join(f"  - {f}" for f in (coverage.missing_required_fixes or []))
        or "  (none listed)"
    )
    instructions = coverage.repair_instructions or "(strengthen the missing/weak sections)"
    debate_block = ""
    if debate_synthesis is not None:
        targets: list[str] = []
        for issue in debate_synthesis.critical_issues:
            targets.append(f"  - [critical] {issue.repair_action or issue.description}")
        for step in debate_synthesis.repair_plan:
            targets.append(f"  - {step}")
        if targets:
            debate_block = (
                "STRUCTURED REPAIR TARGETS (from the synthesized debate — repair "
                "ONLY these):\n" + "\n".join(targets) + "\n\n"
            )
    return (
        "SECTION REPAIR stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "The final answer failed the coverage gate. Repair ONLY the missing, "
        "weak, or unaddressed-fix sections. PRESERVE all existing correct "
        "content verbatim; do not rewrite the whole answer. Do not solve "
        "unrelated sub-problems and do not discuss output format. Return the "
        "FULL repaired final answer.\n\n"
        f"TASK:\n{task}\n\n"
        f"MISSING SECTIONS:\n{missing}\n\n"
        f"WEAK SECTIONS:\n{weak}\n\n"
        f"REQUIRED FIXES STILL MISSING:\n{req_fixes}\n\n"
        f"REPAIR INSTRUCTIONS: {instructions}\n\n"
        f"{debate_block}"
        f"CURRENT FINAL ANSWER:\n{final_answer}\n\n"
        + _contract_block(contract)
        + build_schema_instruction(FinalAnswer)
    )


# --------------------------------------------------------------------------- #
# Emergency FINISH NOW (anytime synthesis, v2.2)
# --------------------------------------------------------------------------- #
def build_emergency_finish_prompt(
    task: str,
    context: ActiveContext,
    strategy: str = "",
) -> str:
    contract = _contract(context)
    sections = (
        "\n".join(f"  - {s}" for s in contract.required_sections)
        if contract and contract.required_sections
        else "  (none specified)"
    )
    strategy_line = f"FINISH STRATEGY: {strategy}\n" if strategy else ""
    return (
        "EMERGENCY FINISH NOW.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        + strategy_line
        + "The pipeline was stopped early. Follow the finish strategy and "
        "synthesize the single best final answer from the verified artifacts "
        "below. Be honest about what is complete vs incomplete. Never fabricate. "
        "Include every required section you can. Label incomplete parts "
        "explicitly.\n\n"
        f"TASK:\n{task}\n\n"
        f"REQUIRED SECTIONS:\n{sections}\n\n"
        + _facts_block(context)
        + f"VERIFIED ARTIFACTS:\n{context.compact_text()}\n\n"
        + build_schema_instruction(EmergencyFinish)
    )


# --------------------------------------------------------------------------- #
# v2.6: self-consistency merge (low-agreement recovery)
# --------------------------------------------------------------------------- #
def build_sc_merge_prompt(original_prompt: str, samples: list[str]) -> str:
    """One merge call for low self-consistency agreement: combine the
    disagreeing candidate answers into the single best answer."""
    listing = "\n".join(f"- {s}" for s in samples[:8]) or "- (none)"
    return (
        "SELF-CONSISTENCY MERGE stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "The independent reasoning paths disagree (low self-consistency "
        "agreement). Combine the candidate answers into the single best "
        "answer. Keep only what is supported by more than one candidate where "
        "they conflict; if they irreconcilably conflict, give the most "
        "defensible answer and note the conflict in `reasoning`.\n\n"
        f"ORIGINAL TASK PROMPT:\n{original_prompt}\n\n"
        f"CANDIDATE ANSWERS:\n{listing}\n\n"
        + build_schema_instruction(SubProblemResult)
    )


# --------------------------------------------------------------------------- #
# v2.6: Lightning — the short pipeline (L0-L3), NOT Phases 0-8
# --------------------------------------------------------------------------- #
# The delivery label: every Lightning answer is self-labelled as fast but not
# deeply verified. It is carried as delivery METADATA (a UI banner), never
# spliced into the answer body.
LIGHTNING_LABEL = "Lightning answer: fast, not deeply verified."

# v2.6: contract-level forbidden patterns ALWAYS checked by the Lightning fast
# gate, regardless of what the intake emits. This makes the safety floor
# contract-driven instead of model-dependent. Includes overclaim patterns for
# life / digital-life tasks.
LIGHTNING_BASE_FORBIDDEN_PATTERNS: list[str] = [
    "TODO",
    "placeholder",
    "I can't",
    "creates real life",
    "creates biological life",
    "guaranteed consciousness",
    "proven sentience",
    "actually alive",
    "conscious digital being",
    "no ethical concerns",
]

# v2.6: words that, when they end an answer, strongly suggest truncation.
_LIGHTNING_TRUNCATION_TAIL_WORDS = (
    "the", "a", "an", "is", "are", "to", "not", "and", "or", "of", "in", "on",
    "for", "with", "that", "which", "as", "by", "at", "but",
)


def build_lightning_intake_prompt(task_text: str) -> str:
    """L0 Intake-lite: one fast call that routes the task and defines the
    minimal answer contract for the single one-shot answer."""
    return (
        "LIGHTNING INTAKE stage.\n"
        + output_mode_directive("MARKED_JSON") + "\n"
        "This is the fast Lightning pipeline: one intake, one answer, one gate. "
        "Classify the task and define the MINIMAL answer contract for a single "
        "fast answer: route (executable|non_executable|hybrid), answer_type "
        "(open_ended_design|code|factual|analysis|hybrid), required_sections "
        "(at most 4 short section titles, e.g. [\"Answer\"]), and "
        "forbidden_patterns (phrases the answer must NOT contain, e.g. "
        "[\"TODO\", \"placeholder\"]).\n"
        "For how-to / build / 'how do I create X' tasks, set "
        "actionability_requirements to the minimal practical coverage, e.g. "
        "[\"minimal_path\", \"success_check\", \"limitations\"] (empty for pure "
        "factual questions). Set needs_reasoning=true ONLY when the task needs "
        "logic, planning, or tradeoffs (otherwise false). Do not solve the task.\n\n"
        f"TASK:\n{task_text}\n\n"
        + build_schema_instruction(LightningIntake)
    )


def build_lightning_answer_prompt(
    task_text: str, intake: LightningIntake, reasoning: str = ""
) -> str:
    """L1 One-shot answer: one LLM call, plain prose, concise, required
    sections only, no overclaiming, exactly one final format.

    ``reasoning`` (optional) carries the bounded P2 reasoning bullets (<=3) so
    the one-shot answer can build on them without re-deriving them."""
    sections = (
        "\n".join(f"  - {s}" for s in intake.required_sections)
        if intake.required_sections
        else "  - Answer"
    )
    forbidden = ", ".join(intake.forbidden_patterns) if intake.forbidden_patterns else "(none)"
    actionability = ""
    if intake.actionability_requirements:
        reqs = ", ".join(intake.actionability_requirements)
        actionability = (
            f" This is a how-to/build task: cover the minimal practical path "
            f"({reqs}) — concrete steps, a success check, and limitations. Do "
            "not give a purely abstract answer.\n"
        )
    reasoning_block = (
        f"\nREASONING BULLETS (already worked out; build on these, do not repeat them verbatim):\n"
        f"{reasoning}\n"
        if reasoning.strip()
        else ""
    )
    return (
        "LIGHTNING ANSWER stage.\n"
        + output_mode_directive("PLAIN_FINAL") + "\n"
        "Produce ONE concise, self-contained answer in plain prose/markdown "
        "(no JSON, no markers, no meta-discussion about the output format). "
        "Include ONLY the required sections below. Do "
        f"not overclaim anything you have not verified. Forbidden patterns: {forbidden}."
        f"{actionability}\n\n"
        f"TASK:\n{task_text}\n\n"
        f"REQUIRED SECTIONS:\n{sections}\n"
        f"{reasoning_block}"
    )


def build_lightning_reasoning_prompt(task_text: str, intake: LightningIntake) -> str:
    """P2: bounded short-reasoning step for logic-heavy Lightning tasks.

    Produces AT MOST 3 short bullets (<=300 tokens) that capture the key logic,
    plan, or tradeoffs — NOT the final answer. The one-shot answer builds on it."""
    return (
        "LIGHTNING REASONING stage.\n"
        "This task needs a little reasoning before the answer. Write AT MOST 3 "
        "short bullets (each <= one line) capturing the key logic, plan, or "
        "tradeoffs. Do NOT write the final answer, no headings, no JSON, no "
        "markers. Keep it under ~300 tokens.\n\n"
        f"TASK:\n{task_text}\n"
    )


def build_lightning_repair_prompt(
    task_text: str, answer: str, issues: list[str]
) -> str:
    """L2 Fast-gate repair: the single allowed repair call when the rule-based
    gate fails. Repairs only the listed issues and returns the full answer."""
    issue_lines = "\n".join(f"  - {i}" for i in issues) or "  (none listed)"
    return (
        "LIGHTNING REPAIR stage.\n"
        + output_mode_directive("PLAIN_FINAL") + "\n"
        "The fast gate rejected the answer below. Repair ONLY the listed "
        "issues and return the FULL corrected answer in plain prose/markdown "
        "(no JSON, no markers), keeping everything else intact.\n\n"
        f"TASK:\n{task_text}\n\n"
        f"GATE ISSUES:\n{issue_lines}\n\n"
        f"CURRENT ANSWER:\n{answer}\n"
    )


# --------------------------------------------------------------------------- #
# Pairwise candidate selection
# --------------------------------------------------------------------------- #
def select_drafts_for_pairwise(
    drafts: list[Draft], evaluations: list[Evaluation], k: int = 3
) -> list[Draft]:
    """Select the top-k drafts (by verifier score) for a pairwise tournament."""
    score = {e.draft_id: e.score for e in evaluations}
    ranked = sorted(drafts, key=lambda d: score.get(d.id, 0.0), reverse=True)
    return ranked[: max(1, int(k))]
