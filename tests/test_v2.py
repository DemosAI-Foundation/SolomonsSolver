"""Non-live tests for the Solomon v2.0 verifier-first architecture.

Covers every mechanic (structured output, budgeting, retrieval, sandbox,
self-consistency, pairwise) plus an end-to-end run of the V2Pipeline through
Phases 0-7, including the executable/sandbox path and FINISH NOW / STOP.
"""
from __future__ import annotations

import json
import re
import time

import pytest

from config.settings import Settings
from src.agents.control import SessionRegistry
from src.agents.synthesis import SynthesisStep
from src.llm.orchestrator import LLMConfig, LLMOrchestrator, LLMResponse
from src.models.dataclasses import Task, new_task_id
from src.v2 import (
    ActiveContext,
    AnswerContract,
    AnswerType,
    BudgetProfile,
    BudgetState,
    ConfidenceLevel,
    CoverageReport,
    CritiqueIssue,
    CritiqueSeverity,
    DebateSynthesis,
    Difficulty,
    EmergencyFinish,
    Evaluation,
    ExecutionResult,
    FactsLedgerEntry,
    LengthPolicy,
    Plan,
    Route,
    SubProblemResult,
    TaskProfile,
    TokenBudget,
    V2Pipeline,
    VerificationReport,
    VerificationStatus,
    build_debate_synthesis_prompt,
    build_emergency_finish_prompt,
    build_facts_ledger,
    build_invalid_critique_reduced_prompt,
    build_invalid_critique_retry_prompt,
    build_phase8_coverage_prompt,
    build_phase8_repair_prompt,
    build_schema_instruction,
    budget_for_difficulty,
    budget_for_mode,
    cluster_answers,
    extract_final_json_block,
    mark_ungrounded,
    merge_budgets,
    normalize_answer,
    output_mode_directive,
    pairwise_judge,
    pairwise_tournament,
    parse_structured,
    self_consistency,
    sanity_check_tests,
)
from src.v2.contracts import BranchSet, Critique, CritiqueVerdict, DecompositionBranch, Draft, SubProblemCandidate
from src.v2.prompts import (
    OUTPUT_MODES,
    V22_SYSTEM_PROMPT,
    V24_SYSTEM_PROMPT,
    build_debate_synthesis_prompt,
    build_invalid_critique_reduced_prompt,
    build_invalid_critique_retry_prompt,
    build_phase0_prompt,
    build_phase1_prompt,
    build_phase2_prompt,
    build_phase3_code_prompt,
    build_phase3_solve_prompt,
    build_phase4_critique_prompt,
    build_phase5_draft_prompt,
    build_phase6_eval_prompt,
    build_phase6_pairwise_prompt,
    build_phase7_synthesis_prompt,
    build_phase7_verify_prompt,
    first_nonempty,
    output_mode_directive,
    phase_system_prompt,
    select_drafts_for_pairwise,
)
from src.v2.retrieval import RetrievalAdapter
from src.v2.sandbox import SandboxAdapter, SubprocessRunner, make_subprocess_runner
from tests.conftest import ScriptedTransport


def _wrap(obj) -> str:
    return f"FINAL_JSON_START\n{json.dumps(obj)}\nFINAL_JSON_END"


# --------------------------------------------------------------------------- #
# Deterministic v2.0 phase handler
# --------------------------------------------------------------------------- #
def v2_handler(prompt: str) -> str:
    """Route a v2.0 phase prompt to a schema-valid, marker-wrapped response."""
    # FINISH NOW synthesis (shared with the legacy synthesis prompt)
    if "Synthesize the single best final answer" in prompt:
        return "Combining the verified artifacts.\nANSWER: 42 (early synthesis)"

    # Phase 0 — intake / routing
    if "INTAKE & ROUTING stage" in prompt:
        route = "executable" if "EXEC_ROUTE" in prompt else "non_executable"
        return _wrap(
            {
                "route": route,
                "difficulty": "easy",
                "summary": "mock task profile",
                "key_requirements": ["exact answer"],
                "retrieval_available": False,
                "facts": [],
            }
        )

    # Phase 1 — planning
    if "PLANNING stage" in prompt:
        return _wrap(
            {
                "approach": "direct computation",
                "strategy": "single-shot",
                "key_requirements": ["correct"],
                "risks": ["none"],
                "verification_focus": ["recompute independently"],
            }
        )

    # Phase 2 — decomposition
    if "DECOMPOSITION stage" in prompt:
        sp_type = "executable" if "EXEC_ROUTE" in prompt else "non_executable"
        return _wrap(
            {
                "branches": [
                    {
                        "id": "b1",
                        "rationale": "single sub-problem",
                        "sub_problems": [
                            {
                                "id": "sp1",
                                "description": "solve the task",
                                "type": sp_type,
                                "depends_on": [],
                                "risk": "low",
                                "verification_hint": "recompute",
                            }
                        ],
                    }
                ]
            }
        )

    # Phase 3 — code (executable path)
    if "WRITING CODE to solve one executable sub-problem" in prompt:
        return _wrap(
            {
                "sub_problem_id": "sp1",
                "answer": "",
                "reasoning": "wrote code",
                "execution": {
                    "sub_problem_id": "sp1",
                    "code": "print(42)",
                    "output": "",
                    "tests_passed": False,
                    "verified": False,
                },
                "verified": False,
                "verification_method": "sandbox",
                "confidence": 0.5,
                "depends_on": [],
            }
        )

    # Phase 3 — solve (non-executable / self-consistency)
    if "SOLVING one sub-problem" in prompt:
        return _wrap(
            {
                "sub_problem_id": "sp1",
                "answer": "42",
                "reasoning": "computed independently",
                "verified": True,
                "verification_method": "recomputed",
                "confidence": 0.9,
                "depends_on": [],
            }
        )

    # Phase 4 — falsifiable critique
    if "VERIFICATION stage" in prompt:
        return _wrap(
            {
                "target": "sub-problem",
                "verdict": "pass",
                "falsifiable_claims": ["answer is 42"],
                "issues": [],
                "suggested_fixes": [],
                "confidence": 0.8,
            }
        )

    # Phase 5 — diverse drafts
    if "producing DRAFT" in prompt:
        m = re.search(r"DRAFT (\d+) of (\d+)", prompt)
        idx = int(m.group(1)) if m else 1
        preferred = " [PREFERRED]" if idx == 1 else ""
        return _wrap(
            {
                "id": f"d{idx}",
                "content": f"The answer is 42.{preferred}",
                "approach": f"approach {idx}",
                "source": "sp1",
                "diversity_note": "distinct approach",
            }
        )

    # Phase 6 — independent verification of a draft
    if "INDEPENDENT VERIFIER" in prompt:
        m = re.search(r"DRAFT (d\d+):", prompt)
        did = m.group(1) if m else "d1"
        score = 9.0 if "PREFERRED" in prompt else 5.0
        return _wrap(
            {
                "draft_id": did,
                "score": score,
                "verification_passed": True,
                "verification_notes": "recomputed and matched",
                "issues": [],
            }
        )

    # Phase 6 — order-swapped pairwise judgment
    if "neutral JUDGE" in prompt:
        a = prompt.split("ANSWER A:\n", 1)[1].split("\n\nANSWER B:", 1)[0]
        b = prompt.split("ANSWER B:\n", 1)[1].split("\n\n", 1)[0]
        if "PREFERRED" in a:
            winner = "A"
        elif "PREFERRED" in b:
            winner = "B"
        else:
            winner = "tie"
        return _wrap({"winner": winner, "reason": "mock judge"})

    # Phase 7 — final synthesis
    if "FINAL SYNTHESIS stage" in prompt:
        return _wrap(
            {
                "answer": "The final answer is 42.",
                "verification_passed": True,
                "verification_notes": "all sub-results verified",
                "confidence": 0.9,
                "ungrounded_claims": [],
            }
        )

    # Phase 7 — final re-verification
    if "FINAL VERIFIER" in prompt:
        return _wrap(
            {
                "target": "final_answer",
                "verdict": "pass",
                "falsifiable_claims": ["the answer is 42"],
                "issues": [],
                "suggested_fixes": [],
                "confidence": 0.9,
            }
        )

    return "ANSWER: default"


def _make_pipeline(handler, sandbox=None, retrieval=None):
    s = Settings()
    s.llm.max_concurrency = 16
    transport = ScriptedTransport(handler)
    llm = LLMOrchestrator(s.llm, transport=transport)
    sessions = SessionRegistry()
    synthesis = SynthesisStep(llm)
    pipeline = V2Pipeline(
        llm,
        None,  # event bus -> no-op emitter
        settings=s,
        sessions=sessions,
        synthesis=synthesis,
        retrieval=retrieval or RetrievalAdapter(),
        sandbox=sandbox or SandboxAdapter(),
    )
    return pipeline, sessions, transport


def _task(prompt: str = "What is 2+2?") -> Task:
    return Task(id=new_task_id(), original_prompt=prompt)


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
class TestContracts:
    def test_route_enum_values(self):
        assert Route.EXECUTABLE.value == "executable"
        assert Route.NON_EXECUTABLE.value == "non_executable"
        assert Route.HYBRID.value == "hybrid"

    def test_task_profile_defaults_and_facts_split(self):
        p = TaskProfile(
            facts=[
                FactsLedgerEntry(fact="a", grounded=True),
                FactsLedgerEntry(fact="b", grounded=False),
            ]
        )
        assert len(p.grounded_facts()) == 1
        assert len(p.ungrounded_facts()) == 1

    def test_confidence_clamped(self):
        r = SubProblemResult(sub_problem_id="sp1", confidence=5.0)
        assert r.confidence == 1.0
        r2 = SubProblemResult(sub_problem_id="sp1", confidence=-3.0)
        assert r2.confidence == 0.0

    def test_pairwise_winner_normalized(self):
        from src.v2.contracts import PairwiseJudgment

        assert PairwiseJudgment(winner="  a ").winner == "A"
        assert PairwiseJudgment(winner="maybe").winner == "tie"

    def test_evaluation_score_clamped(self):
        e = Evaluation(draft_id="d1", score=99.0)
        assert e.score == 10.0

    def test_extra_fields_ignored(self):
        p = TaskProfile.model_validate({"route": "executable", "unknown_key": 1})
        assert p.route == Route.EXECUTABLE


# --------------------------------------------------------------------------- #
# Mechanic C — structured output
# --------------------------------------------------------------------------- #
class TestStructured:
    def test_extract_with_markers(self):
        text = "blah\nFINAL_JSON_START\n{\"a\": 1}\nFINAL_JSON_END\ntail"
        assert extract_final_json_block(text) == '{"a": 1}'

    def test_extract_falls_back_without_markers(self):
        assert extract_final_json_block("here {\"x\": 2} ok") == '{"x": 2}'

    def test_extract_balanced_nested(self):
        text = 'FINAL_JSON_START\n{"a": {"b": [1, 2]}}\nFINAL_JSON_END'
        assert extract_final_json_block(text) == '{"a": {"b": [1, 2]}}'

    def test_extract_none(self):
        assert extract_final_json_block("") is None
        assert extract_final_json_block("no json here") is None

    def test_parse_success(self):
        obj, err = parse_structured(_wrap({"route": "executable"}), TaskProfile)
        assert err is None
        assert obj.route == Route.EXECUTABLE

    def test_parse_invalid_json(self):
        obj, err = parse_structured("FINAL_JSON_START\n{bad}\nFINAL_JSON_END", TaskProfile)
        assert obj is None
        assert "invalid JSON" in err

    def test_parse_validation_error(self):
        obj, err = parse_structured(_wrap({"route": "bogus"}), TaskProfile)
        assert obj is None
        assert "schema validation failed" in err

    def test_parse_no_json(self):
        obj, err = parse_structured("just words", TaskProfile)
        assert obj is None
        assert "no JSON object" in err

    def test_parse_not_object(self):
        # A bare list has no top-level object, so extraction finds nothing.
        obj, err = parse_structured("FINAL_JSON_START\n[1,2]\nFINAL_JSON_END", TaskProfile)
        assert obj is None
        assert "no JSON object" in err

    def test_schema_instruction_has_markers_and_fields(self):
        s = build_schema_instruction(TaskProfile)
        assert "FINAL_JSON_START" in s
        assert "FINAL_JSON_END" in s
        assert "route" in s


# --------------------------------------------------------------------------- #
# Mechanics A/B — budgeting
# --------------------------------------------------------------------------- #
class TestBudget:
    def test_estimate(self):
        tb = TokenBudget()
        assert tb.estimate("") == 0
        assert tb.estimate("abcd" * 100) == 100

    def test_plan_normal(self):
        tb = TokenBudget(window=1000, reserves={"phase0": 200})
        max_tokens, needs = tb.plan("x" * 100, "phase0")  # ~25 tokens
        assert needs is False
        assert max_tokens == 200

    def test_plan_respects_window(self):
        tb = TokenBudget(window=100, reserves={"phase0": 200})
        prompt = "x" * 300  # 75 tokens, leaving 25 for the answer
        max_tokens, needs = tb.plan(prompt, "phase0")
        assert needs is False
        assert max_tokens == 25
        assert max_tokens + tb.estimate(prompt) <= 100

    def test_plan_overflow(self):
        tb = TokenBudget(window=50, reserves={"phase0": 200})
        max_tokens, needs = tb.plan("x" * 1000, "phase0")
        assert needs is True
        assert max_tokens == 1

    def test_truncate(self):
        tb = TokenBudget(chars_per_token=4)
        assert tb.truncate("a" * 100, 10) == "a" * 40
        assert tb.truncate("short", 10) == "short"

    def test_set_reserves_merges(self):
        tb = TokenBudget(reserves={"phase0": 10})
        tb.set_reserves({"phase1": 20})
        assert tb.reserve_for("phase0") == 10
        assert tb.reserve_for("phase1") == 20

    def test_budget_for_difficulty_easy(self):
        b = budget_for_difficulty("easy")
        assert b.decomposition_branches == 1
        assert b.n_drafts == 2
        assert b.n_samples == 3
        assert b.n_repair_rounds == 1

    def test_budget_for_difficulty_hard(self):
        b = budget_for_difficulty(Difficulty.HARD)
        assert b.decomposition_branches == 2
        assert b.n_drafts == 5
        assert b.n_samples == 7

    def test_budget_profile_reserve_fallback(self):
        b = BudgetProfile(phase_reserves={"phase0": 123})
        assert b.reserve_for("phase0") == 123
        assert b.reserve_for("phaseX") == 0


# --------------------------------------------------------------------------- #
# ActiveContext
# --------------------------------------------------------------------------- #
class TestActiveContext:
    def _ctx(self) -> ActiveContext:
        ctx = ActiveContext(task_id="t1", task="what is 2+2?")
        ctx.task_profile = TaskProfile(
            route=Route.EXECUTABLE,
            difficulty=Difficulty.EASY,
            summary="add",
            key_requirements=["exact"],
            facts=[FactsLedgerEntry(fact="2+2=4", grounded=True, source="calc")],
        )
        ctx.plan = Plan(approach="add them", strategy="direct")
        # v2.6 honest verification: ``verified=True`` is only retained when the
        # result carries a real passing report (pass WITH evidence).
        ctx.sub_problem_results.append(
            SubProblemResult(
                sub_problem_id="sp1",
                answer="4",
                verified=True,
                confidence=0.9,
                verification=VerificationReport(
                    status=VerificationStatus.PASS,
                    evidence=["execution: 2+2=4"],
                ),
            )
        )
        return ctx

    def test_compact_text_includes_artifacts(self):
        text = self._ctx().compact_text()
        assert "what is 2+2?" in text
        assert "ROUTE: executable" in text
        assert "2+2=4" in text
        assert "sp1 (verified" in text

    def test_accessors(self):
        ctx = self._ctx()
        assert ctx.result_by_id("sp1").answer == "4"
        assert ctx.result_by_id("nope") is None
        assert ctx.sub_problem_ids() == ["sp1"]
        assert len(ctx.verified_sub_problems()) == 1

    def test_to_dict_roundtrip(self):
        d = self._ctx().to_dict()
        assert d["task_id"] == "t1"
        assert d["task_profile"]["route"] == "executable"
        assert d["sub_problem_results"][0]["answer"] == "4"

    def test_all_sub_problems_across_branches(self):
        ctx = ActiveContext(task_id="t", task="x")
        ctx.branches = [
            DecompositionBranch(
                id="b1",
                sub_problems=[
                    SubProblemCandidate(id="a", description="a"),
                    SubProblemCandidate(id="b", description="b"),
                ],
            )
        ]
        assert [sp.id for sp in ctx.all_sub_problems()] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Mechanic H — retrieval
# --------------------------------------------------------------------------- #
class TestRetrieval:
    async def test_unavailable(self):
        r = RetrievalAdapter()
        assert r.available is False
        avail, facts = await r.retrieve("q")
        assert avail is False and facts == []

    async def test_with_backend(self):
        r = RetrievalAdapter(backend=lambda q: ["fact one", "fact two"])
        assert r.available is True
        avail, facts = await r.retrieve("q")
        assert avail is True
        assert facts == ["fact one", "fact two"]

    async def test_async_backend(self):
        async def backend(q):
            return ["async fact"]

        r = RetrievalAdapter(backend=backend)
        avail, facts = await r.retrieve("q")
        assert avail is True and facts == ["async fact"]

    async def test_erroring_backend_degrades(self):
        def bad(q):
            raise RuntimeError("boom")

        r = RetrievalAdapter(backend=bad)
        avail, facts = await r.retrieve("q")
        assert avail is False and facts == []

    def test_build_facts_ledger(self):
        entries = build_facts_ledger(["a", "b"], True)
        assert len(entries) == 2
        assert all(e.grounded for e in entries)
        assert build_facts_ledger(["a"], False) == []

    def test_mark_ungrounded(self):
        entries = mark_ungrounded(["claim"])
        assert entries[0].grounded is False
        assert entries[0].source == "ungrounded"


# --------------------------------------------------------------------------- #
# Mechanics D/E — sandbox
# --------------------------------------------------------------------------- #
class TestSandbox:
    async def test_unavailable(self):
        s = SandboxAdapter()
        assert s.available is False
        res = await s.run("print(1)")
        assert res.verified is False
        assert res.sandbox_available is False

    async def test_with_runner(self):
        def runner(code, tests):
            return ExecutionResult(code=code, output="7", tests_passed=True, verified=True)

        s = SandboxAdapter(runner=runner)
        res = await s.run("print(7)", tests=[{"name": "t"}])
        assert res.verified is True
        assert res.output == "7"

    async def test_runner_dict_result(self):
        def runner(code, tests):
            return {"code": code, "output": "5", "verified": True, "tests_passed": True}

        s = SandboxAdapter(runner=runner)
        res = await s.run("print(5)")
        assert res.verified is True and res.output == "5"

    async def test_runner_exception_degrades(self):
        def runner(code, tests):
            raise RuntimeError("sandbox down")

        s = SandboxAdapter(runner=runner)
        res = await s.run("print(1)")
        assert res.verified is False
        assert res.sandbox_available is True

    def test_sanity_clean_no_tests(self):
        res = ExecutionResult(sub_problem_id="s", verified=True, tests_passed=True, output="ok")
        sane, issues = sanity_check_tests([], res)
        assert sane is True and issues == []

    def test_sanity_malformed_test(self):
        res = ExecutionResult(sub_problem_id="s", verified=False)
        sane, issues = sanity_check_tests(["not-a-dict"], res)
        assert sane is False
        assert any("not an object" in i for i in issues)

    def test_sanity_verified_without_evidence(self):
        res = ExecutionResult(sub_problem_id="s", verified=True, tests_passed=False)
        sane, issues = sanity_check_tests([], res)
        assert sane is False
        assert any("without execution evidence" in i for i in issues)

    def test_subprocess_runner_executes(self):
        runner = make_subprocess_runner(timeout=10.0)
        res = runner("print(6*7)", [])
        assert res.verified is True
        assert res.output.strip() == "42"


# --------------------------------------------------------------------------- #
# Mechanic G — self-consistency
# --------------------------------------------------------------------------- #
class TestSelfConsistency:
    def test_normalize_answer(self):
        assert normalize_answer("  42. ") == "42"
        assert normalize_answer("Hello, World!") == "hello, world"
        assert normalize_answer("") == ""

    def test_cluster_majority(self):
        rep, frac, count = cluster_answers(["42", "42", "43"])
        assert rep == "42"
        assert count == 2
        assert frac == pytest.approx(2 / 3)

    def test_cluster_empty(self):
        assert cluster_answers([]) == ("", 0.0, 0)

    async def test_self_consistency_majority(self):
        pipeline, _, _ = _make_pipeline(v2_handler)
        res = await self_consistency(
            pipeline.llm, "SOLVING one sub-problem mock", LLMConfig(temperature=0.6),
            "t", "phase3_solve", 3,
        )
        assert res.answer == "42"
        assert res.confidence == pytest.approx(1.0)
        assert res.total == 3


# --------------------------------------------------------------------------- #
# Mechanic F — pairwise
# --------------------------------------------------------------------------- #
def _pairwise_llm(handler) -> LLMOrchestrator:
    s = Settings()
    s.llm.max_concurrency = 16
    return LLMOrchestrator(s.llm, transport=ScriptedTransport(handler))


class TestPairwise:
    async def test_agreement_wins(self):
        llm = _pairwise_llm(v2_handler)
        j = await pairwise_judge(llm, "task", "PREFERRED answer", "other", "t", "pw")
        assert j.winner == "A"

    async def test_disagreement_is_tie(self):
        llm = _pairwise_llm(lambda p: _wrap({"winner": "A", "reason": "x"}))
        j = await pairwise_judge(llm, "task", "x", "y", "t", "pw")
        assert j.winner == "tie"

    async def test_tie_when_neither_preferred(self):
        llm = _pairwise_llm(v2_handler)
        j = await pairwise_judge(llm, "task", "x", "y", "t", "pw")
        assert j.winner == "tie"

    async def test_tournament_single(self):
        llm = _pairwise_llm(v2_handler)
        assert await pairwise_tournament(llm, "t", ["only"], "t", "pw") == 0

    async def test_tournament_definitive(self):
        llm = _pairwise_llm(v2_handler)
        idx = await pairwise_tournament(llm, "t", ["PREFERRED", "b", "c"], "t", "pw")
        assert idx == 0

    async def test_tournament_all_ties(self):
        llm = _pairwise_llm(lambda p: _wrap({"winner": "tie", "reason": "x"}))
        assert await pairwise_tournament(llm, "t", ["a", "b"], "t", "pw") == -1


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
class TestPrompts:
    def test_phase0_prompt_has_task_and_markers(self):
        p = build_phase0_prompt("do X", False, [])
        assert "do X" in p
        assert "FINAL_JSON_START" in p
        assert "no retrieval results available" in p

    def test_phase0_prompt_with_facts(self):
        p = build_phase0_prompt("do X", True, ["f1", "f2"])
        assert "f1" in p and "f2" in p
        assert "RETRIEVAL AVAILABLE: yes" in p

    def test_all_phase_prompts_carry_markers(self):
        ctx = ActiveContext(task_id="t", task="x")
        ctx.task_profile = TaskProfile()
        ctx.branches = [
            DecompositionBranch(
                id="b1", sub_problems=[SubProblemCandidate(id="sp1", description="d")]
            )
        ]
        sp = ctx.all_sub_problems()[0]
        draft = Draft(id="d1", content="c")
        prompts = [
            build_phase1_prompt(ctx),
            build_phase2_prompt(ctx, 1),
            build_phase3_solve_prompt(ctx, sp),
            build_phase3_code_prompt(ctx, sp),
            build_phase4_critique_prompt(ctx, "target", "label"),
            build_phase5_draft_prompt(ctx, 0, 2, []),
            build_phase6_eval_prompt("task", draft),
            build_phase6_pairwise_prompt("task", "a", "b"),
            build_phase7_synthesis_prompt(ctx),
            build_phase7_verify_prompt("task", "ans"),
        ]
        for p in prompts:
            assert "FINAL_JSON_START" in p

    def test_select_drafts_for_pairwise(self):
        drafts = [Draft(id="d1", content="a"), Draft(id="d2", content="b"), Draft(id="d3", content="c")]
        evals = [
            Evaluation(draft_id="d1", score=5.0),
            Evaluation(draft_id="d2", score=9.0),
            Evaluation(draft_id="d3", score=7.0),
        ]
        top = select_drafts_for_pairwise(drafts, evals, 2)
        assert [d.id for d in top] == ["d2", "d3"]

    def test_first_nonempty(self):
        assert first_nonempty("", "  ", "x", "y") == "x"
        assert first_nonempty("", None) == ""


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
class TestPipelineEndToEnd:
    async def test_non_executable_full_run(self):
        pipeline, sessions, _ = _make_pipeline(v2_handler)
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert result.tier == "v2"
        assert "42" in result.final_answer
        assert result.confidence > 0
        assert "route_non_executable" in result.techniques_used
        assert "final_synthesis" in result.techniques_used
        assert result.completion_pct == 1.0
        # session cleaned up after the run
        assert sessions.get(result.task_id) is None

    async def test_run_holds_lock_during_execution(self):
        # The pipeline keeps per-run state (mode + token budget) on the shared
        # instance, so runs are serialized: the lock must be held for the whole
        # run and released afterwards.
        pipeline, _, _ = _make_pipeline(v2_handler)
        observed: dict = {}

        def handler(prompt: str) -> str:
            observed["locked_during"] = pipeline._run_lock.locked()
            return v2_handler(prompt)

        pipeline.llm.transport = ScriptedTransport(handler)
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert observed.get("locked_during") is True
        assert pipeline._run_lock.locked() is False

    async def test_executable_path_uses_sandbox(self):
        def runner(code, tests):
            return ExecutionResult(
                code=code, stdout="42", output="42",
                tests_passed=True, test_results=[{"name": "run", "passed": True}],
                verified=True, errors=[],
            )

        pipeline, _, _ = _make_pipeline(v2_handler, sandbox=SandboxAdapter(runner=runner))
        result = await pipeline.run(_task("EXEC_ROUTE compute 6*7"))
        assert result.failed is False
        assert "route_executable" in result.techniques_used
        assert "42" in result.final_answer

    async def test_retrieval_grounds_facts(self):
        pipeline, _, _ = _make_pipeline(
            v2_handler, retrieval=RetrievalAdapter(backend=lambda q: ["grounded fact"])
        )
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "42" in result.final_answer

    async def test_fallbacks_on_garbage(self):
        pipeline, _, _ = _make_pipeline(lambda p: "I refuse to emit JSON.")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert result.final_answer
        assert any("fallback" in t for t in result.techniques_used)

    async def test_finish_now_early_synthesis(self):
        task = _task("What is 2+2?")

        def handler(prompt):
            if "INTAKE & ROUTING stage" in prompt:
                sess = sessions.get(task.id)
                if sess is not None:
                    sess.request_finish_now()
            return v2_handler(prompt)

        pipeline, sessions, _ = _make_pipeline(handler)
        result = await pipeline.run(task)
        assert result.early_completion is True
        assert result.aborted is False
        assert "finish_now_synthesis" in result.techniques_used
        assert "early synthesis" in result.final_answer

    async def test_stop_aborts(self):
        task = _task("What is 2+2?")

        def handler(prompt):
            if "INTAKE & ROUTING stage" in prompt:
                sess = sessions.get(task.id)
                if sess is not None:
                    sess.request_stop()
            return v2_handler(prompt)

        pipeline, sessions, _ = _make_pipeline(handler)
        result = await pipeline.run(task)
        assert result.aborted is True
        assert result.early_completion is False


# --------------------------------------------------------------------------- #
# v2.2 — answer contract, budget governor, coverage gate, emergency finish
# --------------------------------------------------------------------------- #
def _make_pipeline_mode(handler, mode: str, sandbox=None, retrieval=None):
    s = Settings()
    s.llm.max_concurrency = 16
    transport = ScriptedTransport(handler)
    llm = LLMOrchestrator(s.llm, transport=transport)
    sessions = SessionRegistry()
    synthesis = SynthesisStep(llm)
    pipeline = V2Pipeline(
        llm,
        None,
        settings=s,
        sessions=sessions,
        synthesis=synthesis,
        retrieval=retrieval or RetrievalAdapter(),
        sandbox=sandbox or SandboxAdapter(),
        mode=mode,
    )
    return pipeline, sessions, transport


def _coverage(passed: bool, **kw) -> str:
    base = {
        "passed": passed,
        "missing_sections": kw.get("missing_sections", []),
        "weak_sections": kw.get("weak_sections", []),
        "scope_issues": kw.get("scope_issues", []),
        "forbidden_patterns_detected": kw.get("forbidden_patterns_detected", []),
        "repair_instructions": kw.get("repair_instructions", ""),
    }
    return _wrap(base)


def _v22_handler(prompt: str) -> str:
    """v2_handler extended with v2.2 markers (answer contract, coverage gate)."""
    if "INTAKE & ROUTING stage" in prompt:
        route = "executable" if "EXEC_ROUTE" in prompt else "non_executable"
        return _wrap(
            {
                "route": route,
                "difficulty": "easy",
                "summary": "mock task profile",
                "key_requirements": ["exact answer"],
                "retrieval_available": False,
                "facts": [],
                "answer_contract": {
                    "answer_type": "factual",
                    "length_policy": "concise",
                    "required_sections": ["Answer"],
                    "must_include_subproblem_ids": [],
                    "forbidden_patterns": ["TODO"],
                    "min_completeness_score": 0.5,
                },
            }
        )
    if "COVERAGE VALIDATION stage" in prompt:
        return _coverage(True)
    return v2_handler(prompt)


class TestV22Contracts:
    def test_answer_contract_fields(self):
        ac = AnswerContract(
            answer_type="open_ended_design",
            length_policy="comprehensive",
            required_sections=["Summary", "Design"],
            must_include_subproblem_ids=["sp1"],
            forbidden_patterns=["TODO"],
            min_completeness_score=0.8,
        )
        assert ac.answer_type == AnswerType.OPEN_ENDED_DESIGN
        assert ac.length_policy == LengthPolicy.COMPREHENSIVE
        assert ac.required_sections == ["Summary", "Design"]
        assert ac.forbidden_patterns == ["TODO"]
        assert ac.min_completeness_score == 0.8

    def test_coverage_report_defaults(self):
        cr = CoverageReport(passed=False, missing_sections=["Design"])
        assert cr.passed is False
        assert cr.missing_sections == ["Design"]
        assert cr.weak_sections == []
        assert cr.forbidden_patterns_detected == []

    def test_emergency_finish_confidence(self):
        ef = EmergencyFinish(final_answer="partial", confidence="medium")
        assert ef.confidence == ConfidenceLevel.MEDIUM
        assert ef.completed_parts == []
        assert ef.incomplete_parts == []

    def test_budget_state(self):
        bs = BudgetState(
            mode="quick", used_tokens=100, remaining_tokens=900,
            final_reserve_tokens=2000, partial_answer_allowed=True,
        )
        assert bs.remaining_tokens == 900
        assert bs.partial_answer_allowed is True


class TestV22BudgetGovernor:
    def test_budget_for_mode_values(self):
        quick = budget_for_mode("quick")
        std = budget_for_mode("standard")
        deep = budget_for_mode("deep")
        assert quick.mode == "quick" and quick.final_reserve_tokens == 2000
        assert std.mode == "standard" and std.final_reserve_tokens == 6000
        assert deep.mode == "deep" and deep.final_reserve_tokens == 12000
        assert deep.n_drafts >= std.n_drafts >= quick.n_drafts

    def test_budget_for_mode_unknown_falls_back(self):
        assert budget_for_mode("bogus").mode == "standard"

    def test_merge_preserves_governor_reserves(self):
        merged = merge_budgets(budget_for_mode("quick"), budget_for_difficulty(Difficulty.HARD))
        # numeric fields take the larger of the two
        assert merged.n_drafts == 5
        assert merged.n_samples == 7
        # governor reserves come from the base (mode) budget, not difficulty
        assert merged.mode == "quick"
        assert merged.final_reserve_tokens == 2000
        assert merged.final_repair_reserve_tokens == 1000
        assert merged.pairwise_mode == "top3"  # HARD escalates pairwise mode

    def test_soft_cap_stops_intermediate(self):
        tb = TokenBudget(window=10000, profile=budget_for_mode("quick"))
        assert tb.should_stop_intermediate() is False
        tb.record_intermediate(7000)  # > 60% of 10000 (soft cap = 6000)
        assert tb.should_stop_intermediate() is True
        assert tb.can_afford_intermediate(500) is False
        # final-answer budget is independent of the intermediate cap (used stays 0)
        assert tb.can_afford(2000) is True

    def test_truncation_tracking(self):
        tb = TokenBudget(window=1000, profile=budget_for_mode("standard"))
        tb.record_truncation("phase7 length")
        state = tb.state()
        assert state.truncations == 1
        assert state.partial_answer_allowed is True
        assert "truncation: phase7 length" in tb.warnings


class TestV22Prompts:
    def test_v22_system_prompt(self):
        assert "final answer" in V22_SYSTEM_PROMPT.lower()

    def test_phase8_coverage_prompt_markers(self):
        task = Task(id="t", original_prompt="How could I create digital life?")
        contract = AnswerContract(
            required_sections=["Summary", "Design"], forbidden_patterns=["TODO"]
        )
        p = build_phase8_coverage_prompt(task, "The final answer is 42.", contract)
        assert "COVERAGE VALIDATION stage" in p
        assert "Summary" in p  # required sections surfaced
        assert "TODO" in p  # forbidden patterns surfaced
        assert "FINAL_JSON_START" in p

    def test_phase8_repair_prompt_markers(self):
        task = Task(id="t", original_prompt="x")
        contract = AnswerContract(required_sections=["Design"])
        coverage = CoverageReport(passed=False, missing_sections=["Design"],
                                  repair_instructions="add a design section")
        p = build_phase8_repair_prompt(task, "old answer", contract, coverage)
        assert "SECTION REPAIR stage" in p
        assert "Design" in p

    def test_emergency_finish_prompt_marker(self):
        task = Task(id="t", original_prompt="x")
        ctx = ActiveContext(task_id="t", task="x")
        p = build_emergency_finish_prompt(task.original_prompt, ctx)
        assert "EMERGENCY FINISH NOW." in p
        assert "FINAL_JSON_START" in p


class TestV22Pipeline:
    async def test_full_run_with_answer_contract(self):
        pipeline, _, _ = _make_pipeline_mode(_v22_handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert result.tier == "v2"
        assert "42" in result.final_answer
        assert "coverage_gate" in result.techniques_used
        assert result.completion_pct == 1.0

    async def test_mode_is_recorded(self):
        pipeline, _, _ = _make_pipeline_mode(_v22_handler, "deep")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "mode_deep" in result.techniques_used

    async def test_phase8_coverage_gate_triggers_repair(self):
        calls = {"n": 0}

        def handler(prompt):
            if "COVERAGE VALIDATION stage" in prompt:
                calls["n"] += 1
                if calls["n"] == 1:
                    return _coverage(False, missing_sections=["Design"],
                                     repair_instructions="add a design section")
                return _coverage(True)
            if "SECTION REPAIR stage" in prompt:
                return _wrap({
                    "answer": "The final answer is 42. Design: a cellular automaton.",
                    "verification_passed": True, "verification_notes": "repaired",
                    "confidence": 0.9,
                })
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "coverage_repair_1" in result.techniques_used
        assert "coverage_gate" in result.techniques_used
        assert "Design" in result.final_answer

    async def test_phase8_forbidden_pattern_fails_gate(self):
        def handler(prompt):
            if "COVERAGE VALIDATION stage" in prompt:
                return _coverage(True)
            if "FINAL SYNTHESIS stage" in prompt:
                return _wrap({
                    "answer": "The final answer is 42. TODO: expand later.",
                    "verification_passed": True, "verification_notes": "",
                    "confidence": 0.9,
                })
            return _v22_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        # the deterministic forbidden-pattern check forces the gate to fail
        assert "coverage_gate_failed" in result.techniques_used

    async def test_emergency_finish_schema(self):
        task = _task("What is 2+2?")

        def handler(prompt):
            if "INTAKE & ROUTING stage" in prompt:
                sess = sessions.get(task.id)
                if sess is not None:
                    sess.request_finish_now()
            if "EMERGENCY FINISH NOW." in prompt:
                return _wrap({
                    "final_answer": "42 (emergency finish)",
                    "confidence": "high",
                    "completed_parts": ["solve"],
                    "incomplete_parts": [],
                    "verified_parts": ["solve"],
                    "unverified_claims": [],
                })
            return v2_handler(prompt)

        pipeline, sessions, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(task)
        assert result.early_completion is True
        assert result.aborted is False
        assert "42 (emergency finish)" in result.final_answer
        assert result.synthesis.get("method") == "emergency_finish"
        assert result.synthesis.get("confidence") == "high"

    async def test_self_refine_vague_revise_downgraded(self):
        def handler(prompt):
            if "VERIFICATION stage" in prompt:
                # vague revise: no missing_sections / broken_steps / suggested_fixes
                return _wrap({
                    "target": "sub-problem", "verdict": "revise",
                    "falsifiable_claims": [], "issues": ["could be better"],
                    "suggested_fixes": [], "confidence": 0.5,
                })
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "self_refine_downgrade" in result.techniques_used

    async def test_truncation_recovery_salvages_partial(self):
        def handler(prompt):
            if "FINAL SYNTHESIS stage" in prompt:
                return LLMResponse(
                    text='{"answer": "The final answer is 42. But it got trunc',
                    trace="mock", prompt_tokens=10, completion_tokens=20,
                    latency=0.01, finish_reason="length",
                )
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "truncation_salvage" in result.techniques_used
        assert "42" in result.final_answer


# --------------------------------------------------------------------------- #
# v2.4 — contracts
# --------------------------------------------------------------------------- #
class TestV24Contracts:
    def test_verification_status_values(self):
        assert VerificationStatus.PASS.value == "pass"
        assert VerificationStatus.FAIL.value == "fail"
        assert VerificationStatus.PARTIAL.value == "partial"
        assert VerificationStatus.UNVERIFIED.value == "unverified"

    def test_verification_report_schema(self):
        v = VerificationReport(
            status=VerificationStatus.PASS, evidence=["stdout: 42"], missing=[]
        )
        assert v.is_pass is True
        assert v.evidence == ["stdout: 42"]

    def test_verification_report_defaults_unverified(self):
        v = VerificationReport()
        assert v.status == VerificationStatus.UNVERIFIED
        assert v.is_pass is False

    def test_subproblem_result_carries_verification(self):
        v = VerificationReport(status=VerificationStatus.PASS, evidence=["ok"])
        r = SubProblemResult(sub_problem_id="sp1", verification=v)
        assert r.verification is v
        assert r.verification.status == VerificationStatus.PASS

    def test_critique_issue_schema(self):
        i = CritiqueIssue(
            issue_id="i1",
            severity=CritiqueSeverity.CRITICAL,
            description="wrong answer",
            evidence="42 != 4",
            repair_action="recompute",
        )
        assert i.severity == CritiqueSeverity.CRITICAL
        assert i.repair_action == "recompute"

    def test_critique_structured_issues_ranked(self):
        c = Critique(
            target="sp1",
            verdict=CritiqueVerdict.REVISE,
            critical_issues=[
                CritiqueIssue(issue_id="c", severity=CritiqueSeverity.CRITICAL,
                              description="x", repair_action="fix x")
            ],
            minor_issues=[
                CritiqueIssue(issue_id="m", severity=CritiqueSeverity.MINOR,
                              description="y", repair_action="fix y")
            ],
        )
        assert [i.issue_id for i in c.structured_issues()] == ["c", "m"]
        assert "fix x" in c.actionable_targets()
        assert "fix y" in c.actionable_targets()

    def test_debate_synthesis_to_critique(self):
        ds = DebateSynthesis(
            verdict=CritiqueVerdict.REVISE,
            confidence=0.8,
            critical_issues=[
                CritiqueIssue(issue_id="c", severity=CritiqueSeverity.CRITICAL,
                              description="x", repair_action="fix x")
            ],
            missing_sections=["Design"],
            repair_plan=["recompute"],
        )
        c = ds.to_critique("sp1")
        assert c.target == "sp1"
        assert c.verdict == CritiqueVerdict.REVISE
        assert c.missing_sections == ["Design"]
        assert "fix x" in c.actionable_targets()

    def test_coverage_report_v24_aliases_sync(self):
        cr = CoverageReport(
            coverage_pass=False, forbidden_patterns=["TODO"],
            verification_status="partial",
        )
        assert cr.passed is False
        assert cr.forbidden_patterns_detected == ["TODO"]
        assert cr.verification_status == "partial"
        assert cr.coverage_ok is False

    def test_coverage_report_v24_pass(self):
        cr = CoverageReport(coverage_pass=True)
        assert cr.passed is True
        assert cr.coverage_ok is True

    def test_coverage_report_bad_verification_status(self):
        cr = CoverageReport(verification_status="bogus")
        assert cr.verification_status == "partial"


# --------------------------------------------------------------------------- #
# v2.4 — prompts (no persona, one output mode, new phase prompts)
# --------------------------------------------------------------------------- #
class TestV24Prompts:
    def test_v24_system_prompt_no_persona(self):
        assert "you are" not in V24_SYSTEM_PROMPT.lower()

    def test_v22_alias_is_v24(self):
        assert V22_SYSTEM_PROMPT is V24_SYSTEM_PROMPT

    def test_output_modes_defined(self):
        # v2.6 (Bug 5): MARKED_JSON is a distinct output mode (one JSON object
        # wrapped in FINAL_JSON_START / FINAL_JSON_END), never mixed with the
        # others.
        assert set(OUTPUT_MODES) == {
            "JSON_ONLY", "MARKED_JSON", "PLAIN_FINAL", "SECTION_JSON", "ANSWER_LINE"
        }

    def test_output_mode_directive(self):
        assert "JSON_ONLY" in output_mode_directive("JSON_ONLY")
        assert "PLAIN_FINAL" in output_mode_directive("PLAIN_FINAL")

    def test_all_phase_prompts_have_output_mode(self):
        ctx = ActiveContext(task_id="t", task="x")
        sp = SubProblemCandidate(id="sp1", description="solve")
        draft = Draft(id="d1", content="The answer is 42.")
        contract = AnswerContract(required_sections=["Answer"])
        prompts = [
            build_phase0_prompt("What is 2+2?", False),
            build_phase1_prompt(ctx),
            build_phase2_prompt(ctx, 1),
            build_phase3_solve_prompt(ctx, sp),
            build_phase3_code_prompt(ctx, sp),
            build_phase4_critique_prompt(ctx, "Answer: 42", "sub-problem sp1"),
            build_phase5_draft_prompt(ctx, 0, 2, []),
            build_phase6_eval_prompt("What is 2+2?", draft),
            build_phase7_synthesis_prompt(ctx),
            build_phase7_verify_prompt("What is 2+2?", "The answer is 42."),
            build_phase8_coverage_prompt("What is 2+2?", "The answer is 42.", contract),
        ]
        for p in prompts:
            assert "OUTPUT MODE:" in p, p[:80]
            assert "FINAL_JSON_START" in p, p[:80]

    def test_debate_synthesis_prompt_markers(self):
        ctx = ActiveContext(task_id="t", task="x")
        p = build_debate_synthesis_prompt(
            ctx, ["[critical] answer wrong (repair: recompute)"]
        )
        assert "DEBATE SYNTHESIS stage" in p
        assert "OUTPUT MODE:" in p
        assert "FINAL_JSON_START" in p
        assert "answer wrong" in p

    def test_invalid_critique_retry_prompt_markers(self):
        ctx = ActiveContext(task_id="t", task="x")
        p = build_invalid_critique_retry_prompt(
            ctx, "Answer: 42", "sub-problem sp1", "invalid JSON"
        )
        assert "VERIFICATION stage (retry)" in p
        assert "invalid JSON" in p
        assert "FINAL_JSON_START" in p

    def test_invalid_critique_reduced_prompt_markers(self):
        ctx = ActiveContext(task_id="t", task="x")
        p = build_invalid_critique_reduced_prompt(ctx, "Answer: 42", "sub-problem sp1")
        assert "VERIFICATION stage (reduced scope)" in p
        assert "FINAL_JSON_START" in p

    def test_phase8_repair_prompt_structured_targets(self):
        contract = AnswerContract(required_sections=["Design"])
        coverage = CoverageReport(passed=False, missing_sections=["Design"])
        ds = DebateSynthesis(
            verdict=CritiqueVerdict.REVISE,
            critical_issues=[
                CritiqueIssue(issue_id="c", severity=CritiqueSeverity.CRITICAL,
                              description="x", repair_action="add a design section")
            ],
            repair_plan=["add a design section"],
        )
        p = build_phase8_repair_prompt(
            "x", "old", contract, coverage, debate_synthesis=ds
        )
        assert "SECTION REPAIR stage" in p
        assert "STRUCTURED REPAIR TARGETS" in p
        assert "add a design section" in p

    def test_emergency_finish_prompt_strategy(self):
        ctx = ActiveContext(task_id="t", task="x")
        p = build_emergency_finish_prompt("x", ctx, strategy="synthesize_verified")
        assert "EMERGENCY FINISH NOW." in p
        assert "FINISH STRATEGY:" in p
        assert "synthesize_verified" in p


# --------------------------------------------------------------------------- #
# v2.4 — sandbox kill (dynamic FINISH NOW)
# --------------------------------------------------------------------------- #
class TestV24Sandbox:
    def test_subprocess_runner_executes_and_killable(self):
        runner = SubprocessRunner(timeout=5.0)
        res = runner("print(1+1)", [])
        assert res.verified is True
        assert res.output == "2"
        # No in-flight process -> kill() is a no-op returning False.
        assert runner.kill() is False

    def test_subprocess_runner_kills_inflight(self):
        import threading

        runner = SubprocessRunner(timeout=10.0)

        def run_slow():
            runner("import time; time.sleep(30)", [])

        t = threading.Thread(target=run_slow, daemon=True)
        t.start()
        deadline = time.monotonic() + 5.0
        while not runner.running and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runner.running, "subprocess did not start in time"
        assert runner.kill() is True
        t.join(timeout=5.0)
        assert not t.is_alive()

    def test_sandbox_adapter_kill_delegates(self):
        class FakeRunner:
            killed = False

            def __call__(self, code, tests):
                return ExecutionResult(code=code, output="ok", verified=True)

            def kill(self):
                FakeRunner.killed = True
                return True

        adapter = SandboxAdapter(runner=FakeRunner())
        assert adapter.kill() is True
        assert FakeRunner.killed is True
        # A plain callable without kill() -> no-op False.
        adapter2 = SandboxAdapter(runner=lambda code, tests: ExecutionResult(code=code))
        assert adapter2.kill() is False


# --------------------------------------------------------------------------- #
# v2.4 — pipeline (verification, debate synthesis, invalid critique, FINISH NOW)
# --------------------------------------------------------------------------- #
class TestV24Pipeline:
    async def test_verification_report_attached(self):
        pipeline, _, _ = _make_pipeline_mode(v2_handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        ctx = pipeline.active_context
        assert ctx is not None
        assert len(ctx.sub_problem_results) == 1
        ver = ctx.sub_problem_results[0].verification
        assert ver is not None
        assert ver.status == VerificationStatus.PASS
        assert ver.evidence

    async def test_sandbox_verification_report(self):
        def runner(code, tests):
            return ExecutionResult(
                code=code, stdout="42", output="42",
                tests_passed=True, test_results=[{"name": "run", "passed": True}],
                verified=True, errors=[],
            )

        pipeline, _, _ = _make_pipeline_mode(
            v2_handler, "standard", sandbox=SandboxAdapter(runner=runner)
        )
        result = await pipeline.run(_task("EXEC_ROUTE compute 6*7"))
        assert result.failed is False
        ctx = pipeline.active_context
        ver = ctx.sub_problem_results[0].verification
        assert ver is not None
        assert ver.status == VerificationStatus.PASS
        assert any("stdout" in e for e in ver.evidence)

    async def test_debate_synthesis_before_repair(self):
        flags = {"debate": False}

        def handler(prompt):
            if "DEBATE SYNTHESIS stage" in prompt:
                flags["debate"] = True
                return _wrap({
                    "verdict": "revise", "confidence": 0.8,
                    "critical_issues": [{
                        "issue_id": "i1", "severity": "critical",
                        "description": "answer is wrong", "evidence": "42 != 4",
                        "repair_action": "recompute the sum",
                    }],
                    "minor_issues": [], "missing_sections": [],
                    "repair_plan": ["recompute the sum"],
                })
            if "VERIFICATION stage" in prompt:
                return _wrap({
                    "target": "sub-problem", "verdict": "revise",
                    "falsifiable_claims": [], "issues": [], "suggested_fixes": [],
                    "confidence": 0.7,
                    "critical_issues": [{
                        "issue_id": "c1", "severity": "critical",
                        "description": "answer is wrong", "evidence": "42 != 4",
                        "repair_action": "recompute the sum",
                    }],
                    "minor_issues": [], "repair_plan": ["recompute the sum"],
                })
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert flags["debate"] is True
        assert result.failed is False
        assert "debate_synthesis" in result.techniques_used
        assert pipeline.active_context.debate_synthesis is not None
        assert "recompute the sum" in pipeline.active_context.debate_synthesis.repair_plan

    async def test_invalid_critique_fallback(self):
        flags = {"retry": False, "reduced": False}

        def handler(prompt):
            if "VERIFICATION stage (retry)" in prompt:
                flags["retry"] = True
            if "VERIFICATION stage (reduced scope)" in prompt:
                flags["reduced"] = True
            if "VERIFICATION stage" in prompt:
                return "I think the answer is a bit off, hard to say."
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert flags["retry"] is True
        assert flags["reduced"] is True
        assert "42" in result.final_answer

    async def test_finish_now_interrupts_mid_phase(self):
        task = _task("What is 2+2?")
        calls = {"drafts": 0}

        def handler(prompt):
            if "producing DRAFT" in prompt:
                calls["drafts"] += 1
                if calls["drafts"] == 1:
                    sess = sessions.get(task.id)
                    if sess is not None:
                        sess.request_finish_now()
                return _wrap({
                    "id": f"d{calls['drafts']}",
                    "content": "The answer is 42.",
                    "approach": f"approach {calls['drafts']}",
                    "source": "sp1",
                    "diversity_note": "distinct approach",
                })
            if "EMERGENCY FINISH NOW." in prompt:
                return _wrap({
                    "final_answer": "42 (mid-phase finish)",
                    "confidence": "high",
                    "completed_parts": ["solve"],
                    "incomplete_parts": ["drafts"],
                    "verified_parts": ["solve"],
                    "unverified_claims": [],
                })
            return v2_handler(prompt)

        pipeline, sessions, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(task)
        assert result.early_completion is True
        assert result.aborted is False
        assert "42 (mid-phase finish)" in result.final_answer
        assert result.synthesis.get("method") == "emergency_finish"


# --------------------------------------------------------------------------- #
# v2.5 — adaptive per-phase system prompts + contract deltas
# --------------------------------------------------------------------------- #
class TestV25:
    # -- Rule 1: adaptive per-phase system prompts ------------------------- #
    def test_phase_system_prompt_has_phase_rules(self):
        for phase, name in (
            ("phase0", "Phase: 0 Intake"),
            ("phase1", "Phase: 1 Planning"),
            ("phase2", "Phase: 2 Decomposition"),
            ("phase3", "Phase: 3 Sub-problem solving"),
            ("phase4", "Phase: 4 Debate synthesis"),
            ("phase5", "Phase: 5 Drafting"),
            ("phase6", "Phase: 6 Selection"),
            ("phase7", "Phase: 7 Final synthesis"),
            ("phase7_verify", "Phase: 7 Final verification"),
            ("phase8", "Phase: 8 Coverage gate"),
            ("emergency_finish", "Phase: Emergency Finish"),
        ):
            sp = phase_system_prompt(phase)
            assert name in sp, (phase, sp[:80])
            assert "Stance:" in sp, phase
            assert "Constraints:" in sp, phase
            assert "Output mode:" in sp, phase

    def test_phase_system_prompt_no_persona(self):
        for phase in (
            "phase0", "phase1", "phase2", "phase3", "phase4", "phase5",
            "phase6", "phase7", "phase7_verify", "phase8", "emergency_finish",
        ):
            assert "you are" not in phase_system_prompt(phase).lower(), phase

    def test_phase_system_prompt_normalization(self):
        # Granular pipeline labels collapse to the same canonical prompt.
        assert phase_system_prompt("phase3_code") == phase_system_prompt("phase3_solve")
        assert phase_system_prompt("phase3_code") == phase_system_prompt("phase3")
        assert phase_system_prompt("phase6_eval") == phase_system_prompt("phase6")
        assert phase_system_prompt("phase8_repair") == phase_system_prompt("phase8")
        assert phase_system_prompt("phase8_coverage") == phase_system_prompt("phase8")
        assert phase_system_prompt("finish_now") == phase_system_prompt("emergency_finish")
        # phase7_verify is a distinct phase from phase7.
        assert phase_system_prompt("phase7_verify") != phase_system_prompt("phase7")

    def test_phase_system_prompt_fallback_unknown(self):
        assert phase_system_prompt("nonsense") == V24_SYSTEM_PROMPT
        assert phase_system_prompt("") == V24_SYSTEM_PROMPT

    async def test_pipeline_uses_adaptive_system_prompt(self):
        pipeline, _, transport = _make_pipeline_mode(v2_handler, "standard")
        await pipeline.run(_task("What is 2+2?"))
        expected = (
            ("INTAKE & ROUTING stage", "Phase: 0 Intake"),
            ("PLANNING stage", "Phase: 1 Planning"),
            ("DECOMPOSITION stage", "Phase: 2 Decomposition"),
            ("SOLVING one sub-problem", "Phase: 3 Sub-problem solving"),
            ("producing DRAFT", "Phase: 5 Drafting"),
            ("INDEPENDENT VERIFIER", "Phase: 6 Selection"),
            ("FINAL SYNTHESIS stage", "Phase: 7 Final synthesis"),
            ("COVERAGE VALIDATION stage", "Phase: 8 Coverage gate"),
        )
        seen = {}
        for call in transport.calls:
            sysmsg = call["messages"][0]["content"]
            prompt = call["prompt"]
            for marker, name in expected:
                if marker in prompt and marker not in seen:
                    seen[marker] = (sysmsg, name)
        for marker, (sysmsg, name) in seen.items():
            assert name in sysmsg, (marker, sysmsg[:80])
        for marker, _name in expected:
            assert marker in seen, f"no LLM call captured for {marker!r}"

    # -- Phase 1 Plan contract --------------------------------------------- #
    def test_plan_has_v25_fields(self):
        p = Plan(
            approach="a",
            acceptance_criteria=["correct"],
            riskiest_subproblem="the sum",
            budget=BudgetProfile(final_reserve_tokens=5000),
        )
        assert p.acceptance_criteria == ["correct"]
        assert p.riskiest_subproblem == "the sum"
        assert p.budget is not None
        assert p.budget.final_reserve_tokens == 5000
        empty = Plan(approach="a")
        assert empty.acceptance_criteria == []
        assert empty.riskiest_subproblem == ""
        assert empty.budget is None

    def test_phase1_prompt_mentions_v25_fields(self):
        ctx = ActiveContext(task_id="t", task="x")
        p = build_phase1_prompt(ctx)
        assert "acceptance_criteria" in p
        assert "riskiest_subproblem" in p
        assert "final_reserve_tokens" in p

    # -- Phase 6 SelectionResult contract ---------------------------------- #
    def test_selection_result_has_v25_fields(self):
        from src.v2.contracts import SelectionResult

        s = SelectionResult(
            selected_draft_id="d1",
            required_fixes=["add design"],
            unsupported_claims=["unverified claim"],
        )
        assert s.required_fixes == ["add design"]
        assert s.unsupported_claims == ["unverified claim"]
        empty = SelectionResult(selected_draft_id="d1")
        assert empty.required_fixes == []
        assert empty.unsupported_claims == []

    async def test_selection_fixes_derived_from_evaluation(self):
        def handler(prompt):
            if "INDEPENDENT VERIFIER" in prompt:
                return _wrap(
                    {
                        "draft_id": "d1", "score": 8.0, "verification_passed": True,
                        "issues": ["contains an unsupported claim"],
                        "missing_sections": ["Design"],
                    }
                )
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        await pipeline.run(_task("What is 2+2?"))
        sel = pipeline.active_context.selection
        assert sel is not None
        assert "Design" in sel.required_fixes
        assert any("unsupported" in c for c in sel.unsupported_claims)

    # -- Phase 1 budget merge ---------------------------------------------- #
    async def test_plan_budget_merged_into_run(self):
        def handler(prompt):
            if "PLANNING stage" in prompt:
                return _wrap(
                    {
                        "approach": "a", "strategy": "s",
                        "key_requirements": ["correct"],
                        "acceptance_criteria": ["correct"],
                        "riskiest_subproblem": "the sum",
                        "budget": {
                            "final_reserve_tokens": 4321,
                            "final_repair_reserve_tokens": 1234,
                        },
                    }
                )
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        await pipeline.run(_task("What is 2+2?"))
        budget = pipeline.active_context.task_profile.budget
        assert budget.final_reserve_tokens == 4321
        assert budget.final_repair_reserve_tokens == 1234

    # -- Phase 7 synthesis carries the selector's fixes -------------------- #
    def test_phase7_prompt_includes_required_fixes(self):
        from src.v2.contracts import SelectionResult

        ctx = ActiveContext(task_id="t", task="x")
        ctx.selection = SelectionResult(
            selected_draft_id="d1",
            selected_content="The answer is 42.",
            required_fixes=["add a design section"],
            unsupported_claims=["unverified claim X"],
        )
        p = build_phase7_synthesis_prompt(ctx)
        assert "REQUIRED FIXES" in p
        assert "add a design section" in p
        assert "UNSUPPORTED CLAIMS" in p
        assert "unverified claim X" in p


# --------------------------------------------------------------------------- #
# Clean-plan regression tests
# --------------------------------------------------------------------------- #
class TestCleanPlanRegression:
    def test_pairwise_judge_once_exists(self):
        import inspect

        from src.v2 import pairwise

        assert hasattr(pairwise, "_judge_once")
        assert inspect.iscoroutinefunction(pairwise._judge_once)

    def test_budget_mode_keys_are_clean(self):
        from src.v2.budget import VALID_MODES

        assert set(VALID_MODES) == {"quick", "standard", "deep"}

    def test_budget_for_mode_reserves_match_plan(self):
        assert budget_for_mode("quick").final_reserve_tokens == 2000
        assert budget_for_mode("quick").final_repair_reserve_tokens == 1000
        assert budget_for_mode("standard").final_reserve_tokens == 6000
        assert budget_for_mode("standard").final_repair_reserve_tokens == 2000
        assert budget_for_mode("deep").final_reserve_tokens == 12000
        assert budget_for_mode("deep").final_repair_reserve_tokens == 4000

    def test_coverage_report_accepts_pass_alias(self):
        assert CoverageReport.model_validate({"pass": True}).passed is True
        assert CoverageReport.model_validate({"pass": False}).passed is False

    def test_coverage_report_coerces_repair_instructions(self):
        assert CoverageReport(repair_instructions=["fix a", "fix b"]).repair_instructions == "fix a; fix b"
        assert CoverageReport(repair_instructions=("x", "y")).repair_instructions == "x; y"
        assert CoverageReport(repair_instructions=None).repair_instructions == ""

    def test_coverage_ok_fails_on_missing_required_fixes(self):
        cr = CoverageReport(passed=True, missing_required_fixes=["add design"])
        assert cr.coverage_ok is False

    def test_coverage_ok_fails_when_actionability_false(self):
        cr = CoverageReport(passed=True, actionability_pass=False)
        assert cr.coverage_ok is False

    def test_coverage_ok_passes_when_all_good(self):
        assert CoverageReport(passed=True).coverage_ok is True

    def test_active_context_to_dict_serializes_branch_set(self):
        ctx = ActiveContext(task_id="t", task="x")
        ctx.branch_set = BranchSet(
            branches=[
                DecompositionBranch(
                    id="b1",
                    sub_problems=[SubProblemCandidate(id="sp1", description="solve")],
                )
            ],
            integration_requirement="combine",
        )
        d = ctx.to_dict()
        assert d["branch_set"] is not None
        assert d["branch_set"]["branches"][0]["id"] == "b1"
        assert d["branch_set"]["integration_requirement"] == "combine"

    async def test_phase2_drops_empty_branches_and_falls_back(self):
        def handler(prompt):
            if "DECOMPOSITION stage" in prompt:
                return _wrap(
                    {
                        "branches": [
                            {"id": "b1", "rationale": "empty", "sub_problems": []},
                        ]
                    }
                )
            return v2_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "phase2_retry" in result.techniques_used
        assert "phase2_fallback" in result.techniques_used

    async def test_phase8_fails_gate_when_required_fixes_missing(self):
        def handler(prompt):
            if "COVERAGE VALIDATION stage" in prompt:
                return _wrap(
                    {
                        "passed": True,
                        "missing_required_fixes": ["add a design section"],
                    }
                )
            if "FINAL SYNTHESIS stage" in prompt:
                return _wrap(
                    {
                        "answer": "The final answer is 42.",
                        "verification_passed": True,
                        "verification_notes": "",
                        "confidence": 0.9,
                    }
                )
            return _v22_handler(prompt)

        pipeline, _, _ = _make_pipeline_mode(handler, "standard")
        result = await pipeline.run(_task("What is 2+2?"))
        assert result.failed is False
        assert "coverage_gate_failed" in result.techniques_used
