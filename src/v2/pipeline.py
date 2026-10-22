"""Solomon v2.0 pipeline — the verifier-first orchestrator.

Runs Phases 0-7 in fresh, isolated context windows. Only the compact
:class:`ActiveContext` (verified artifacts) travels between phases — never raw
traces. The pipeline registers a control session so FINISH NOW / STOP work, and
degrades safely when the sandbox or retrieval backends are unavailable.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from config.settings import Settings, get_settings
from src.agents.control import SessionRegistry
from src.agents.state import AgentState, SolvedSubProblem
from src.llm.orchestrator import ControlInterrupted, LLMConfig, LLMOrchestrator
from src.monitoring.events import EventBus
from src.models.dataclasses import AgentResult, Task
from src.v2.active_context import ActiveContext
from src.v2.budget import (
    TokenBudget,
    budget_for_difficulty,
    budget_for_mode,
    merge_budgets,
)
from src.v2.contracts import (
    AnswerType,
    BranchSet,
    CoverageReport,
    Critique,
    CritiqueSeverity,
    CritiqueVerdict,
    DebateSynthesis,
    DecompositionBranch,
    Draft,
    Difficulty,
    EmergencyFinish,
    Evaluation,
    FinalAnswer,
    LightningIntake,
    PairwiseMode,
    Plan,
    RiskLevel,
    Route,
    SelectionResult,
    SubProblemCandidate,
    SubProblemResult,
    TaskProfile,
    VerificationReport,
    VerificationStatus,
)
from src.v2.finish_manager import FinishManager
from src.v2.pairwise import pairwise_tournament
from src.v2.prompts import (
    LIGHTNING_BASE_FORBIDDEN_PATTERNS,
    LIGHTNING_LABEL,
    build_debate_synthesis_prompt,
    build_emergency_finish_prompt,
    build_invalid_critique_reduced_prompt,
    build_invalid_critique_retry_prompt,
    build_lightning_answer_prompt,
    build_lightning_intake_prompt,
    build_lightning_reasoning_prompt,
    build_lightning_repair_prompt,
    _LIGHTNING_TRUNCATION_TAIL_WORDS,
    build_phase0_prompt,
    build_phase1_prompt,
    build_phase2_prompt,
    build_phase3_code_prompt,
    build_phase3_solve_prompt,
    build_phase4_critique_prompt,
    build_phase5_draft_prompt,
    build_phase6_eval_prompt,
    build_phase7_recover_prompt,
    build_phase7_synthesis_prompt,
    build_phase7_verify_prompt,
    build_phase8_coverage_prompt,
    build_phase8_repair_prompt,
    phase_system_prompt,
    select_drafts_for_pairwise,
)
from src.v2.reasoning import (
    ReasoningLevel,
    ReasoningPolicy,
    reasoning_directive,
    reasoning_policy_for,
    reason_then_extract,
    sub_problem_reasoning_level,
)
from src.v2.retrieval import RetrievalAdapter, build_facts_ledger
from src.v2.sandbox import SandboxAdapter
from src.v2.self_consistency import (
    is_rejected_answer,
    merge_samples,
    self_consistency,
)
from src.v2.structured import FINAL_JSON_END, FINAL_JSON_START, parse_structured

Emit = Optional[object]


class _Abort(Exception):
    pass


class _Finish(Exception):
    pass


class V2Pipeline:
    tier = "v2"

    def __init__(
        self,
        llm: LLMOrchestrator,
        event_bus: Optional[EventBus] = None,
        settings: Optional[Settings] = None,
        sessions: Optional[SessionRegistry] = None,
        synthesis: Optional[object] = None,
        retrieval: Optional[RetrievalAdapter] = None,
        sandbox: Optional[SandboxAdapter] = None,
        token_budget: Optional[TokenBudget] = None,
        mode: str = "standard",
        finish_manager: Optional[FinishManager] = None,
    ):
        self.llm = llm
        self.event_bus = event_bus
        self.settings = settings or get_settings()
        self.sessions = sessions
        self.synthesis = synthesis
        self.retrieval = retrieval or RetrievalAdapter()
        self.sandbox = sandbox or SandboxAdapter()
        self.token_budget = token_budget or TokenBudget()
        self.mode = (mode or "standard").strip().lower() or "standard"
        # NB: an empty FinishManager is falsy (it defines __len__), so use an
        # explicit None check — otherwise an injected (not-yet-populated) manager
        # would be silently replaced and the server's FINISH NOW presses would
        # never reach this pipeline's gate.
        self.finish_manager = (
            finish_manager
            if finish_manager is not None
            else (FinishManager(sessions=sessions) if sessions is not None else FinishManager())
        )
        self.active_context: Optional[ActiveContext] = None
        self.agent_state: Optional[AgentState] = None
        # The pipeline keeps per-run state (mode + token budget) on this shared
        # instance, so runs are serialized to avoid cross-task corruption.
        self._run_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def _emitter(self, task_id: str):
        if self.event_bus is None:
            return _noop_emit
        return self.event_bus.make_emitter(task_id)

    def _usage(self, task_id: str) -> tuple[float, int, int]:
        pt, ct = self.llm.cost_tracker.get_task_tokens(task_id)
        return (self.llm.cost_tracker.get_task_cost(task_id), pt, ct)

    def _register(self, task_id: str, state: AgentState) -> None:
        if self.sessions is not None:
            self.sessions.get_or_create(task_id).state = state

    def _unregister(self, task_id: str) -> None:
        if self.sessions is not None:
            self.sessions.remove(task_id)

    async def _gate(self, task_id: str) -> str:
        # v2.6: the global FINISH NOW / STOP flag (finish_event) wins — it is
        # set by the FinishManager on every press, including presses that
        # arrive mid-stream between phase gates.
        sig = self.finish_manager.signal(task_id)
        if sig != "continue":
            return sig
        if self.sessions is not None:
            sess = self.sessions.get(task_id)
            if sess is not None:
                return sess.signal()
        return "continue"

    def _bind_control(self, task_id: str) -> None:
        """v2.6: register this run with the FinishManager so a FINISH NOW can
        cancel the pipeline task and kill the sandbox immediately."""
        control = self.finish_manager.register(task_id)
        control.set_sandbox(self.sandbox)
        try:
            current = asyncio.current_task()
        except RuntimeError:  # pragma: no cover
            current = None
        if current is not None:
            control.bind_pipeline_task(current)

    @staticmethod
    def _raise_for_control(ci: ControlInterrupted) -> None:
        """Translate an in-flight control interruption into the pipeline signal."""
        if ci.signal == "stop":
            raise _Abort()
        raise _Finish()

    def _reasoning_policy(
        self,
        phase: str,
        level: Optional[ReasoningLevel] = None,
        use_rte: Optional[bool] = None,
    ) -> ReasoningPolicy:
        """Build the v2.6 reasoning policy for a phase from the run budget.

        ``level`` overrides the base level (adaptive Phase 3, truncation
        recovery). ``use_rte`` forces on/off the two-call reason-then-extract
        pattern for a phase.
        """
        return reasoning_policy_for(
            phase,
            mode=self.mode,
            reasoning_budget=self.token_budget.reasoning_budget,
            completion_budget=self.token_budget.completion_budget,
            level=level,
            use_reason_then_extract=use_rte,
        )

    async def _reason_then_extract_structured(
        self,
        prompt: str,
        schema: type,
        phase: str,
        policy: ReasoningPolicy,
        config: Optional[LLMConfig] = None,
        task_id: str = "",
        emit: Emit = None,
        intermediate: bool = True,
        check_gate: bool = True,
    ):
        """v2.6 two-call reason-then-extract for a high-stakes decision.

        Call 1 produces brief reasoning notes; call 2 produces the strict
        structured output derived from those notes. Both calls are budgeted and
        gate-checked; usage is fed to the governor (reasoning is intermediate
        work and can never consume the protected final-answer reserve).
        """
        # v2.4 Rule 7: FINISH NOW / STOP interrupts before any new LLM call.
        if check_gate and task_id:
            sig = await self._gate(task_id)
            if sig == "finish":
                raise _Finish
            if sig == "stop":
                raise _Abort
        return await reason_then_extract(
            self.llm,
            prompt,
            schema,
            policy,
            task_id=task_id,
            phase=phase,
            config=config,
            emit=emit,
            gate=check_gate,
            system=phase_system_prompt(phase),
            record_usage=(lambda r: self._record_usage(r, intermediate)),
        )

    async def _generate_structured(
        self,
        prompt: str,
        schema: type,
        phase: str,
        config: Optional[LLMConfig] = None,
        task_id: str = "",
        emit: Emit = None,
        retry: bool = True,
        intermediate: bool = True,
        check_gate: bool = True,
        reasoning: Optional[ReasoningPolicy] = None,
    ):
        """Mechanics A/B/C: budgeted, marker-wrapped, schema-validated call
        with a single structured retry on failure.

        ``intermediate`` marks the spend against the v2.2 intermediate soft cap
        (final-answer and coverage-repair calls pass ``intermediate=False`` so
        they are never starved by the cap). ``check_gate`` disables the
        FINISH NOW / STOP interruption (used by the emergency finish itself).

        v2.6: ``reasoning`` (a :class:`ReasoningPolicy`) allocates a reasoning
        level for this call. When the policy requests reason-then-extract, the
        two-call pattern is used; otherwise the level is applied as a prompt
        directive on the single structured call.
        """
        # v2.6: high-stakes decisions use the two-call reason-then-extract path.
        if reasoning is not None and reasoning.use_reason_then_extract and reasoning.enabled:
            return await self._reason_then_extract_structured(
                prompt, schema, phase, reasoning, config, task_id, emit,
                intermediate, check_gate,
            )
        # v2.4 Rule 7: FINISH NOW / STOP interrupts before any new LLM call,
        # not just at phase boundaries.
        if check_gate and task_id:
            sig = await self._gate(task_id)
            if sig == "finish":
                raise _Finish
            if sig == "stop":
                raise _Abort
        if config is None:
            config = LLMConfig(temperature=0.0)
        # v2.6: apply the reasoning level as a prompt directive (single call).
        if reasoning is not None and reasoning.enabled:
            directive = reasoning_directive(reasoning.level)
            if directive:
                prompt = f"{prompt}\n\n{directive}"
        max_tokens, needs_compression = self.token_budget.plan(prompt, phase)
        call_prompt = prompt
        if needs_compression:
            keep = self.token_budget.window - self.token_budget.reserve_for(phase)
            call_prompt = self.token_budget.truncate(prompt, keep)
            call_prompt += (
                "\n\nRespond with ONLY the JSON object between FINAL_JSON_START "
                "and FINAL_JSON_END."
            )
        cfg = LLMConfig(
            temperature=config.temperature,
            max_tokens=max_tokens,
            context_isolation=config.context_isolation,
            system=config.system or phase_system_prompt(phase),
        )
        try:
            resp = await self.llm.generate(
                prompt=call_prompt, config=cfg, task_id=task_id, phase=phase,
                gate=check_gate,
            )
        except ControlInterrupted as ci:
            self._raise_for_control(ci)
        self._record_usage(resp, intermediate)
        obj, err = parse_structured(resp.text, schema)
        if obj is not None:
            return obj, None
        if not retry:
            return None, err
        retry_prompt = (
            call_prompt
            + f"\n\nYour previous response was invalid: {err}. Respond again "
            "with ONLY valid JSON between the markers."
        )
        retry_max, _ = self.token_budget.plan(retry_prompt, phase)
        retry_cfg = LLMConfig(
            temperature=0.0, max_tokens=retry_max, context_isolation=config.context_isolation,
            system=config.system or phase_system_prompt(phase),
        )
        try:
            resp2 = await self.llm.generate(
                prompt=retry_prompt, config=retry_cfg, task_id=task_id,
                phase=f"{phase}_retry", gate=check_gate,
            )
        except ControlInterrupted as ci:
            self._raise_for_control(ci)
        self._record_usage(resp2, intermediate)
        obj2, err2 = parse_structured(resp2.text, schema)
        if obj2 is not None:
            return obj2, None
        return None, err2 or err

    def _record_usage(self, resp, intermediate: bool = True) -> None:
        """Feed the budget governor with a call's token usage.

        v2.6 P1-4: if the backend reported zero usage for a non-empty response,
        estimate locally (``len(text) // 4``) so telemetry is never 0/0."""
        prompt_tokens = int(getattr(resp, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(resp, "completion_tokens", 0) or 0)
        text = getattr(resp, "text", "") or ""
        if completion_tokens <= 0 and text:
            completion_tokens = max(1, len(text) // 4)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, len(text) // 8 + 1)
        tokens = prompt_tokens + completion_tokens
        if tokens <= 0:
            return
        self.token_budget.record(tokens)
        if intermediate:
            self.token_budget.record_intermediate(tokens)

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    async def run(self, task: Task, mode: Optional[str] = None) -> AgentResult:
        async with self._run_lock:
            if mode:
                self.mode = (mode or "standard").strip().lower() or "standard"
            start = time.monotonic()
            context = ActiveContext(task_id=task.id, task=task.original_prompt)
            state = AgentState(
                task=task.original_prompt,
                tier="v2",
                total_steps=4 if self.mode == "lightning" else 9,
            )
            self.active_context = context
            self.agent_state = state
            self._register(task.id, state)
            self._bind_control(task.id)
            emit = self._emitter(task.id)
            techniques: list[str] = []
            techniques.append(f"mode_{self.mode}")

            async def gate() -> None:
                sig = await self._gate(task.id)
                if sig == "stop":
                    raise _Abort()
                if sig == "finish":
                    raise _Finish()

            try:
                await gate()
                if self.mode == "lightning":
                    # v2.6 Lightning: a short pipeline (L0 intake-lite -> L1
                    # one-shot answer -> L2 fast gate -> L3 labeled delivery)
                    # that skips Phase 2-8 entirely.
                    await self._run_lightning(task, context, state, emit, techniques)
                    return self._build_result(task, state, context, techniques, start)
                await self._phase0(task, context, state, emit, techniques)
                await gate()
                await self._phase1(context, state, emit, techniques)
                await gate()
                await self._phase2(context, state, emit, techniques)
                await gate()
                await self._phase3(context, state, emit, techniques)
                await gate()
                await self._phase4(context, state, emit, techniques)
                await gate()
                await self._phase5(context, state, emit, techniques)
                await gate()
                await self._phase6(context, state, emit, techniques)
                await gate()
                await self._phase7(context, state, emit, techniques)
                await gate()
                await self._phase8(context, state, emit, techniques)
                return self._build_result(task, state, context, techniques, start)
            except _Abort:
                return await self._finalize_aborted(task, state, context, techniques, start, emit)
            except _Finish:
                return await self._finish(task, state, context, techniques, start, emit)
            except asyncio.CancelledError:
                # v2.6: the FinishManager cancelled this task on a FINISH NOW
                # press (real global interrupt). Translate the cancellation into
                # the same terminal paths the gate checks use.
                sig = await self._gate(task.id)
                if sig == "finish":
                    return await self._finish(task, state, context, techniques, start, emit)
                if sig == "stop":
                    return await self._finalize_aborted(task, state, context, techniques, start, emit)
                raise
            finally:
                self._unregister(task.id)

    # ------------------------------------------------------------------ #
    # v2.6 Lightning — the short pipeline (L0-L3), NOT Phases 0-8
    # ------------------------------------------------------------------ #
    _LIGHTNING_MIN_ANSWER_CHARS = 40

    @classmethod
    def _lightning_gate(cls, answer: str, intake: LightningIntake) -> list[str]:
        """L2 Fast gate: rule-based checks only (no LLM). Returns the list of
        issues; an empty list means the answer passes.

        Checks: required sections present, non-empty, not truncated, forbidden
        patterns absent (intake + contract-level base set), no JSON markers, no
        draft metadata, minimum length, and (for how-to tasks) the minimal
        actionability requirements are covered."""
        issues: list[str] = []
        text = (answer or "").strip()
        low = text.lower()
        if FINAL_JSON_START in text or FINAL_JSON_END in text:
            issues.append("marker garbage: FINAL_JSON markers present in a plain answer")
        if not text:
            issues.append("empty answer")
            return issues
        if len(text) < cls._LIGHTNING_MIN_ANSWER_CHARS:
            issues.append(f"too short (<{cls._LIGHTNING_MIN_ANSWER_CHARS} chars)")
        # Forbidden patterns: intake-emitted + contract-level base set (deduped).
        forbidden = list(dict.fromkeys(
            [p for p in (intake.forbidden_patterns or []) if p]
            + list(LIGHTNING_BASE_FORBIDDEN_PATTERNS)
        ))
        for pat in forbidden:
            if pat.lower() in low:
                issues.append(f"forbidden pattern present: {pat!r}")
        for section in intake.required_sections or []:
            if not section:
                continue
            # A section passes when its title appears (heading or plain line).
            if section.lower() not in low:
                issues.append(f"missing required section: {section!r}")
        # How-to actionability: each requirement keyword must be covered. The
        # model writes "minimal path" while the contract says "minimal_path", so
        # accept either the underscore form or the space form.
        for req in intake.actionability_requirements or []:
            if not req:
                continue
            needle = req.lower().strip()
            if needle not in low and needle.replace("_", " ") not in low:
                issues.append(f"missing actionability: {req!r}")
        # Draft metadata that must not leak into a final answer.
        for marker in ("[draft]", "draft metadata", "todo:"):
            if marker in low:
                issues.append(f"draft metadata present: {marker!r}")
        # Truncation: an answer ending on a dangling function word is likely cut.
        if cls._looks_truncated(text):
            issues.append("possible truncation: answer ends mid-sentence")
        return issues

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        """Heuristic: the last word is a dangling function word (the/a/is/...)."""
        stripped = text.rstrip()
        if not stripped:
            return False
        # Strip trailing punctuation so "…and" is detected as "…and".
        last = stripped.rstrip(".,;:!?").split()[-1] if stripped.split() else ""
        return last.lower() in _LIGHTNING_TRUNCATION_TAIL_WORDS

    @staticmethod
    def _lightning_metadata(truncated: bool = False) -> dict:
        """v2.6 P0-2: Lightning delivery metadata. The label is a UI banner
        carried HERE, never spliced into the answer body."""
        return {
            "mode": "lightning",
            "verified": False,
            "label": LIGHTNING_LABEL,
            "truncated": truncated,
        }

    async def _run_lightning(self, task, context, state, emit, techniques) -> None:
        """v2.6 Lightning: L0 intake-lite -> (P2 optional short reasoning) ->
        L1 one-shot answer -> L2 fast gate -> L3 delivery. Skips Phase 2-8
        entirely.

        - P0-1: the full L1 answer is stored as a source-of-truth artifact
          (``context.artifacts["lightning_answer"]``); delivery reads from it,
          never from a live stream buffer.
        - P0-2: the "fast, not deeply verified" label is delivery metadata, not
          part of the answer body.
        - P0-3: the L2 fast gate checks sections, forbidden patterns,
          truncation, and minimum length; ONE targeted repair on failure.
        - P1-4: token usage is recorded (with a local estimate fallback) and
          truncation is flagged when stop_reason == length.
        - P2-8: FINISH NOW after L1 returns the stored partial answer
          immediately (see ``_finish_lightning``)."""
        state.current_phase = "lightning"
        await emit("lightning", "phase_start")
        truncated = False

        # L0 — Intake-lite: one fast call.
        iprompt = build_lightning_intake_prompt(task.original_prompt)
        intake, _err = await self._generate_structured(
            iprompt, LightningIntake, "lightning_intake",
            LLMConfig(temperature=0.0), task.id, emit,
        )
        if intake is None:
            intake = LightningIntake(
                route=Route.NON_EXECUTABLE,
                answer_type=AnswerType.FACTUAL,
                required_sections=["Answer"],
                forbidden_patterns=["TODO"],
            )
            techniques.append("lightning_intake_fallback")
        else:
            techniques.append("lightning_intake")
        state.completed_steps = 1
        state.context_text = context.compact_text()
        await emit("lightning_intake", "intake", route=intake.route.value)

        # P2-7 — Optional bounded short reasoning for logic-heavy tasks (<=3
        # bullets / <=300 tokens). Skipped for simple factual/formatting tasks.
        reasoning = ""
        if intake.needs_reasoning:
            try:
                rresp = await self.llm.generate(
                    prompt=build_lightning_reasoning_prompt(task.original_prompt, intake),
                    config=LLMConfig(
                        temperature=0.2,
                        max_tokens=300,
                        system=phase_system_prompt("lightning_answer"),
                    ),
                    task_id=task.id,
                    phase="lightning_reasoning",
                )
            except ControlInterrupted as ci:
                self._raise_for_control(ci)
            reasoning = (rresp.text or "").strip()
            self._record_usage(rresp)
            techniques.append("lightning_short_reasoning")

        # L1 — One-shot answer: ONE LLM call, plain prose (PLAIN_FINAL).
        aprompt = build_lightning_answer_prompt(task.original_prompt, intake, reasoning)
        try:
            resp = await self.llm.generate(
                prompt=aprompt,
                config=LLMConfig(
                    temperature=0.2,
                    max_tokens=self.settings.llm.default_max_tokens,
                    system=phase_system_prompt("lightning_answer"),
                ),
                task_id=task.id,
                phase="lightning_answer",
            )
        except ControlInterrupted as ci:
            self._raise_for_control(ci)
        answer = (resp.text or "").strip()
        self._record_usage(resp)
        # P1-4: record truncation when the backend stopped for length.
        if resp.finish_reason == "length":
            truncated = True
            context.truncations += 1
        techniques.append("lightning_oneshot")
        state.completed_steps = 2
        if answer:
            # P0-1: store the COMPLETE L1 output as the source-of-truth artifact.
            context.artifacts["lightning_answer"] = answer
            state.best_answer = answer
            # Provisional final answer so a FINISH NOW pressed after L1 (during
            # the gate/repair) can return this partial answer immediately (P2-8).
            context.final_answer = FinalAnswer(
                answer=answer,
                verification_passed=False,
                confidence=0.5,
                verification_notes="lightning: fast, not deeply verified (pre-gate)",
                metadata=self._lightning_metadata(truncated),
            )
        state.context_text = context.compact_text()

        # L2 — Fast gate: rule-based; ONE repair call on failure only.
        issues = self._lightning_gate(answer, intake)
        if issues:
            techniques.append("lightning_fast_gate_failed")
            await emit("lightning_gate", "gate_failed", issues=issues)
            rprompt = build_lightning_repair_prompt(task.original_prompt, answer, issues)
            try:
                rresp = await self.llm.generate(
                    prompt=rprompt,
                    config=LLMConfig(
                        temperature=0.2,
                        max_tokens=self.settings.llm.default_max_tokens,
                        system=phase_system_prompt("lightning_repair"),
                    ),
                    task_id=task.id,
                    phase="lightning_repair",
                )
            except ControlInterrupted as ci:
                self._raise_for_control(ci)
            repaired = (rresp.text or "").strip()
            self._record_usage(rresp)
            techniques.append("lightning_repair")
            if repaired and not self._lightning_gate(repaired, intake):
                answer = repaired
                if rresp.finish_reason == "length":
                    truncated = True
                    context.truncations += 1
                techniques.append("lightning_fast_gate")
            else:
                if repaired:
                    answer = repaired
                techniques.append("lightning_fast_gate_failed")
            # P0-1: keep the artifact as the source of truth for the (possibly
            # repaired) answer.
            if answer:
                context.artifacts["lightning_answer"] = answer
                if context.final_answer is not None:
                    context.final_answer.answer = answer
                    context.final_answer.metadata = self._lightning_metadata(truncated)
        else:
            techniques.append("lightning_fast_gate")

        # L3 — Deliver: the stored artifact is the source of truth (P0-1). The
        # label is delivery metadata (a UI banner), never in the answer body
        # (P0-2). Never presented as verified.
        final = context.artifacts.get("lightning_answer") or answer or ""
        if not final:
            final = "I could not produce an answer before finishing."
        context.final_answer = FinalAnswer(
            answer=final,
            verification_passed=False,
            verification_notes="lightning: fast, not deeply verified",
            confidence=0.5,
            metadata=self._lightning_metadata(truncated),
        )
        state.best_answer = final
        state.consensus_confidence = 0.5
        state.completed_steps = 4
        state.context_text = context.compact_text()
        await emit("lightning", "phase_end", answer=final[:200])

    # ------------------------------------------------------------------ #
    # Phase 0 — intake, retrieval, routing, difficulty, budget
    # ------------------------------------------------------------------ #
    async def _phase0(self, task, context, state, emit, techniques) -> None:
        state.current_phase = "phase0"
        await emit("phase0", "phase_start")
        retrieval_available, retrieved = await self.retrieval.retrieve(
            task.original_prompt, task.id, emit
        )
        prompt = build_phase0_prompt(task.original_prompt, retrieval_available, retrieved)
        profile, _err = await self._generate_structured(
            prompt, TaskProfile, "phase0", LLMConfig(temperature=0.0), task.id, emit,
            reasoning=self._reasoning_policy("phase0"),
        )
        if profile is None:
            profile = TaskProfile(
                route=Route.NON_EXECUTABLE,
                difficulty=Difficulty.EASY,
                summary=task.original_prompt[:200],
            )
            techniques.append("phase0_fallback")
        # v2.2: combine the mode budget (protected reserves + soft cap) with a
        # difficulty-refined budget, taking the larger numeric fields.
        base = budget_for_mode(self.mode)
        budget = merge_budgets(base, budget_for_difficulty(profile.difficulty))
        self.token_budget.set_reserves(budget.phase_reserves)
        self.token_budget.set_mode_budget(budget)
        profile.budget = budget
        profile.retrieval_available = retrieval_available
        profile.facts = build_facts_ledger(retrieved, retrieval_available)
        context.task_profile = profile
        state.context_text = context.compact_text()
        state.completed_steps = 1
        techniques.append(f"route_{profile.route.value}")
        techniques.append(f"difficulty_{profile.difficulty.value}")
        await emit(
            "phase0",
            "phase_end",
            route=profile.route.value,
            difficulty=profile.difficulty.value,
            retrieval=retrieval_available,
        )

    # ------------------------------------------------------------------ #
    # Phase 1 — planning
    # ------------------------------------------------------------------ #
    async def _phase1(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase1"
        await emit("phase1", "phase_start")
        prompt = build_phase1_prompt(context)
        plan, _err = await self._generate_structured(
            prompt, Plan, "phase1", LLMConfig(temperature=0.2), context.task_id, emit,
            reasoning=self._reasoning_policy("phase1"),
        )
        if plan is None:
            plan = Plan(approach="Direct solution.", strategy="single-shot")
            techniques.append("phase1_fallback")
        context.plan = plan
        # v2.5 Phase 1: merge the planner's proposed budget reserves into the
        # run budget so the final synthesis and its repair stay protected.
        self._merge_plan_budget(context, plan)
        state.context_text = context.compact_text()
        state.completed_steps = 2
        techniques.append("planning")
        await emit("phase1", "phase_end", approach=plan.approach[:200])

    def _merge_plan_budget(self, context, plan: Plan) -> None:
        """Merge the planner's proposed budget reserves into the run budget.

        Only non-zero planner values override the mode defaults, so a plan that
        omits the budget (or leaves a reserve at 0) keeps the safe defaults.
        """
        pb = plan.budget
        profile = context.task_profile
        if pb is None or profile is None or profile.budget is None:
            return
        budget = profile.budget
        if pb.final_reserve_tokens:
            budget.final_reserve_tokens = int(pb.final_reserve_tokens)
        if pb.final_repair_reserve_tokens:
            budget.final_repair_reserve_tokens = int(pb.final_repair_reserve_tokens)
        if pb.intermediate_soft_cap:
            budget.intermediate_soft_cap = float(pb.intermediate_soft_cap)
        # Apply the (possibly planner-tuned) reserves to the live governor so
        # the final synthesis and its repair stay protected for the rest of run.
        self.token_budget.set_mode_budget(budget)

    # ------------------------------------------------------------------ #
    # Phase 2 — decomposition + meaningful diversity
    # ------------------------------------------------------------------ #
    async def _phase2(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase2"
        await emit("phase2", "phase_start")
        n_branches = context.task_profile.budget.decomposition_branches
        prompt = build_phase2_prompt(context, n_branches)
        branch_set, _err = await self._generate_structured(
            prompt, BranchSet, "phase2", LLMConfig(temperature=0.4), context.task_id, emit,
            reasoning=self._reasoning_policy("phase2"),
        )
        # Clean-plan DoD #3: a branch with no sub_problems is invalid. Drop
        # empty branches; if none remain, retry once with a stricter prompt,
        # then fall back. Never proceed with an empty decomposition.
        branches = self._valid_branches(branch_set)
        if not branches:
            techniques.append("phase2_retry")
            retry_prompt = (
                build_phase2_prompt(context, n_branches)
                + "\n\nIMPORTANT: Every branch MUST contain at least one "
                "sub_problem. Do not emit any empty branch."
            )
            branch_set, _err = await self._generate_structured(
                retry_prompt, BranchSet, "phase2",
                LLMConfig(temperature=0.3), context.task_id, emit,
                reasoning=self._reasoning_policy("phase2", use_rte=False),
            )
            branches = self._valid_branches(branch_set)
        if not branches:
            branches = [_fallback_branch(context)]
            techniques.append("phase2_fallback")
        context.branches = branches
        if branch_set is not None:
            context.branch_set = branch_set
            context.integration_requirement = branch_set.integration_requirement
        state.sub_problems = [sp.description for sp in context.all_sub_problems()]
        state.context_text = context.compact_text()
        state.completed_steps = 3
        techniques.append(f"decomposition_{len(branches)}branch")
        await emit(
            "phase2",
            "phase_end",
            n_branches=len(branches),
            n_subproblems=len(context.all_sub_problems()),
        )

    # ------------------------------------------------------------------ #
    # Phase 3 — execution loop (sandbox / self-consistency)
    # ------------------------------------------------------------------ #
    async def _phase3(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase3"
        await emit("phase3", "phase_start")
        budget = context.task_profile.budget
        ordered = self._topo_sort(context.all_sub_problems())
        for sp in ordered:
            if self.token_budget.should_stop_intermediate():
                self.token_budget.warnings.append(
                    "phase3: intermediate soft cap reached; stopping expansion"
                )
                techniques.append("budget_soft_cap_phase3")
                break
            result = await self._solve_sub_problem(context, sp, budget, emit, techniques)
            context.sub_problem_results.append(result)
            state.solved.append(
                SolvedSubProblem(
                    sub_problem=sp.description,
                    answer=result.answer,
                    trace=result.reasoning,
                )
            )
            if result.answer:
                state.best_answer = result.answer
            state.context_text = context.compact_text()
        state.completed_steps = 4
        techniques.append("execution_loop")
        await emit(
            "phase3",
            "phase_end",
            n_results=len(context.sub_problem_results),
            verified=sum(1 for r in context.sub_problem_results if r.verified),
        )

    @staticmethod
    def _verification_from_execution(run) -> VerificationReport:
        """v2.4 Rule 3: build a VerificationReport from real execution evidence."""
        evidence: list[str] = []
        if run.stdout:
            evidence.append(f"stdout: {run.stdout.strip()[:200]}")
        if run.output:
            evidence.append(f"output: {run.output.strip()[:200]}")
        if run.tests_passed:
            evidence.append("tests passed")
        for tr in (run.test_results or [])[:5]:
            if isinstance(tr, dict):
                evidence.append(f"test {tr.get('name', '?')}: {tr.get('passed')}")
        missing: list[str] = []
        if not run.tests_passed and run.test_results:
            missing.append("some tests failed")
        if run.errors:
            missing.extend(run.errors[:3])
        status = (
            VerificationStatus.PASS
            if run.verified
            else (VerificationStatus.PARTIAL if evidence else VerificationStatus.FAIL)
        )
        return VerificationReport(status=status, evidence=evidence, missing=missing)

    async def _solve_sub_problem(self, context, sp, budget, emit, techniques):
        # Executable / hybrid path (only when a sandbox is available).
        if sp.type in (Route.EXECUTABLE, Route.HYBRID) and self.sandbox.available:
            last_error = ""
            # v2.6: code / debugging sub-problems reason at HIGH.
            code_policy = self._reasoning_policy("phase3", level=sub_problem_reasoning_level(sp))
            for attempt in range(budget.n_repair_rounds + 1):
                prompt = build_phase3_code_prompt(context, sp)
                if attempt > 0 and last_error:
                    prompt += f"\n\nPrevious attempt failed: {last_error}. Fix and retry."
                obj, _err = await self._generate_structured(
                    prompt,
                    SubProblemResult,
                    "phase3_code",
                    LLMConfig(temperature=0.2),
                    context.task_id,
                    emit,
                    reasoning=code_policy,
                )
                code = ""
                if obj is not None:
                    if obj.execution is not None and obj.execution.code:
                        code = obj.execution.code
                    elif obj.answer:
                        code = obj.answer
                code = self._extract_code(code)
                run = await self.sandbox.run(
                    code,
                    tests=[],
                    sub_problem_id=sp.id,
                    task_id=context.task_id,
                    emit=emit,
                )
                if run.verified and (run.output or run.stdout):
                    return SubProblemResult(
                        sub_problem_id=sp.id,
                        answer=run.output or run.stdout.strip(),
                        reasoning=f"Solved by sandbox execution (attempt {attempt + 1}).",
                        execution=run,
                        verified=True,
                        verification_method="sandbox",
                        confidence=0.9,
                        depends_on=list(sp.depends_on),
                        verification=self._verification_from_execution(run),
                    )
                last_error = "; ".join(run.errors) or "execution produced no output"
            techniques.append("sandbox_failed_fallback")
        # Non-executable path (or safe downgrade): self-consistency.
        prompt = build_phase3_solve_prompt(context, sp)
        # v2.6: adaptive reasoning level by sub-problem type (Section 3). The
        # level is applied as a prompt directive on each self-consistency sample.
        directive = reasoning_directive(sub_problem_reasoning_level(sp))
        if directive:
            prompt = f"{prompt}\n\n{directive}"
        # v2.6 (Bug 6): number of samples depends on mode + sub-problem risk,
        # capped by the budget. Lightning always samples once; deep spends 3-5
        # samples only on hard/risky sub-problems.
        n_samples = self._sc_sample_count(self.mode, sp.risk, budget.n_samples)
        sc = await self_consistency(
            self.llm,
            prompt,
            LLMConfig(temperature=0.6, system=phase_system_prompt("phase3")),
            context.task_id,
            "phase3_solve",
            n_samples,
            emit=emit,
        )
        # v2.6 honest verification: ``verified`` is COMPUTED by the pipeline
        # from evidence; a model-emitted ``verified`` flag is never trusted.
        # For non-executable work the only strong checklist evidence is a
        # self-consistency agreement >= 0.5 (plus retrieval grounding when
        # available). Agreement below 0.5 is partial/unverified, never pass.
        agreement = sc.confidence
        answer = sc.answer
        evidence = [
            f"self-consistency agreement {agreement:.2f} over "
            f"{sc.total} samples (cluster {sc.count}/{sc.total})"
        ]
        if context.task_profile is not None and context.task_profile.retrieval_available:
            evidence.append("retrieval grounding available for this sub-problem")
        if agreement >= 0.5 and sc.answer:
            verified = True
            verification = VerificationReport(
                status=VerificationStatus.PASS,
                evidence=evidence,
                missing=[],
            )
            confidence = min(agreement, 1.0)
        else:
            verified = False
            # Low agreement: one optional merge call, then stay partial with a
            # low_self_consistency issue. The merged answer is still unverified.
            if sc.answers and len(set(sc.answers)) > 1:
                merged = await merge_samples(
                    self.llm,
                    prompt,
                    sc.answers,
                    LLMConfig(temperature=0.6, system=phase_system_prompt("phase3")),
                    context.task_id,
                    "phase3_solve",
                )
                if merged:
                    answer = merged
                    techniques.append("sc_low_agreement_merge")
            verification = VerificationReport(
                status=VerificationStatus.PARTIAL,
                evidence=evidence,
                missing=["low agreement — answer may be unreliable"],
                reason=(
                    f"self-consistency agreement {agreement:.2f} is below the "
                    "0.5 threshold"
                ),
            )
            techniques.append("low_self_consistency")
            confidence = min(agreement, 1.0)
        return SubProblemResult(
            sub_problem_id=sp.id,
            answer=answer,
            reasoning=(
                f"Self-consistency over {sc.total} samples "
                f"(agreement {agreement:.2f}, rejected {sc.rejected}, "
                f"retries {sc.retries})."
            ),
            verified=verified,
            verification_method="self_consistency",
            confidence=confidence,
            depends_on=list(sp.depends_on),
            verification=verification,
            sc_agreement=agreement,
        )

    @staticmethod
    def _sc_sample_count(mode: str, risk, budget_cap: int) -> int:
        """v2.6 (Bug 6): self-consistency samples by mode + sub-problem risk,
        capped by the budget's ``n_samples``. Lightning always samples once;
        standard scales 1-3 by risk; deep spends 3-5 on harder/riskier work."""
        risk_value = risk.value if hasattr(risk, "value") else str(risk).lower()
        tables = {
            "lightning": {"low": 1, "medium": 1, "high": 1},
            "quick": {"low": 1, "medium": 2, "high": 2},
            "standard": {"low": 1, "medium": 2, "high": 3},
            "deep": {"low": 3, "medium": 3, "high": 5},
        }
        table = tables.get((mode or "standard").strip().lower(), tables["standard"])
        n = table.get(risk_value, table["medium"])
        cap = max(1, int(budget_cap or 1))
        return max(1, min(int(n), cap))

    # ------------------------------------------------------------------ #
    # Phase 4 — structured critique + debate synthesis (v2.4 Rules 4 & 5)
    # ------------------------------------------------------------------ #
    async def _generate_critique_with_fallback(
        self,
        context,
        target: str,
        label: str,
        emit,
        sp: Optional[SubProblemCandidate] = None,
        result: Optional[SubProblemResult] = None,
        agreement: Optional[float] = None,
    ) -> Critique:
        """v2.4 Rule 5: never let an unparseable critique contaminate repair.

        Tries the normal critique, then a stricter JSON retry, then a
        reduced-scope retry, then returns a *valid structured* fallback
        critique (never raw text).

        v2.6: the critique receives the full context — the sub-problem
        statement/type/hint/dependencies, the verification report, and the
        self-consistency agreement (with an explicit low-agreement flag).
        """
        prompt = build_phase4_critique_prompt(
            context, target, label, sp=sp, result=result, agreement=agreement
        )
        critique, err = await self._generate_structured(
            prompt, Critique, "phase4", LLMConfig(temperature=0.2), context.task_id, emit,
            reasoning=self._reasoning_policy("phase4"),
        )
        if critique is not None:
            return critique
        retry_prompt = build_invalid_critique_retry_prompt(
            context, target, label, err or "unparseable JSON"
        )
        critique, err2 = await self._generate_structured(
            retry_prompt, Critique, "phase4", LLMConfig(temperature=0.0), context.task_id, emit,
            reasoning=self._reasoning_policy("phase4", use_rte=False),
        )
        if critique is not None:
            return critique
        reduced_prompt = build_invalid_critique_reduced_prompt(context, target, label)
        critique, _err3 = await self._generate_structured(
            reduced_prompt, Critique, "phase4", LLMConfig(temperature=0.0), context.task_id, emit,
            reasoning=self._reasoning_policy("phase4", use_rte=False),
        )
        if critique is not None:
            return critique
        return Critique(
            target=label,
            verdict=CritiqueVerdict.REVISE,
            confidence=0.5,
            critical_issues=[],
            minor_issues=[],
            repair_plan=[
                "Resolve the highest-severity debate findings.",
                "Repair missing sections.",
                "Verify the corrected result.",
            ],
        )

    @staticmethod
    def _fallback_debate_synthesis(context) -> DebateSynthesis:
        """Fold the per-sub-problem structured critiques into a synthesis."""
        critical: list = []
        minor: list = []
        repair_plan: list[str] = []
        missing: set[str] = set()
        for c in context.critiques:
            for issue in c.structured_issues():
                (critical if issue.severity == CritiqueSeverity.CRITICAL else minor).append(issue)
            for t in c.actionable_targets():
                if t not in repair_plan:
                    repair_plan.append(t)
            missing.update(c.missing_sections)
        verdict = CritiqueVerdict.REVISE if (critical or repair_plan) else CritiqueVerdict.PASS
        return DebateSynthesis(
            verdict=verdict,
            confidence=0.6,
            critical_issues=critical,
            minor_issues=minor,
            missing_sections=sorted(missing),
            repair_plan=repair_plan,
        )

    async def _phase4_debate_synthesis(self, context, state, emit, techniques) -> None:
        """v2.4 Rule 4: synthesize raw debate findings into ONE structured
        critique *before* any repair."""
        raw_findings: list[str] = []
        for c in context.critiques:
            for issue in c.structured_issues():
                raw_findings.append(
                    f"[{issue.severity.value}] {issue.description} "
                    f"(repair: {issue.repair_action})"
                )
            for t in c.actionable_targets():
                raw_findings.append(f"target: {t}")
        if not raw_findings:
            context.debate_synthesis = DebateSynthesis(
                verdict=CritiqueVerdict.PASS,
                confidence=0.9,
                critical_issues=[],
                minor_issues=[],
                missing_sections=[],
                repair_plan=[],
            )
            return
        prompt = build_debate_synthesis_prompt(context, raw_findings[:40])
        synthesis, _err = await self._generate_structured(
            prompt, DebateSynthesis, "phase4", LLMConfig(temperature=0.2), context.task_id, emit
        )
        if synthesis is None:
            synthesis = self._fallback_debate_synthesis(context)
        context.debate_synthesis = synthesis
        state.context_text = context.compact_text()
        techniques.append("debate_synthesis")

    async def _phase4(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase4"
        await emit("phase4", "phase_start")
        for r in list(context.sub_problem_results):
            target = f"Answer: {r.answer}\nReasoning: {r.reasoning}"
            label = f"sub-problem {r.sub_problem_id}"
            # v2.6: give the critic the full context — the sub-problem
            # statement/type/hint/deps, the pipeline's verification report, and
            # the self-consistency agreement (low agreement is flagged so the
            # critique can treat it as a critical issue).
            sp = self._find_sub_problem(context, r.sub_problem_id)
            critique = await self._generate_critique_with_fallback(
                context, target, label, emit, sp=sp, result=r, agreement=r.sc_agreement
            )
            # v2.2 structured self-refine: a vague 'revise' (no actionable
            # repair target) is downgraded to pass with a warning.
            if (
                critique.verdict == CritiqueVerdict.REVISE
                and not critique.actionable_targets()
            ):
                critique.verdict = CritiqueVerdict.PASS
                critique.issues = list(critique.issues) + [
                    "vague revise downgraded to pass (no actionable target)"
                ]
                self.token_budget.warnings.append(
                    f"phase4: vague revise on {r.sub_problem_id} downgraded"
                )
                techniques.append("self_refine_downgrade")
            context.critiques.append(critique)
            if critique.verdict == CritiqueVerdict.FAIL:
                sp = self._find_sub_problem(context, r.sub_problem_id)
                if sp is not None:
                    new_r = await self._solve_sub_problem(
                        context, sp, context.task_profile.budget, emit, techniques
                    )
                    context.sub_problem_results = [
                        x for x in context.sub_problem_results
                        if x.sub_problem_id != r.sub_problem_id
                    ] + [new_r]
                    techniques.append(f"rederive_{r.sub_problem_id}")
            state.context_text = context.compact_text()
        # v2.4 Rule 4: synthesize the raw debate/critique findings into ONE
        # structured critique before any repair.
        await self._phase4_debate_synthesis(context, state, emit, techniques)
        state.completed_steps = 5
        techniques.append("falsifiable_critique")
        await emit("phase4", "phase_end", n_critiques=len(context.critiques))

    # ------------------------------------------------------------------ #
    # Phase 5 — diverse drafts
    # ------------------------------------------------------------------ #
    async def _phase5(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase5"
        await emit("phase5", "phase_start")
        n_drafts = context.task_profile.budget.n_drafts
        avoid: list[str] = []
        for i in range(n_drafts):
            if i > 0 and self.token_budget.should_stop_intermediate():
                self.token_budget.warnings.append(
                    "phase5: intermediate soft cap reached; stopping draft expansion"
                )
                techniques.append("budget_soft_cap_phase5")
                break
            prompt = build_phase5_draft_prompt(context, i, n_drafts, avoid)
            draft, _err = await self._generate_structured(
                prompt, Draft, "phase5", LLMConfig(temperature=0.7), context.task_id, emit,
                reasoning=self._reasoning_policy("phase5"),
            )
            if draft is None:
                draft = Draft(
                    id=f"d{i + 1}",
                    content=self._compose_from_subproblems(context),
                    approach="composed from verified sub-problems",
                    diversity_note="fallback composition",
                )
                techniques.append("phase5_fallback")
            elif not draft.id:
                draft.id = f"d{i + 1}"
            context.drafts.append(draft)
            avoid.append(draft.approach or draft.content[:100])
            if draft.content:
                state.best_answer = draft.content
            state.context_text = context.compact_text()
        state.completed_steps = 6
        techniques.append(f"diverse_drafts_{n_drafts}")
        await emit("phase5", "phase_end", n_drafts=len(context.drafts))

    # ------------------------------------------------------------------ #
    # Phase 6 — verifier-first selection
    # ------------------------------------------------------------------ #
    async def _phase6(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase6"
        await emit("phase6", "phase_start")
        evaluations: list[Evaluation] = []
        for d in context.drafts:
            prompt = build_phase6_eval_prompt(context.task, d)
            ev, _err = await self._generate_structured(
                prompt, Evaluation, "phase6_eval", LLMConfig(temperature=0.0), context.task_id, emit,
                reasoning=self._reasoning_policy("phase6"),
            )
            if ev is None:
                ev = Evaluation(
                    draft_id=d.id,
                    score=5.0,
                    verification_passed=False,
                    verification_notes="unparseable evaluation",
                )
            if not ev.draft_id:
                ev.draft_id = d.id
            evaluations.append(ev)
        context.evaluations = evaluations

        score = {e.draft_id: e.score for e in evaluations}
        ranked = sorted(
            context.drafts, key=lambda d: score.get(d.id, 0.0), reverse=True
        )
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        top_score = score.get(top.id, 0.0)
        second_score = score.get(second.id, 0.0) if second else 0.0
        margin = top_score - second_score

        if second is None or margin >= 2.0:
            selection = SelectionResult(
                selected_draft_id=top.id,
                selected_content=top.content,
                evaluations=evaluations,
                confidence=min(1.0, top_score / 10.0),
                method="verification",
            )
        else:
            budget = context.task_profile.budget
            k = 3 if budget.pairwise_mode == PairwiseMode.TOP3 else 2
            candidates = select_drafts_for_pairwise(context.drafts, evaluations, k)
            answers = [d.content for d in candidates]
            winner_idx = await pairwise_tournament(
                self.llm, context.task, answers, context.task_id, "phase6_pw", emit=emit
            )
            if winner_idx >= 0:
                winner = candidates[winner_idx]
                selection = SelectionResult(
                    selected_draft_id=winner.id,
                    selected_content=winner.content,
                    evaluations=evaluations,
                    confidence=min(1.0, score.get(winner.id, 5.0) / 10.0 + 0.1),
                    method="pairwise",
                )
            else:
                selection = SelectionResult(
                    selected_draft_id=top.id,
                    selected_content=top.content,
                    evaluations=evaluations,
                    confidence=min(1.0, top_score / 10.0),
                    method="verification_tie_fallback",
                )
        # v2.5 Phase 6: derive concrete required fixes and unsupported claims
        # from the selected draft's evaluation, to feed into final synthesis.
        sel_ev = next(
            (e for e in evaluations if e.draft_id == selection.selected_draft_id), None
        )
        if sel_ev is not None:
            selection.required_fixes = list(
                dict.fromkeys(list(sel_ev.missing_sections) + list(sel_ev.issues))
            )
            _unverified_kw = (
                "unsupported", "unverified", "ungrounded", "no evidence", "not verified",
            )
            selection.unsupported_claims = [
                i for i in sel_ev.issues
                if any(k in i.lower() for k in _unverified_kw)
            ]
        context.selection = selection
        if selection.selected_content:
            state.best_answer = selection.selected_content
        state.context_text = context.compact_text()
        state.completed_steps = 7
        techniques.append("verifier_first_selection")
        await emit(
            "phase6",
            "phase_end",
            selected=selection.selected_draft_id,
            method=selection.method,
        )

    # ------------------------------------------------------------------ #
    # Phase 7 — final synthesis + re-verification
    # ------------------------------------------------------------------ #
    async def _phase7(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase7"
        await emit("phase7", "phase_start")
        prompt = build_phase7_synthesis_prompt(context)
        # v2.6: apply the Phase 7 reasoning level as a prompt directive.
        directive = reasoning_directive(self._reasoning_policy("phase7").level)
        if directive:
            prompt = f"{prompt}\n\n{directive}"
        # Raw generate so we can inspect finish_reason for truncation recovery.
        max_tokens, needs_compression = self.token_budget.plan(prompt, "phase7")
        call_prompt = prompt
        if needs_compression:
            keep = self.token_budget.window - self.token_budget.reserve_for("phase7")
            call_prompt = self.token_budget.truncate(prompt, keep)
        cfg = LLMConfig(temperature=0.2, max_tokens=max_tokens, context_isolation=True,
                        system=phase_system_prompt("phase7"))
        try:
            resp = await self.llm.generate(
                prompt=call_prompt, config=cfg, task_id=context.task_id, phase="phase7"
            )
        except ControlInterrupted as ci:
            self._raise_for_control(ci)
        # The final answer is protected: record total usage but NOT intermediate.
        self._record_usage(resp, intermediate=False)
        truncated = (getattr(resp, "finish_reason", "") or "") == "length"
        final, _err = parse_structured(resp.text, FinalAnswer)
        selected = (
            context.selection.selected_content
            if context.selection and context.selection.selected_content
            else state.best_answer
        )

        def _salvage() -> FinalAnswer:
            answer = self._salvage_answer(resp.text, selected)
            # Clean-plan DoD #8: never silently return a raw draft. When the
            # salvage has to fall back to the selected draft verbatim, label it.
            used_raw_draft = bool(selected) and answer.strip() == selected.strip()
            notes = "truncated or unparseable synthesis; salvaged partial answer"
            if used_raw_draft:
                notes = (
                    "Partial fallback: full synthesis was interrupted; the raw "
                    "selected draft is returned as-is and is NOT a verified "
                    "final answer."
                )
            fa = FinalAnswer(
                answer=answer,
                verification_passed=False,
                confidence=0.5,
                verification_notes=notes,
            )
            techniques.append("phase7_fallback")
            if used_raw_draft:
                techniques.append("phase7_raw_draft_fallback")
            return fa

        if truncated:
            # v2.6 truncation recovery: reduce reasoning and retry with a
            # smaller scope (missing sections only). Never repeat the same
            # full generation. If the recovery fails, fall back to salvage.
            recovered = await self._phase7_truncation_recovery(
                context, emit, techniques, resp.text, selected
            )
            if recovered is not None:
                final = recovered
                techniques.append("truncation_recovery")
            else:
                final = _salvage()
                techniques.append("truncation_salvage")
            self._mark_partial(context, "phase7 synthesis truncated")
        elif final is None:
            final = _salvage()
        # Re-verify the final answer independently (skip when partial to save budget).
        if not context.partial:
            verify_prompt = build_phase7_verify_prompt(context.task, final.answer)
            critique, _verr = await self._generate_structured(
                verify_prompt, Critique, "phase7_verify", LLMConfig(temperature=0.2),
                context.task_id, emit, intermediate=False,
            )
            if critique is not None:
                context.critiques.append(critique)
                if critique.verdict == CritiqueVerdict.FAIL:
                    final.verification_passed = False
                    final.confidence = min(final.confidence, 0.4)
                    notes = "; ".join(critique.issues)
                    final.verification_notes = (
                        f"{final.verification_notes} | re-verification failed: {notes}"
                        if final.verification_notes
                        else f"re-verification failed: {notes}"
                    )
                elif critique.verdict == CritiqueVerdict.PASS:
                    final.verification_passed = True
        context.final_answer = final
        if final.answer:
            state.best_answer = final.answer
        state.best_trace = context.compact_text()
        state.context_text = context.compact_text()
        state.consensus_answer = final.answer
        state.consensus_confidence = final.confidence
        state.completed_steps = 8
        techniques.append("final_synthesis")
        await emit(
            "phase7",
            "phase_end",
            answer=final.answer[:200],
            verified=final.verification_passed,
            confidence=final.confidence,
            partial=context.partial,
        )

    async def _phase7_truncation_recovery(
        self,
        context,
        emit,
        techniques: list[str],
        partial_text: str,
        selected: str,
    ) -> Optional[FinalAnswer]:
        """v2.6 truncation recovery (Phase 7).

        After the full synthesis truncates, retry with a *smaller scope*:
        reduced reasoning (LOW) and a prompt that generates ONLY the missing
        sections, capped to the protected final-answer reserve. This never
        repeats the same full generation. Returns a recovered :class:`FinalAnswer`
        or ``None`` when the recovery itself cannot produce a usable answer.
        """
        contract = context.answer_contract
        missing: list[str] = []
        if contract and contract.required_sections:
            low = (partial_text or "").lower()
            missing = [s for s in contract.required_sections if s.lower() not in low]
        rprompt = build_phase7_recover_prompt(context.task, partial_text or "", missing)
        # Reduced scope: cap the recovery to the protected final-answer reserve
        # (smaller than the full synthesis budget) — never the full generation.
        max_tokens = max(1, self.token_budget.reserve_for("phase7"))
        cfg = LLMConfig(
            temperature=0.2,
            max_tokens=max_tokens,
            context_isolation=True,
            system=phase_system_prompt("phase7"),
        )
        try:
            rresp = await self.llm.generate(
                prompt=rprompt, config=cfg, task_id=context.task_id, phase="phase7_recover"
            )
        except ControlInterrupted as ci:
            self._raise_for_control(ci)
        # The recovered answer is final-answer work: record total usage only.
        self._record_usage(rresp, intermediate=False)
        recovered, _err = parse_structured(rresp.text, FinalAnswer)
        if recovered is not None and recovered.answer:
            recovered.verification_passed = False
            recovered.confidence = min(recovered.confidence, 0.6)
            recovered.verification_notes = (
                "Recovered after truncation: reduced-reasoning, missing-sections "
                "retry. Not independently re-verified."
            )
            return recovered
        # The recovery produced nothing usable; fall back to salvage by the caller.
        _ = selected
        return None

    def _mark_partial(self, context, reason: str) -> None:
        context.partial = True
        context.truncations += 1
        self.token_budget.record_truncation(reason)

    # ------------------------------------------------------------------ #
    # Phase 8 — final coverage gate & repair (v2.2)
    # ------------------------------------------------------------------ #
    async def _phase8(self, context, state, emit, techniques) -> None:
        state.current_phase = "phase8"
        await emit("phase8", "phase_start")
        final = context.final_answer
        if final is None or not final.answer:
            state.completed_steps = 9
            await emit("phase8", "phase_end", skipped=True)
            return
        contract = context.answer_contract
        budget = context.task_profile.budget if context.task_profile else None
        n_repair = int(budget.n_repair_rounds) if budget else 1

        def _forbidden_hits(text: str) -> list[str]:
            if not contract or not contract.forbidden_patterns:
                return []
            low = text.lower()
            return [p for p in contract.forbidden_patterns if p.lower() in low]

        # Clean-plan DoD #9: the gate must confirm the selector's required fixes
        # were actually addressed, not just that headings exist.
        required_fixes = (
            list(context.selection.required_fixes)
            if context.selection is not None
            else []
        )

        async def _validate(answer: str) -> CoverageReport:
            forbidden = _forbidden_hits(answer)
            cprompt = build_phase8_coverage_prompt(
                context.task, answer, contract, required_fixes=required_fixes
            )
            coverage, _err = await self._generate_structured(
                cprompt, CoverageReport, "phase8_coverage",
                LLMConfig(temperature=0.0), context.task_id, emit, intermediate=False,
            )
            if coverage is None:
                coverage = CoverageReport(
                    passed=True, scope_issues=["coverage validation unavailable"]
                )
                techniques.append("phase8_validation_unavailable")
            if forbidden:
                merged = list(dict.fromkeys(list(coverage.forbidden_patterns_detected) + forbidden))
                coverage.forbidden_patterns_detected = merged
                coverage.passed = False
            return coverage

        coverage = await _validate(final.answer)
        context.coverage_report = coverage

        round_i = 0
        while (
            not coverage.coverage_ok
            and round_i < max(1, n_repair)
            and self.token_budget.can_afford_intermediate(500)
        ):
            round_i += 1
            rprompt = build_phase8_repair_prompt(
                context.task, final.answer, contract, coverage,
                debate_synthesis=context.debate_synthesis,
            )
            repaired, _rerr = await self._generate_structured(
                rprompt, FinalAnswer, "phase8_repair",
                LLMConfig(temperature=0.3), context.task_id, emit, intermediate=False,
            )
            if repaired is None or not repaired.answer:
                break
            final.answer = repaired.answer
            techniques.append(f"coverage_repair_{round_i}")
            coverage = await _validate(final.answer)
            context.coverage_report = coverage

        if coverage.coverage_ok:
            techniques.append("coverage_gate")
        else:
            techniques.append("coverage_gate_failed")
        state.context_text = context.compact_text()
        state.completed_steps = 9
        await emit(
            "phase8",
            "phase_end",
            passed=coverage.passed,
            missing=len(coverage.missing_sections),
            repairs=round_i,
        )

    # ------------------------------------------------------------------ #
    # result construction
    # ------------------------------------------------------------------ #
    def _build_result(
        self, task, state, context, techniques, start,
        early_completion=False, aborted=False, synthesis_meta=None,
        error=None, failed=False,
    ) -> AgentResult:
        cost, tp, tc = self._usage(task.id)
        final_answer = (
            context.final_answer.answer if context.final_answer else state.best_answer
        )
        confidence = state.consensus_confidence or 0.0
        metadata = (
            (context.final_answer.metadata if context.final_answer else None) or {}
        )
        return AgentResult(
            task_id=task.id,
            final_answer=final_answer,
            reasoning_trace=state.best_trace or context.compact_text(),
            tier="v2",
            critique_history=list(context.critiques),
            iterations=9,
            total_cost=cost,
            total_latency=time.monotonic() - start,
            techniques_used=techniques,
            tokens_prompt=tp,
            tokens_completion=tc,
            early_completion=early_completion,
            aborted=aborted,
            confidence=confidence,
            completion_pct=round(state.completion_pct, 3),
            state=state.to_dict(),
            synthesis=synthesis_meta,
            failed=failed,
            error=error,
            metadata=metadata,
        )

    async def _finalize_aborted(self, task, state, context, techniques, start, emit) -> AgentResult:
        state.aborted = True
        result = self._build_result(task, state, context, techniques, start, aborted=True)
        await emit(
            "v2",
            "phase_end",
            answer=result.final_answer,
            latency=result.total_latency,
            aborted=True,
            early_completion=False,
        )
        return result

    @staticmethod
    def _choose_finish_strategy(context) -> str:
        """v2.4 Rule 7: pick the best completion strategy from the captured state."""
        if context.final_answer is not None and context.final_answer.answer:
            return "return_interrupted"
        if context.selection is not None and context.selection.selected_content:
            return "finalize_selected"
        if context.drafts:
            return "pick_best_draft"
        if context.debate_synthesis is not None:
            return "summarize_issues"
        if context.sub_problem_results:
            return "synthesize_verified"
        if context.plan is not None:
            return "plan_plus_partial"
        if context.task_profile is not None:
            return "profile_direct"
        return "direct_best_effort"

    @staticmethod
    def _direct_finish_meta(state, context, method: str, confidence: str) -> dict:
        """v2.6: honest partial-report meta built from the captured context
        (same shape as the EmergencyFinish schema) without any LLM call."""
        completed = [
            f"sub-problem {r.sub_problem_id}"
            for r in context.sub_problem_results
            if r.answer
        ]
        verified = [
            f"sub-problem {r.sub_problem_id}"
            for r in context.verified_sub_problems()
        ]
        unverified = [
            r.answer[:120]
            for r in context.sub_problem_results
            if r.answer and not r.verified
        ]
        incomplete: list[str] = []
        if not context.drafts:
            incomplete.append("drafts")
        if context.final_answer is None or not context.final_answer.answer:
            incomplete.append("final synthesis")
        incomplete.append("coverage validation")
        return {
            "level": state.completeness_level(),
            "confidence": confidence,
            "completed_parts": completed,
            "incomplete_parts": incomplete,
            "verified_parts": verified,
            "unverified_claims": unverified,
            "method": method,
        }

    async def _emergency_finish_shielded(self, task, context, strategy, emit):
        """v2.6: run the ONE emergency synthesis call in a separate, non-
        cancelled task.

        A FINISH NOW cancels the pipeline task; awaiting the emergency call
        inline would immediately raise CancelledError and starve the very call
        that produces the partial answer. So it runs as its own task, awaited
        through ``asyncio.shield`` with ``check_gate=False`` — it can never
        re-trigger the coverage gate or any phase boundary. If a *second*
        interrupt arrives while waiting, we give up waiting and return None
        (the caller falls back to the best existing partial answer)."""

        async def _call():
            eprompt = build_emergency_finish_prompt(
                task.original_prompt, context, strategy=strategy
            )
            obj, _err = await self._generate_structured(
                eprompt, EmergencyFinish, "finish_now",
                LLMConfig(temperature=0.2), task.id, emit, retry=False,
                intermediate=False, check_gate=False,
            )
            return obj

        etask = asyncio.create_task(_call())
        try:
            return await asyncio.shield(etask)
        except BaseException:
            return None

    async def _finish_lightning(self, task, state, context, techniques, start, emit) -> AgentResult:
        """v2.6 P2-8: FINISH NOW inside Lightning returns the best available
        partial answer immediately, labeled with what is complete and what is
        not (no LLM call, no deep phases)."""
        if self.sandbox is not None:
            try:
                self.sandbox.kill()
            except Exception:  # noqa: BLE001
                pass
        control = self.finish_manager.get(task.id)
        if control is not None:
            control.snapshot(context)
        context.partial = True

        # The stored L1 artifact (or provisional final answer) is the best
        # available partial answer.
        answer = context.artifacts.get("lightning_answer") or ""
        if not answer and context.final_answer is not None:
            answer = context.final_answer.answer or ""

        completed = ["L0 intake"]
        incomplete = ["L1 answer", "L2 fast gate", "L3 delivery"]
        confidence = "low"
        if answer:
            completed.append("L1 answer")
            incomplete = ["L2 fast gate", "L3 delivery"]
            confidence = "medium"
        else:
            answer = (
                "I was interrupted before completing the Lightning answer for: "
                f"{task.original_prompt}\n"
                "(confidence: low — no answer produced yet)"
            )

        meta = {
            "level": state.completeness_level(),
            "confidence": confidence,
            "completed_parts": completed,
            "incomplete_parts": incomplete,
            "verified_parts": [],
            "unverified_claims": [],
            "method": "lightning_interrupted",
            "finish_strategy": "lightning_partial",
            "partial": True,
            "mode": "lightning",
            "label": LIGHTNING_LABEL,
        }
        if answer and (context.final_answer is None or context.final_answer.answer != answer):
            context.final_answer = FinalAnswer(
                answer=answer,
                verification_passed=False,
                confidence=0.5 if confidence == "medium" else 0.3,
                verification_notes="FINISH NOW partial (lightning)",
                metadata=self._lightning_metadata(),
            )
        state.finish_now = True
        state.synthesis = meta
        conf_map = {"low": 0.3, "medium": 0.5, "high": 0.8}
        state.consensus_confidence = conf_map.get(confidence, 0.5)
        techniques = list(techniques) + [
            "finish_now_synthesis",
            "finish_strategy_lightning_partial",
        ]
        result = self._build_result(
            task, state, context, techniques, start,
            early_completion=True, synthesis_meta=meta,
        )
        await emit(
            "v2", "phase_end",
            answer=result.final_answer, latency=result.total_latency,
            aborted=False, early_completion=True,
        )
        return result

    async def _finish(self, task, state, context, techniques, start, emit) -> AgentResult:
        # v2.6 P2-8: Lightning has its own immediate partial-answer path.
        if self.mode == "lightning":
            return await self._finish_lightning(task, state, context, techniques, start, emit)
        meta: dict = {}
        answer = ""
        # v2.4 Rule 7: FINISH NOW is immediate — abort in-flight work first.
        if self.sandbox is not None:
            try:
                self.sandbox.kill()
            except Exception:  # noqa: BLE001
                pass
        # v2.6: capture the ActiveContext snapshot for the interrupt record.
        control = self.finish_manager.get(task.id)
        if control is not None:
            control.snapshot(context)
        strategy = self._choose_finish_strategy(context)
        context.finish_strategy = strategy
        context.partial = True
        state.context_text = context.compact_text()

        if strategy == "return_interrupted" and context.final_answer is not None:
            # v2.6: pressed during/after final synthesis — return the current
            # best partial/final answer directly, with no LLM call.
            fa = context.final_answer
            answer = fa.answer
            meta = self._direct_finish_meta(
                state, context,
                method="return_interrupted",
                confidence="high" if fa.verification_passed else "medium",
            )
        elif strategy == "direct_best_effort":
            # v2.6: pressed before any work — direct low-confidence answer,
            # no LLM call.
            answer = (
                f"I was interrupted before completing any work on: "
                f"{task.original_prompt}\n"
                "(confidence: low — no work completed)"
            )
            meta = self._direct_finish_meta(
                state, context, method="direct_best_effort", confidence="low"
            )
        else:
            # v2.6: exactly ONE emergency synthesis call (EmergencyFinish
            # schema), in a separate non-cancelled task, never followed by
            # Phase 8.
            ef = await self._emergency_finish_shielded(task, context, strategy, emit)
            if ef is not None and ef.final_answer:
                answer = ef.final_answer
                meta = {
                    "level": state.completeness_level(),
                    "confidence": ef.confidence.value,
                    "completed_parts": ef.completed_parts,
                    "incomplete_parts": ef.incomplete_parts,
                    "verified_parts": ef.verified_parts,
                    "unverified_claims": ef.unverified_claims,
                    "method": "emergency_finish",
                    "finish_strategy": strategy,
                }
            # Legacy fallback (also the path used when the emergency schema is
            # absent or the emergency call was double-interrupted).
            if not answer and self.synthesis is not None:
                answer, trace, meta = await self.synthesis.synthesize(
                    state, task.id, emit=emit
                )
                state.record_answer(answer, trace)
            if not answer:
                answer = state.best_answer or ""
                meta = meta or {
                    "level": state.completeness_level(),
                    "note": "no synthesis configured",
                }
        # v2.4 Rule 7: always record the chosen finish strategy + partial label.
        meta = dict(meta)
        meta.setdefault("finish_strategy", strategy)
        meta.setdefault("partial", True)
        techniques = list(techniques) + ["finish_now_synthesis", f"finish_strategy_{strategy}"]
        if strategy != "return_interrupted" or context.final_answer.answer != answer:
            context.final_answer = FinalAnswer(
                answer=answer,
                verification_passed=False,
                confidence=0.5,
                verification_notes="FINISH NOW partial synthesis",
            )
        state.finish_now = True
        state.synthesis = meta
        conf_map = {"low": 0.3, "medium": 0.5, "high": 0.8}
        state.consensus_confidence = conf_map.get(str(meta.get("confidence", "")), 0.5)
        result = self._build_result(
            task, state, context, techniques, start,
            early_completion=True, synthesis_meta=meta,
        )
        await emit(
            "v2",
            "phase_end",
            answer=result.final_answer,
            latency=result.total_latency,
            aborted=False,
            early_completion=True,
        )
        return result

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _valid_branches(branch_set: Optional[BranchSet]) -> list[DecompositionBranch]:
        """Clean-plan DoD #3: return only branches that contain at least one
        sub_problem; empty branches are invalid and must never be kept."""
        if branch_set is None:
            return []
        return [b for b in (branch_set.branches or []) if b.sub_problems]

    @staticmethod
    def _topo_sort(sub_problems: list[SubProblemCandidate]) -> list[SubProblemCandidate]:
        by_id = {sp.id: sp for sp in sub_problems}
        visited: set[str] = set()
        order: list[SubProblemCandidate] = []

        def visit(sp: SubProblemCandidate, stack: set[str]) -> None:
            if sp.id in visited or sp.id in stack:
                return
            stack.add(sp.id)
            for dep in sp.depends_on:
                if dep in by_id:
                    visit(by_id[dep], stack)
            stack.discard(sp.id)
            visited.add(sp.id)
            order.append(sp)

        for sp in sub_problems:
            visit(sp, set())
        return order

    @staticmethod
    def _find_sub_problem(context, sub_problem_id: str) -> Optional[SubProblemCandidate]:
        for sp in context.all_sub_problems():
            if sp.id == sub_problem_id:
                return sp
        return None

    @staticmethod
    def _extract_code(text: str) -> str:
        if not text:
            return ""
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _salvage_answer(text: str, fallback: str) -> str:
        """Recover a usable answer from truncated / unparseable synthesis text."""
        if text:
            m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.DOTALL)
            if m:
                val = m.group(1).strip()
                if val:
                    return val
        if text and text.strip():
            return text.strip()
        return fallback or ""

    @staticmethod
    def _compose_from_subproblems(context) -> str:
        parts = [f"{r.sub_problem_id}: {r.answer}" for r in context.sub_problem_results]
        return " ".join(parts) if parts else context.task


async def _noop_emit(phase: str, event_type: str, **data) -> None:  # pragma: no cover
    pass


def _fallback_branch(context: ActiveContext) -> DecompositionBranch:
    return DecompositionBranch(
        id="b1",
        rationale="Fallback: treat the whole task as a single sub-problem.",
        sub_problems=[
            SubProblemCandidate(
                id="sp1",
                description=context.task,
                type=(
                    context.task_profile.route
                    if context.task_profile
                    else Route.NON_EXECUTABLE
                ),
                risk=RiskLevel.MEDIUM,
            )
        ],
    )
