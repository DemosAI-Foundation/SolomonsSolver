"""v2.2 Budget Governor.

The budget governor is the single source of truth for how much token budget the
pipeline is allowed to spend. It keeps the v2.0 per-phase reserve / compression
behaviour (``plan`` / ``truncate`` / ``set_reserves``) and adds:

- a per-mode budget (quick / standard / deep) with protected reserves for the
  final answer and its coverage repair (spec: "Budget Governor");
- ``merge_budgets`` — combines a base (mode) budget with a difficulty-refined
  budget, taking the larger numeric fields while always preserving the mode's
  protected reserves;
- cumulative usage tracking and an *intermediate soft cap* so the pipeline
  stops expanding sub-problems / drafts once it has spent a fraction of the
  window, leaving room for the final answer;
- truncation tracking so Phase 7/8 can switch to partial-answer salvage.
"""
from __future__ import annotations

from src.v2.contracts import BudgetProfile, BudgetState, Difficulty

MAX_CONTEXT_WINDOW = 128000

DEFAULT_PHASE_RESERVES = {
    "phase0": 1000,
    "phase1": 2000,
    "phase2": 2000,
    "phase3": 6000,
    "phase4": 2000,
    "phase5": 4000,
    "phase6": 2000,
    "phase7": 6000,
}


# --------------------------------------------------------------------------- #
# Mode budgets (v2.2)
# --------------------------------------------------------------------------- #
# Each mode carries its own protected reserves and intermediate soft cap.
# ``merge_budgets`` preserves these even when a difficulty budget is larger.
MODE_BUDGETS: dict[str, BudgetProfile] = {
    "quick": BudgetProfile(
        difficulty=Difficulty.EASY,
        mode="quick",
        decomposition_branches=1,
        n_drafts=2,
        n_samples=2,
        n_repair_rounds=1,
        pairwise_mode="tie_break",
        final_reserve_tokens=2000,
        final_repair_reserve_tokens=1000,
        intermediate_soft_cap=0.6,
        # v2.6: separate reasoning / completion pools (Section 7 defaults).
        reasoning_budget_tokens=500,
        completion_budget_tokens=1500,
        phase_reserves=dict(DEFAULT_PHASE_RESERVES),
    ),
    "standard": BudgetProfile(
        difficulty=Difficulty.MEDIUM,
        mode="standard",
        decomposition_branches=1,
        n_drafts=3,
        n_samples=3,
        n_repair_rounds=1,
        pairwise_mode="tie_break",
        final_reserve_tokens=6000,
        final_repair_reserve_tokens=2000,
        intermediate_soft_cap=0.7,
        reasoning_budget_tokens=2000,
        completion_budget_tokens=4000,
        phase_reserves=dict(DEFAULT_PHASE_RESERVES),
    ),
    "deep": BudgetProfile(
        difficulty=Difficulty.HARD,
        mode="deep",
        decomposition_branches=2,
        n_drafts=5,
        n_samples=5,
        n_repair_rounds=2,
        pairwise_mode="top3",
        final_reserve_tokens=12000,
        final_repair_reserve_tokens=4000,
        intermediate_soft_cap=0.7,
        reasoning_budget_tokens=6000,
        completion_budget_tokens=8000,
        phase_reserves={
            "phase0": 1500,
            "phase1": 3000,
            "phase2": 3000,
            "phase3": 10000,
            "phase4": 3000,
            "phase5": 8000,
            "phase6": 3000,
            "phase7": 12000,
        },
    ),
}

VALID_MODES = tuple(MODE_BUDGETS.keys())


def budget_for_mode(mode: str) -> BudgetProfile:
    """Return a copy of the budget for a named mode (quick/standard/deep).

    Unknown modes fall back to ``standard``.
    """
    key = (mode or "standard").strip().lower()
    if key not in MODE_BUDGETS:
        key = "standard"
    return MODE_BUDGETS[key].model_copy(deep=True)


def budget_for_difficulty(difficulty) -> BudgetProfile:
    """Backward-compatible difficulty-based budget (v2.0 behaviour).

    Accepts either a :class:`Difficulty` enum or its string value.
    """
    if isinstance(difficulty, str):
        difficulty = Difficulty(difficulty)
    if difficulty == Difficulty.EASY:
        return BudgetProfile(
            difficulty=difficulty,
            decomposition_branches=1,
            n_drafts=2,
            n_samples=3,
            n_repair_rounds=1,
            pairwise_mode="tie_break",
            phase_reserves=dict(DEFAULT_PHASE_RESERVES),
        )
    if difficulty == Difficulty.MEDIUM:
        return BudgetProfile(
            difficulty=difficulty,
            decomposition_branches=1,
            n_drafts=3,
            n_samples=5,
            n_repair_rounds=1,
            pairwise_mode="tie_break",
            phase_reserves=dict(DEFAULT_PHASE_RESERVES),
        )
    return BudgetProfile(
        difficulty=difficulty,
        decomposition_branches=2,
        n_drafts=5,
        n_samples=7,
        n_repair_rounds=2,
        pairwise_mode="top3",
        phase_reserves={
            "phase0": 1500,
            "phase1": 3000,
            "phase2": 3000,
            "phase3": 10000,
            "phase4": 3000,
            "phase5": 8000,
            "phase6": 3000,
            "phase7": 12000,
        },
    )


def merge_budgets(base: BudgetProfile, refine: BudgetProfile) -> BudgetProfile:
    """Merge a base (mode) budget with a difficulty-refined budget.

    Numeric fields take the larger value so a harder task never gets *less*
    budget than its mode baseline, but the base mode's protected reserves
    (final reserve, repair reserve, intermediate soft cap) always win so the
    final answer is never starved by a difficulty-based reserve.
    """
    merged = base.model_copy(deep=True)
    if refine.difficulty != Difficulty.EASY:
        merged.difficulty = refine.difficulty
    merged.decomposition_branches = max(base.decomposition_branches, refine.decomposition_branches)
    merged.n_drafts = max(base.n_drafts, refine.n_drafts)
    merged.n_samples = max(base.n_samples, refine.n_samples)
    merged.n_repair_rounds = max(base.n_repair_rounds, refine.n_repair_rounds)
    # Pairwise: prefer the more expensive mode if either requests "top3".
    if "top3" in (base.pairwise_mode, refine.pairwise_mode):
        merged.pairwise_mode = "top3"
    else:
        merged.pairwise_mode = base.pairwise_mode
    # Merge per-phase reserves: take the max per phase.
    reserves: dict[str, int] = dict(base.phase_reserves)
    for k, v in refine.phase_reserves.items():
        reserves[k] = max(int(reserves.get(k, 0)), int(v))
    merged.phase_reserves = reserves
    # Preserve the base mode's governor settings (reserves + soft cap).
    merged.mode = base.mode
    merged.final_reserve_tokens = base.final_reserve_tokens
    merged.final_repair_reserve_tokens = base.final_repair_reserve_tokens
    merged.intermediate_soft_cap = base.intermediate_soft_cap
    # v2.6: preserve the base mode's reasoning / completion pools so a
    # difficulty budget can never starve (or inflate) the reasoning budget.
    merged.reasoning_budget_tokens = base.reasoning_budget_tokens
    merged.completion_budget_tokens = base.completion_budget_tokens
    return merged


# --------------------------------------------------------------------------- #
# TokenBudget (v2.0 compression + v2.2 governor)
# --------------------------------------------------------------------------- #
class TokenBudget:
    """Tracks token usage against a budget, plans per-call compression (v2.0)
    and enforces the v2.2 governor (protected reserves + soft cap)."""

    def __init__(
        self,
        window: int = MAX_CONTEXT_WINDOW,
        reserves: dict[str, int] | None = None,
        chars_per_token: int = 4,
        profile: BudgetProfile | None = None,
        final_reserve_tokens: int | None = None,
        final_repair_reserve_tokens: int | None = None,
        intermediate_soft_cap: float | None = None,
    ) -> None:
        self.window = int(window)
        self.used = 0
        self.chars_per_token = max(1, int(chars_per_token))
        self._reserves: dict[str, int] = dict(DEFAULT_PHASE_RESERVES)
        if reserves:
            self._reserves.update(reserves)
        self.profile = profile or BudgetProfile()
        # v2.2 governor state.
        self.final_reserve_tokens = (
            int(final_reserve_tokens)
            if final_reserve_tokens is not None
            else int(self.profile.final_reserve_tokens)
        )
        self.final_repair_reserve_tokens = (
            int(final_repair_reserve_tokens)
            if final_repair_reserve_tokens is not None
            else int(self.profile.final_repair_reserve_tokens)
        )
        self.intermediate_soft_cap = (
            float(intermediate_soft_cap)
            if intermediate_soft_cap is not None
            else float(self.profile.intermediate_soft_cap)
        )
        self.spent_intermediate = 0
        self.truncations = 0
        self.warnings: list[str] = []

    # -- v2.0 per-phase reserves / compression ---------------------------- #
    def estimate(self, text: str) -> int:
        """Rough token estimate (chars / chars_per_token)."""
        return len(text) // self.chars_per_token

    def set_reserves(self, reserves: dict[str, int]) -> None:
        """Merge new per-phase reserves into the existing set."""
        self._reserves.update(reserves or {})

    def reserve_for(self, phase: str) -> int:
        return int(self._reserves.get(phase, 0) or 0)

    def plan(self, prompt: str, phase: str) -> tuple[int, bool]:
        """Return ``(max_tokens, needs_compression)`` for a call.

        ``max_tokens`` is the answer budget, capped by the phase reserve.
        ``needs_compression`` is True when the prompt alone does not fit in the
        remaining window (the caller should truncate the prompt).
        """
        prompt_tokens = self.estimate(prompt)
        reserve = self.reserve_for(phase)
        available = max(0, self.window - self.used)
        if prompt_tokens >= available:
            return (1, True)
        answer_budget = available - prompt_tokens
        max_tokens = min(reserve, answer_budget) if reserve > 0 else answer_budget
        return (max_tokens, False)

    def truncate(self, text: str, keep_tokens: int) -> str:
        keep_chars = max(0, int(keep_tokens * self.chars_per_token))
        if len(text) <= keep_chars:
            return text
        return text[:keep_chars]

    def remaining(self) -> int:
        return max(0, self.window - self.used)

    def exhausted(self) -> bool:
        return self.used >= self.window

    # -- v2.2 governor configuration -------------------------------------- #
    def set_mode_budget(
        self,
        profile: BudgetProfile,
        *,
        final_reserve_tokens: int | None = None,
        final_repair_reserve_tokens: int | None = None,
        intermediate_soft_cap: float | None = None,
    ) -> None:
        """Adopt a mode budget's governor settings (used by the pipeline)."""
        self.profile = profile
        self.final_reserve_tokens = (
            int(final_reserve_tokens)
            if final_reserve_tokens is not None
            else int(profile.final_reserve_tokens)
        )
        self.final_repair_reserve_tokens = (
            int(final_repair_reserve_tokens)
            if final_repair_reserve_tokens is not None
            else int(profile.final_repair_reserve_tokens)
        )
        self.intermediate_soft_cap = (
            float(intermediate_soft_cap)
            if intermediate_soft_cap is not None
            else float(profile.intermediate_soft_cap)
        )

    # -- v2.2 usage accounting -------------------------------------------- #
    def record(self, tokens: int) -> None:
        """Record total token usage (prompt + completion)."""
        self.used += max(0, int(tokens))

    def record_intermediate(self, tokens: int) -> None:
        """Record usage that counts against the intermediate soft cap.

        Final-answer and coverage-repair spend is *not* recorded here so the
        soft cap can never eat into the protected final-answer budget.
        """
        self.spent_intermediate += max(0, int(tokens))

    # -- v2.2 governor queries -------------------------------------------- #
    def intermediate_budget(self) -> int:
        """How much of the window may be spent on intermediate work."""
        protected = self.final_reserve_tokens + self.final_repair_reserve_tokens
        return max(0, self.window - protected)

    def intermediate_cap(self) -> int:
        """The absolute intermediate spend cap (soft cap fraction of window)."""
        return int(self.window * self.intermediate_soft_cap)

    def should_stop_intermediate(self) -> bool:
        """True once intermediate spend has hit the soft cap or the
        intermediate budget (window minus protected reserves)."""
        if self.spent_intermediate <= 0:
            return False
        cap = min(self.intermediate_cap(), self.intermediate_budget())
        return self.spent_intermediate >= cap

    def can_afford(self, tokens: int) -> bool:
        """True if the *total* budget can still cover ``tokens``."""
        return self.remaining() >= max(0, int(tokens))

    def can_afford_intermediate(self, tokens: int) -> bool:
        """True if intermediate work can still spend ``tokens`` without
        breaching the soft cap or the protected-reserve boundary."""
        if self.should_stop_intermediate():
            return False
        cap = min(self.intermediate_cap(), self.intermediate_budget())
        return (self.spent_intermediate + max(0, int(tokens))) <= cap

    # -- v2.6 reasoning / completion pools --------------------------------- #
    @property
    def reasoning_budget(self) -> int:
        """The mode's total reasoning-token pool (Section 7)."""
        return int(getattr(self.profile, "reasoning_budget_tokens", 0) or 0)

    @property
    def completion_budget(self) -> int:
        """The mode's structured-completion-token pool (Section 7)."""
        return int(getattr(self.profile, "completion_budget_tokens", 0) or 0)

    def record_truncation(self, reason: str = "truncated") -> None:
        """Record a truncation event so later phases can salvage."""
        self.truncations += 1
        self.warnings.append(f"truncation: {reason}")

    def state(self) -> BudgetState:
        """Snapshot for ActiveContext / final answer metadata."""
        return BudgetState(
            mode=self.profile.mode,
            final_reserve_tokens=self.final_reserve_tokens,
            final_repair_reserve_tokens=self.final_repair_reserve_tokens,
            intermediate_soft_cap=self.intermediate_soft_cap,
            used_tokens=self.used,
            remaining_tokens=self.remaining(),
            truncations=self.truncations,
            warnings=list(self.warnings),
            final_answer_protected=True,
            partial_answer_allowed=self.truncations > 0,
        )
