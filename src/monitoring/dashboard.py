"""Monitoring dashboard: aggregate metrics over completed tasks."""
from __future__ import annotations

from collections import defaultdict

from src.models.dataclasses import AgentResult, TIER_DEEP, TIER_LIGHTNING, TIER_OMEGA
from src.models.dataclasses import TIERS


class MonitoringDashboard:
    """Real-time metrics for agent performance."""

    def __init__(self):
        self._latency: dict[str, list[float]] = defaultdict(list)
        self._cost: dict[str, list[float]] = defaultdict(list)
        self._tokens: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._critique_verdicts: list[str] = []
        self._refine_iterations: list[int] = []
        self._technique_tasks: dict[str, list[str]] = defaultdict(list)
        self._failure_modes: dict[str, int] = defaultdict(int)
        self.tasks_processed = 0
        self.tasks_by_tier = {t: 0 for t in TIERS}
        self.escalations = 0
        self.early_completions = 0
        self.aborted = 0

    def record_task(self, result: AgentResult) -> None:
        self.tasks_processed += 1
        tier = result.tier if result.tier in self.tasks_by_tier else TIER_LIGHTNING
        self.tasks_by_tier[tier] += 1
        self._latency[tier].append(result.total_latency)
        self._cost[tier].append(result.total_cost)
        self._tokens[tier].append((result.tokens_prompt, result.tokens_completion))
        for critique in result.critique_history:
            verdict = getattr(critique, "verdict", None)
            self._critique_verdicts.append(
                verdict.value if hasattr(verdict, "value") else str(verdict)
            )
        self._refine_iterations.append(result.iterations)
        for tech in result.techniques_used:
            self._technique_tasks[tech].append(result.task_id)
        if result.failed:
            mode = (result.error or "unknown").split(":")[0][:60]
            self._failure_modes[mode] += 1
        if result.escalated_from:
            self.escalations += 1
        if result.early_completion:
            self.early_completions += 1
        if result.aborted:
            self.aborted += 1

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def get_dashboard(self) -> dict:
        pass_rate = 0.0
        if self._critique_verdicts:
            pass_rate = (
                sum(1 for v in self._critique_verdicts if v == "pass")
                / len(self._critique_verdicts)
            )
        return {
            "tasks_processed": self.tasks_processed,
            "tasks_by_tier": dict(self.tasks_by_tier),
            "avg_latency_by_tier": {
                t: round(self._avg(self._latency[t]), 3) for t in TIERS
            },
            "avg_cost_by_tier": {t: round(self._avg(self._cost[t]), 6) for t in TIERS},
            "tokens_by_tier": {
                t: {
                    "prompt": sum(p for p, _ in self._tokens[t]),
                    "completion": sum(c for _, c in self._tokens[t]),
                }
                for t in TIERS
            },
            "critique_pass_rate": round(pass_rate, 4),
            "critique_verdicts": dict(
                defaultdict(int, {v: self._critique_verdicts.count(v) for v in set(self._critique_verdicts)})
            ),
            "refine_iterations_avg": round(self._avg(self._refine_iterations), 3),
            "technique_usage": {k: len(v) for k, v in self._technique_tasks.items()},
            "failure_modes": dict(self._failure_modes),
            "escalation_rate": (
                round(self.escalations / self.tasks_processed, 4)
                if self.tasks_processed
                else 0.0
            ),
            "early_completions": self.early_completions,
            "aborted": self.aborted,
        }
