"""Configuration for the Solomon system.

All settings can be overridden via environment variables (SOLOMON_*).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SOLOMON_{name}", default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(f"SOLOMON_{name}", default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(f"SOLOMON_{name}", default))
    except (TypeError, ValueError):
        return default


@dataclass
class LLMSettings:
    """Connection + generation settings for the local llama.cpp server."""

    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:8080"))
    model: str = field(
        default_factory=lambda: _env(
            "LLM_MODEL", "/home/bb/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf"
        )
    )
    # Reasoning models spend tokens on internal thinking; keep headroom.
    default_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))
    solve_max_tokens: int = field(default_factory=lambda: _env_int("LLM_SOLVE_MAX_TOKENS", 3072))
    default_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))
    request_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 600.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_RETRIES", 2))
    # Concurrency cap on outgoing LLM calls (llama.cpp queues the rest).
    max_concurrency: int = field(default_factory=lambda: _env_int("LLM_CONCURRENCY", 4))
    # Stream responses from llama.cpp so FINISH NOW can abort mid-stream by
    # checking the control gate on every chunk (SOLOMON_LLM_STREAM=0 disables).
    stream: bool = field(default_factory=lambda: _env("LLM_STREAM", "1") == "1")
    # Local model: $0. For hosted models set price per 1K tokens.
    price_per_1k_prompt: float = field(
        default_factory=lambda: _env_float("PRICE_1K_PROMPT", 0.0)
    )
    price_per_1k_completion: float = field(
        default_factory=lambda: _env_float("PRICE_1K_COMPLETION", 0.0)
    )

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"


@dataclass
class PipelineSettings:
    """Shared pipeline knobs."""

    # FINISH NOW: minimum confidence before the button is enabled.
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.7)
    )


@dataclass
class V2Settings:
    """Solomon v2.0 pipeline knobs (verifier-first architecture)."""

    max_context_window: int = field(
        default_factory=lambda: _env_int("V2_MAX_CONTEXT_WINDOW", 131072)
    )
    default_temperature: float = field(
        default_factory=lambda: _env_float("V2_TEMPERATURE", 0.2)
    )
    # v2.2 runtime mode: quick | standard | deep.
    default_mode: str = field(
        default_factory=lambda: _env("V2_MODE", "standard")
    )
    sandbox_enabled: bool = field(
        default_factory=lambda: os.environ.get("SOLOMON_V2_SANDBOX", "1") == "1"
    )
    retrieval_enabled: bool = field(
        default_factory=lambda: os.environ.get("SOLOMON_V2_RETRIEVAL", "0") == "1"
    )
    sandbox_timeout: float = field(
        default_factory=lambda: _env_float("V2_SANDBOX_TIMEOUT", 30.0)
    )
    # Phase answer reserves (tokens) for the dynamic max_tokens rule.
    phase_reserves: dict[str, int] = field(
        default_factory=lambda: {
            "phase0": 2000,
            "phase1": 6000,
            "phase2": 3000,
            "phase3": 10000,
            "phase4": 5000,
            "phase5": 16000,
            "phase6": 5000,
            "phase7": 16000,
        }
    )
    # Budget gates by difficulty (easy / medium / hard).
    budgets: dict[str, dict] = field(
        default_factory=lambda: {
            "easy": {
                "decomposition_branches": 1,
                "drafts": 2,
                "self_consistency_samples": 3,
                "execution_repair_rounds": 1,
                "pairwise_mode": "tiebreak",
            },
            "medium": {
                "decomposition_branches": 1,
                "drafts": 3,
                "self_consistency_samples": 5,
                "execution_repair_rounds": 2,
                "pairwise_mode": "top3",
            },
            "hard": {
                "decomposition_branches": 2,
                "drafts": 5,
                "self_consistency_samples": 7,
                "execution_repair_rounds": 2,
                "pairwise_mode": "top3",
            },
        }
    )


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    v2: V2Settings = field(default_factory=V2Settings)
    db_path: Path = field(
        default_factory=lambda: Path(_env("DB_PATH", str(DATA_DIR / "solomon.db")))
    )
    log_dir: Path = field(
        default_factory=lambda: Path(_env("LOG_DIR", str(DATA_DIR / "logs")))
    )
    server_host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
