"""SQLite-backed trace/answer store.

Tables:
- tasks:  one row per routed task with final result + metrics
- events: full pipeline trace (every phase, LLM call, critique, debate round)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from src.models.dataclasses import AgentResult, Task

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    tier TEXT,
    task_type TEXT,
    complexity REAL,
    status TEXT DEFAULT 'running',
    created_at REAL,
    finished_at REAL,
    final_answer TEXT,
    reasoning_trace TEXT,
    iterations INTEGER DEFAULT 0,
    latency REAL DEFAULT 0,
    cost REAL DEFAULT 0,
    tokens_prompt INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,
    techniques TEXT DEFAULT '[]',
    failed INTEGER DEFAULT 0,
    error TEXT,
    escalated_from TEXT,
    early_completion INTEGER DEFAULT 0,
    aborted INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0,
    completion_pct REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    ts REAL NOT NULL,
    phase TEXT,
    event_type TEXT,
    data TEXT,
    parent TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, ts);
"""


class TraceStore:
    """Thread-safe SQLite store for tasks and trace events."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after initial schema to older databases."""
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(tasks)")
        }
        for col, decl in (
            ("early_completion", "INTEGER DEFAULT 0"),
            ("aborted", "INTEGER DEFAULT 0"),
            ("confidence", "REAL DEFAULT 0"),
            ("completion_pct", "REAL DEFAULT 0"),
        ):
            if col not in existing:
                self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {decl}")

    # ---------- tasks ----------

    def upsert_task(self, task: Task) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO tasks (id, prompt, tier, task_type, complexity, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?)
                   ON CONFLICT(id) DO UPDATE SET
                     tier=excluded.tier, task_type=excluded.task_type,
                     complexity=excluded.complexity""",
                (
                    task.id,
                    task.original_prompt,
                    task.tier,
                    task.task_type,
                    task.complexity_score,
                    task.created_at,
                ),
            )
            self._conn.commit()

    def finish_task(self, result: AgentResult) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE tasks SET status='done', finished_at=?, final_answer=?,
                    reasoning_trace=?, iterations=?, latency=?, cost=?,
                    tokens_prompt=?, tokens_completion=?, techniques=?,
                    failed=?, error=?, tier=?, escalated_from=?,
                    early_completion=?, aborted=?, confidence=?, completion_pct=?
                    WHERE id=?""",
                (
                    time.time(),
                    result.final_answer,
                    result.reasoning_trace,
                    result.iterations,
                    result.total_latency,
                    result.total_cost,
                    result.tokens_prompt,
                    result.tokens_completion,
                    json.dumps(result.techniques_used),
                    int(result.failed),
                    result.error,
                    result.tier,
                    result.escalated_from,
                    int(result.early_completion),
                    int(result.aborted),
                    result.confidence,
                    result.completion_pct,
                    result.task_id,
                ),
            )
            self._conn.commit()

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["techniques"] = json.loads(d.get("techniques") or "[]")
        except json.JSONDecodeError:
            d["techniques"] = []
        return d

    # ---------- events ----------

    def record_event(
        self,
        task_id: str,
        phase: str,
        event_type: str,
        data: Optional[dict] = None,
        parent: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (task_id, ts, phase, event_type, data, parent) VALUES (?,?,?,?,?,?)",
                (
                    task_id,
                    ts or time.time(),
                    phase,
                    event_type,
                    json.dumps(data or {}, default=str),
                    parent,
                ),
            )
            self._conn.commit()

    def get_task_events(self, task_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, phase, event_type, data, parent FROM events WHERE task_id=? ORDER BY ts, id",
                (task_id,),
            ).fetchall()
        out = []
        for r in rows:
            try:
                data = json.loads(r["data"] or "{}")
            except json.JSONDecodeError:
                data = {}
            out.append(
                {
                    "ts": r["ts"],
                    "phase": r["phase"],
                    "event_type": r["event_type"],
                    "data": data,
                    "parent": r["parent"],
                }
            )
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
