"""Command-line interface for the Solomon system.

Usage:
  python -m src.cli chat "your task" [--tier auto|lightning|deep|omega]
  python -m src.cli serve [--host H] [--port P]
  python -m src.cli tasks [--limit N]
  python -m src.cli show TASK_ID
  python -m src.cli metrics
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from config.settings import get_settings
from src.app import SolomonApp
from src.models.dataclasses import Task, new_task_id


def _print_task(task: Task, result) -> None:
    print("=" * 72)
    print(f"Task:      {task.id}")
    print(f"Tier:      {result.tier}" + (f" (escalated from {result.escalated_from})" if result.escalated_from else ""))
    print(f"Type:      {task.task_type}")
    print(f"Complexity: {task.complexity_score:.3f}")
    print(f"Latency:   {result.total_latency:.2f}s")
    print(f"Cost:      ${result.total_cost:.4f}")
    print(f"Tokens:    {result.tokens_prompt} prompt / {result.tokens_completion} completion")
    print(f"Iterations: {result.iterations}")
    print(f"Techniques: {', '.join(result.techniques_used)}")
    if result.failed:
        print(f"FAILED: {result.error}")
    print("-" * 72)
    print("ANSWER:")
    print(result.final_answer)
    print("=" * 72)
    for i, c in enumerate(result.critique_history, 1):
        verdict = getattr(c, "verdict", None)
        verdict = verdict.value if hasattr(verdict, "value") else str(verdict)
        print(f"  Critique #{i}: {verdict} (confidence {c.confidence:.2f})")
        for issue in c.issues[:5]:
            print(f"    - {issue}")


async def _cmd_chat(args) -> int:
    settings = get_settings()
    solomon = SolomonApp(settings)
    try:
        task = Task(id=new_task_id(), original_prompt=args.message)
        result = await solomon.router.route(task, mode=args.tier)
        solomon.dashboard.record_task(result)
        solomon.trace_store.upsert_task(task)
        solomon.trace_store.finish_task(result)
        _print_task(task, result)
        return 0 if not result.failed else 1
    finally:
        await solomon.close()


def _cmd_serve(args) -> int:
    import uvicorn

    from src.server import create_app

    settings = get_settings()
    settings.server_host = args.host
    settings.server_port = args.port
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_tasks(args) -> int:
    settings = get_settings()
    store = __import__("src.monitoring.trace_store", fromlist=["TraceStore"]).TraceStore(
        settings.db_path
    )
    try:
        for t in store.list_tasks(args.limit):
            status = "FAILED" if t.get("failed") else t.get("status", "?")
            print(
                f"{t['id']}  {t.get('tier', '?'):>9}  {status:>8}  "
                f"{t.get('latency', 0):8.2f}s  {t['prompt'][:60]}"
            )
    finally:
        store.close()
    return 0


def _cmd_show(args) -> int:
    settings = get_settings()
    from src.monitoring.trace_store import TraceStore

    store = TraceStore(settings.db_path)
    try:
        task = store.get_task(args.task_id)
        if task is None:
            print(f"Task {args.task_id} not found")
            return 1
        print(json.dumps(task, indent=2))
        if args.events:
            print("\nEVENTS:")
            for ev in store.get_task_events(args.task_id):
                print(
                    f"  [{ev['ts']:.3f}] {ev['phase']:<28} {ev['event_type']:<12} "
                    f"{json.dumps(ev['data'])[:120]}"
                )
    finally:
        store.close()
    return 0


def _cmd_metrics(args) -> int:
    settings = get_settings()
    from src.monitoring.trace_store import TraceStore
    from src.monitoring.dashboard import MonitoringDashboard

    store = TraceStore(settings.db_path)
    dashboard = MonitoringDashboard()
    try:
        for t in store.list_tasks(1000):
            if t.get("status") == "done":
                from src.models.dataclasses import AgentResult

                dashboard.record_task(
                    AgentResult(
                        task_id=t["id"],
                        final_answer=t.get("final_answer") or "",
                        reasoning_trace="",
                        tier=t.get("tier") or "lightning",
                        iterations=t.get("iterations") or 0,
                        total_cost=t.get("cost") or 0.0,
                        total_latency=t.get("latency") or 0.0,
                        techniques_used=t.get("techniques") or [],
                        tokens_prompt=t.get("tokens_prompt") or 0,
                        tokens_completion=t.get("tokens_completion") or 0,
                        failed=bool(t.get("failed")),
                        error=t.get("error"),
                    )
                )
        print(json.dumps(dashboard.get_dashboard(), indent=2))
    finally:
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solomon", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="Run a single task through the system")
    p_chat.add_argument("message", help="The task prompt")
    p_chat.add_argument(
        "--tier",
        default="auto",
        choices=["auto", "lightning", "deep", "omega"],
        help="Force a tier (default: auto via router)",
    )
    p_chat.set_defaults(func=lambda a: asyncio.run(_cmd_chat(a)))

    p_serve = sub.add_parser("serve", help="Start the web server (chat UI + API)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_tasks = sub.add_parser("tasks", help="List stored tasks")
    p_tasks.add_argument("--limit", type=int, default=20)
    p_tasks.set_defaults(func=_cmd_tasks)

    p_show = sub.add_parser("show", help="Show a stored task")
    p_show.add_argument("task_id")
    p_show.add_argument("--events", action="store_true", help="Include trace events")
    p_show.set_defaults(func=_cmd_show)

    p_metrics = sub.add_parser("metrics", help="Show dashboard metrics")
    p_metrics.set_defaults(func=_cmd_metrics)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
