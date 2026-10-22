"""FastAPI server: REST API + WebSocket trace streaming + static chat UI."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.app import SolomonApp
from src.models.dataclasses import MODES
from src.v2.budget import VALID_MODES

STATIC_DIR = Path(__file__).parent / "web" / "static"


class ChatRequest(BaseModel):
    message: str
    task_id: str | None = None
    mode: str = "auto"  # auto | lightning | deep | omega | v2
    v2_mode: str | None = None  # quick | standard | deep (only for mode=v2)


def create_app(app: SolomonApp | None = None) -> FastAPI:
    solomon = app or SolomonApp()

    api = FastAPI(title="Solomon — Three-Tier Intelligence Agent", version="0.1.0")
    api.state.solomon = solomon

    @api.on_event("startup")
    async def _startup() -> None:
        solomon.event_bus.subscribe(
            lambda ev: _queue_ws_event(solomon, ev)
        )

    @api.on_event("shutdown")
    async def _shutdown() -> None:
        await solomon.close()

    # ---------- REST ----------

    @api.post("/api/chat")
    async def chat(req: ChatRequest) -> JSONResponse:
        if not req.message.strip():
            return JSONResponse({"error": "message is empty"}, status_code=400)
        mode = req.mode if req.mode in MODES else "auto"
        v2_mode = req.v2_mode if req.v2_mode in VALID_MODES else None
        task = solomon.submit_task(
            req.message.strip(), req.task_id, mode=mode, v2_mode=v2_mode
        )
        return JSONResponse(
            {"task_id": task.id, "status": "running", "mode": mode, "v2_mode": v2_mode}
        )

    @api.post("/api/tasks/{task_id}/finish")
    async def finish_now(task_id: str) -> JSONResponse:
        ok = solomon.request_finish_now(task_id)
        if not ok:
            return JSONResponse(
                {"error": "task not running or already finished", "task_id": task_id},
                status_code=409,
            )
        return JSONResponse({"task_id": task_id, "status": "finishing"})

    @api.post("/api/tasks/{task_id}/stop")
    async def stop(task_id: str) -> JSONResponse:
        ok = solomon.request_stop(task_id)
        if not ok:
            return JSONResponse(
                {"error": "task not running or already finished", "task_id": task_id},
                status_code=409,
            )
        return JSONResponse({"task_id": task_id, "status": "stopping"})

    @api.get("/api/tasks/{task_id}/state")
    async def state(task_id: str) -> JSONResponse:
        st = solomon.task_state(task_id)
        if st is None:
            return JSONResponse(
                {"task_id": task_id, "running": False, "state": None}, status_code=200
            )
        return JSONResponse({"task_id": task_id, "running": True, **st})

    @api.get("/api/tasks")
    async def list_tasks(limit: int = 50) -> JSONResponse:
        return JSONResponse({"tasks": solomon.trace_store.list_tasks(limit)})

    @api.get("/api/tasks/{task_id}")
    async def get_task(task_id: str) -> JSONResponse:
        task = solomon.trace_store.get_task(task_id)
        if task is None:
            return JSONResponse({"error": "task not found"}, status_code=404)
        events = solomon.trace_store.get_task_events(task_id)
        return JSONResponse({"task": task, "events": events})

    @api.get("/api/tasks/{task_id}/trace")
    async def get_trace(task_id: str) -> JSONResponse:
        return JSONResponse(
            {"task_id": task_id, "events": solomon.trace_store.get_task_events(task_id)}
        )

    @api.get("/api/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse(
            {
                "dashboard": solomon.dashboard.get_dashboard(),
                "cost": solomon.llm.cost_tracker.snapshot(),
            }
        )

    @api.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # ---------- WebSocket: live trace for a task ----------

    @api.websocket("/ws/{task_id}")
    async def ws_trace(ws: WebSocket, task_id: str) -> None:
        await ws.accept()
        solomon.register_ws(task_id, ws)
        try:
            # Replay history first.
            history = solomon.trace_store.get_task_events(task_id)
            if history:
                await ws.send_json({"type": "history", "events": history})
            task = solomon.trace_store.get_task(task_id)
            await ws.send_json({"type": "task", "task": task})
            await ws.send_json({"type": "ping"})
            while True:
                # Keep the socket alive; client may send pings.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            solomon.unregister_ws(task_id, ws)

    # ---------- static UI ----------

    api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return api


def _queue_ws_event(solomon: SolomonApp, event) -> None:
    """Bridge the event bus to per-task WebSocket subscribers."""
    import asyncio

    async def _send() -> None:
        await solomon.broadcast(
            event.task_id,
            {
                "type": "event",
                "event": {
                    "ts": event.ts,
                    "phase": event.phase,
                    "event_type": event.event_type,
                    "data": event.data,
                },
            },
        )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send())
    except RuntimeError:
        pass


def main() -> None:
    import uvicorn

    from config.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        create_app(),
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
