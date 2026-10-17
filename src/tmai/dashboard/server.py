"""`tmai dashboard`: the local web page (FastAPI + one WebSocket).

It runs as its own process next to `tmai train`, which reaches it through
DashboardHooks (see link.py). The page gets a full snapshot when it connects,
then a message every `TICK_S` with new events and the latest telemetry.
Commands come back as `POST /api/command`. Only localhost is ever bound.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tmai.dashboard.state import RunState

STATIC_DIR = Path(__file__).parent / "static"
TICK_S = 0.1


class CommandIn(BaseModel):
    kind: str
    name: str | None = None
    value: float | None = None


def create_app(state: RunState):
    app = FastAPI(title="tmai dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/report")
    def report():
        path = state.report_path()
        if path is None:
            raise HTTPException(status_code=404, detail="no run report yet; it is written when training stops")
        return FileResponse(path)

    @app.get("/api/snapshot")
    def snapshot():
        return state.snapshot()

    @app.post("/api/command")
    def command(cmd: CommandIn):
        try:
            sent = state.submit(cmd.kind, cmd.name, cmd.value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "id": sent["id"], "value": sent["value"]}

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            snap = state.snapshot()
            seq, tel_seq = snap["seq"], snap["telemetry_seq"]
            await ws.send_json({"type": "snapshot", **snap})
            while True:
                await asyncio.sleep(TICK_S)
                events = state.events_since(seq)
                if events is None or any(e["kind"] == "reload" for e in events):
                    snap = state.snapshot()
                    seq, tel_seq = snap["seq"], snap["telemetry_seq"]
                    await ws.send_json({"type": "snapshot", **snap})
                    continue
                new_tel_seq, tel = state.latest_telemetry()
                if not events and new_tel_seq == tel_seq:
                    continue
                msg = {"type": "update", "events": events}
                if new_tel_seq != tel_seq:
                    msg["telemetry"], tel_seq = tel, new_tel_seq
                if events:
                    seq = events[-1]["seq"]
                await ws.send_json(msg)
        except WebSocketDisconnect:
            return

    return app


def run_dashboard(cfg: dict, demo: bool = False) -> int:
    import uvicorn

    run, dash = cfg["run"], cfg["dashboard"]
    run_dir = Path(run["output_dir"]) / (f"{run['name']}-demo" if demo else run["name"])
    try:
        state = RunState(run_dir, dash["link_port"])
    except OSError as exc:
        print(f"cannot listen for the trainer on port {dash['link_port']}: {exc}")
        print("Is another `tmai dashboard` already running?")
        return 1
    fake = None
    if demo:
        from tmai.dashboard.fake import FakeTrainer

        fake = FakeTrainer(cfg, run_dir)
        fake.start()
    url = f"http://{dash['host']}:{dash['port']}"
    print(f"dashboard on {url}  (Ctrl+C closes it; training keeps running)")
    if demo:
        print("demo mode: a fake trainer is producing sample data")
    try:
        uvicorn.run(create_app(state), host=dash["host"], port=dash["port"], log_level="warning")
    finally:
        if fake:
            fake.close()
        state.close()
    return 0
