"""The dashboard's side of the link: listens for the trainer on localhost UDP,
keeps the run's history and live values, and sends the user's commands back.
The web server only reads `snapshot()` / `events_since()` and calls `submit()`."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from pathlib import Path

from tmai.dashboard.link import ParamSpec, decode, drop_non_finite, encode, read_jsonl, udp_socket
from tmai.run.stop import read_status, request_stop

LINK_COMMANDS = ("pause", "resume", "reset_track", "save_checkpoint", "set_param")

HISTORY_POINTS = 20_000   # per series kept in memory; the jsonl files keep everything
EVENT_BUFFER = 4_000      # recent events a slow browser can catch up on
TRAINER_TIMEOUT_S = 3.0   # no hello for this long = trainer not running
RESEND_S = 0.3
COMMAND_EXPIRY_S = 5.0
STATUS_EVERY_S = 0.5      # how often control/status.json is re-read


class RunState:
    def __init__(self, run_dir: str | Path | None, link_port: int):
        self._lock = threading.Lock()
        self._sock = udp_socket(link_port)
        self._sock.settimeout(0.05)
        self.link_port = self._sock.getsockname()[1]
        self._trainer_addr = None
        self._last_hello = 0.0
        self._seq = 0
        self._events: deque[dict] = deque(maxlen=EVENT_BUFFER)
        self._telemetry: dict = {}
        self._telemetry_seq = 0
        self._trainer: dict = {"state": "not connected", "paused": False}
        self._run_status: dict = {"state": "unknown"}
        self._last_status_read = 0.0
        self._params: list[dict] = []
        self._outbox: dict[str, dict] = {}   # unacked commands by id
        self._load(Path(run_dir) if run_dir else None)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="dashboard-link", daemon=True)
        self._thread.start()

    # ---- read side ---------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "seq": self._seq,
                "run_dir": str(self._run_dir) if self._run_dir else None,
                "connected": self._connected(),
                "trainer": dict(self._trainer),
                "run_status": dict(self._run_status),
                "params": list(self._params),
                "telemetry": dict(self._telemetry),
                "telemetry_seq": self._telemetry_seq,
                "scalars": {k: list(v) for k, v in self._scalars.items()},
                "episodes": list(self._episodes),
                "best": self._best,
            }

    def events_since(self, seq: int) -> list[dict] | None:
        """Events after `seq`, or None when some were dropped and the caller
        should take a fresh snapshot."""
        with self._lock:
            if seq >= self._seq:
                return []
            if not self._events or self._events[0]["seq"] > seq + 1:
                return None
            return [e for e in self._events if e["seq"] > seq]

    def latest_telemetry(self) -> tuple[int, dict]:
        with self._lock:
            return self._telemetry_seq, dict(self._telemetry)

    # ---- commands ------------------------------------------------------------

    def submit(self, kind: str, name: str | None = None, value=None) -> dict:
        """Send a command to the trainer. Raises ValueError when it cannot be
        valid; the trainer re-checks it and acks or rejects it."""
        if kind == "stop":
            with self._lock:
                if self._run_dir is None:
                    raise ValueError("no run loaded")
                request_stop(self._run_dir, reason="dashboard button", source="dashboard")
                cmd = {"id": uuid.uuid4().hex[:12], "kind": "stop", "name": None, "value": None}
                self._emit("command", {**cmd, "status": "done", "note": "the trainer saves everything and exits"})
            return cmd
        if kind not in LINK_COMMANDS:
            raise ValueError(f"unknown command {kind!r}")
        with self._lock:
            if not self._connected():
                raise ValueError("the trainer is not running")
            if kind == "set_param":
                spec = next((p for p in self._params if p["name"] == name), None)
                if spec is None:
                    raise ValueError(f"{name!r} is not adjustable in this run")
                value = ParamSpec(**spec).coerce(value)
            else:
                name = value = None
            cmd = {"type": "cmd", "id": uuid.uuid4().hex[:12], "kind": kind, "name": name, "value": value}
            self._outbox[cmd["id"]] = {"msg": cmd, "first": time.monotonic(), "last": 0.0}
            self._emit("command", {"id": cmd["id"], "kind": kind, "name": name, "value": value,
                                   "status": "sent"})
        self._flush_outbox()
        return cmd

    def report_path(self) -> Path | None:
        """The run report, if the shutdown has written one."""
        with self._lock:
            run_dir = self._run_dir
            reported = self._run_status.get("report")
        if run_dir is None:
            return None
        for path in ([Path(reported)] if reported else []) + [run_dir / "report" / "report.html"]:
            try:
                if path.resolve().is_relative_to(run_dir) and path.is_file():
                    return path
            except OSError:
                continue
        return None

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._sock.close()

    # ---- internals -----------------------------------------------------------

    def _connected(self) -> bool:
        return self._trainer_addr is not None and time.monotonic() - self._last_hello < TRAINER_TIMEOUT_S

    def _load(self, run_dir: Path | None) -> None:
        """(Re)load history from a run folder. Call with the lock held or before start."""
        self._run_dir = run_dir.resolve() if run_dir else None
        self._scalars: dict[str, deque] = {}
        self._episodes: deque[dict] = deque(maxlen=HISTORY_POINTS)
        self._best: dict | None = None
        if run_dir:
            for row in read_jsonl(run_dir / "metrics.jsonl"):
                self._add_metrics(row.get("step", 0), {k: v for k, v in row.items() if k not in ("step", "t")})
            for row in read_jsonl(run_dir / "episodes.jsonl"):
                self._add_episode(row)

    # A step or episode number lower than the last one means the trainer
    # started over (--fresh) in the same run folder: show only the new run.

    def _add_metrics(self, step: int, values: dict) -> None:
        for k, v in values.items():
            if isinstance(v, (int, float)):
                series = self._scalars.setdefault(k, deque(maxlen=HISTORY_POINTS))
                if series and step < series[-1][0]:
                    series.clear()
                series.append((step, v))

    def _add_episode(self, row: dict) -> bool:
        """Returns True when this row started a new run."""
        drop_non_finite(row)   # history written before this was fixed may hold NaN
        restarted = bool(self._episodes) and (row.get("episode") or 0) <= (self._episodes[-1].get("episode") or 0)
        if restarted:
            self._episodes.clear()
            self._best = None
        self._episodes.append(row)
        lap = row.get("lap_ms")
        if lap is not None and (self._best is None or lap < self._best["lap_ms"]):
            self._best = {"lap_ms": lap, "splits_ms": row.get("splits_ms") or [],
                          "episode": row.get("episode")}
        return restarted

    def _emit(self, kind: str, data: dict) -> None:
        self._seq += 1
        self._events.append({"seq": self._seq, "kind": kind, "data": data})

    def _loop(self) -> None:
        was_connected = False
        while self._running:
            try:
                data, addr = self._sock.recvfrom(65536)
            except (TimeoutError, BlockingIOError, InterruptedError):
                data = None
            except OSError:
                if not self._running:
                    return
                time.sleep(0.05)
                data = None
            if data:
                msg = decode(data)
                if msg:
                    with self._lock:
                        self._on_message(msg, addr)
            with self._lock:
                now_connected = self._connected()
                if now_connected != was_connected:
                    if not now_connected:
                        self._trainer = {**self._trainer, "state": "not connected"}
                    self._emit("trainer", {"connected": now_connected, **self._trainer})
                    was_connected = now_connected
            self._flush_outbox()
            self._poll_status()

    def _poll_status(self) -> None:
        now = time.monotonic()
        if now - self._last_status_read < STATUS_EVERY_S or self._run_dir is None:
            return
        self._last_status_read = now
        try:
            status = read_status(self._run_dir)
        except (OSError, ValueError):
            return
        with self._lock:
            if status != self._run_status:
                self._run_status = status
                self._emit("run_status", status)

    def _on_message(self, msg: dict, addr) -> None:
        kind = msg["type"]
        if kind == "hello":
            self._trainer_addr, self._last_hello = addr, time.monotonic()
            run_dir = Path(msg.get("run_dir") or "")
            if msg.get("run_dir") and run_dir.resolve() != self._run_dir:
                self._load(run_dir)
                self._emit("reload", {})
            info = {k: msg.get(k) for k in ("state", "paused", "episodes", "updates", "env_steps", "best_ms")}
            if info != {k: self._trainer.get(k) for k in info}:
                self._trainer = info
                self._emit("trainer", {"connected": True, **info})
            if msg.get("params") != self._params:
                self._params = msg.get("params") or []
                self._emit("params", {"params": self._params})
        elif kind == "tel":
            msg.pop("type")
            self._telemetry = msg
            self._telemetry_seq += 1
        elif kind == "metrics":
            values, step = msg.get("values") or {}, msg.get("step", 0)
            if any(self._scalars.get(k) and step < self._scalars[k][-1][0] for k in values):
                self._emit("reload", {})
            self._add_metrics(step, values)
            self._emit("metrics", {"step": msg.get("step", 0), "values": values})
        elif kind == "episode":
            row = msg.get("row") or {}
            if self._add_episode(row):
                self._emit("reload", {})
            self._emit("episode", {"row": row, "best": self._best})
        elif kind == "ack":
            entry = self._outbox.pop(str(msg.get("id")), None)
            if entry:
                cmd = entry["msg"]
                self._emit("command", {"id": cmd["id"], "kind": cmd["kind"], "name": cmd["name"],
                                       "value": cmd["value"], "status": "done" if msg.get("ok") else "rejected",
                                       "error": msg.get("error", ""), "note": msg.get("note", "")})

    def _flush_outbox(self) -> None:
        now = time.monotonic()
        with self._lock:
            addr = self._trainer_addr
            for cid, entry in list(self._outbox.items()):
                if now - entry["first"] > COMMAND_EXPIRY_S:
                    del self._outbox[cid]
                    cmd = entry["msg"]
                    self._emit("command", {"id": cid, "kind": cmd["kind"], "name": cmd["name"],
                                           "value": cmd["value"], "status": "rejected",
                                           "error": "the trainer did not answer"})
                elif addr and now - entry["last"] >= RESEND_S:
                    entry["last"] = now
                    try:
                        self._sock.sendto(encode(entry["msg"]), addr)
                    except OSError:
                        pass

