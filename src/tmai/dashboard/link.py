"""The trainer's side of the dashboard: `DashboardHook`.

The dashboard runs in its own process (`tmai dashboard`) so serving the web
page never delays a 20 Hz control step. The two talk over localhost UDP:
one JSON object per datagram, fire-and-forget, so the trainer never waits
on the dashboard and keeps running if the dashboard is closed or restarted.

    trainer -> dashboard (dashboard.link_port):
        {"type": "hello", "run_dir", "state", "paused", "params": [ParamSpec...],
                          "episodes", "updates", "env_steps", "best_ms"}          every second
        {"type": "tel", "speed_kmh", "steer", "gas", "brake", "gear", "rpm",
                        "race_time_ms", "progress", "progress_m", "reward",
                        "splits_ms", "best_splits_ms"}                           up to 10 Hz
        {"type": "metrics", "step", "t", "values": {name: number}}
        {"type": "episode", "row": {...}}
        {"type": "ack", "id", "ok", "error", "note"}
    dashboard -> trainer (back to the address the trainer sent from):
        {"type": "cmd", "id", "kind", "name", "value"}
        kind: pause | resume | reset_track | save_checkpoint | set_param (name, value)

Stopping is not part of this link: the dashboard calls
tmai.run.stop.request_stop(run_dir), the same file `tmai stop` writes, and
reads the shutdown's progress back with read_status(run_dir).

The dashboard re-sends a command until it is acked, so ids are de-duplicated
here. Metrics and episodes are also appended to `metrics.jsonl` and
`episodes.jsonl` in the run folder; that is where the dashboard loads history
from when it starts, so curves survive restarts of either side.
"""

from __future__ import annotations

import json
import math
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tmai.run.hooks import TrainerHooks

HELLO_EVERY_S = 1.0
TELEMETRY_EVERY_S = 0.1
METRICS_EVERY_S = 1.0
MAX_DATAGRAM = 60_000


def encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":"), default=_plain).encode()


def decode(data: bytes) -> dict | None:
    try:
        msg = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None
    return msg if isinstance(msg, dict) and "type" in msg else None


def udp_socket(bind_port: int | None = None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        # Windows reports "connection reset" on the next receive after sending
        # to a port nobody listens on (dashboard closed). Turn that off.
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)
    sock.bind(("127.0.0.1", bind_port or 0))
    sock.setblocking(False)
    return sock


def drop_non_finite(row: dict) -> dict:
    """NaN/inf -> None, in place. Episodes before the first learner update have
    NaN losses, and NaN is not valid JSON: the browser would reject the page."""
    for k, v in row.items():
        if isinstance(v, float) and not math.isfinite(v):
            row[k] = None
    return row


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue   # a line cut short by a crash
    return rows


@dataclass(frozen=True)
class ParamSpec:
    """A hyperparameter the user may change while training runs."""
    name: str
    value: float
    min: float
    max: float
    integer: bool = False
    log: bool = False          # the UI uses a log-scale slider (learning rates)
    help: str = ""

    def coerce(self, value: Any) -> float | int:
        v = float(value)
        if not math.isfinite(v) or not self.min <= v <= self.max:
            raise ValueError(f"{self.name} must be between {self.min:g} and {self.max:g}")
        if self.integer:
            if v != int(v):
                raise ValueError(f"{self.name} must be a whole number")
            return int(v)
        return v


# What the dashboard can change on a running tmai.rl.train.Trainer:
# name -> (spec bounds and help, read the current value, apply a new one).
def _optimizers(t):
    return [getattr(t.agent, n) for n in ("critic_opt", "actor_opt", "alpha_opt") if hasattr(t.agent, n)]


def _set_lr(t, v):
    for opt in _optimizers(t):
        for group in opt.param_groups:
            group["lr"] = v


def _set_demo_ratio(t, v):
    t.demo_n = int(round(t.batch_size * v))


PARAMS = {
    "learning_rate": (dict(min=1e-6, max=1e-2, log=True,
                           help="Adam step size for actor, critics and temperature"),
                      lambda t: _optimizers(t)[0].param_groups[0]["lr"], _set_lr),
    "updates_per_step": (dict(min=0, max=16, integer=True,
                              help="gradient updates per driving step (UTD); 0 stops learning"),
                         lambda t: t.utd, lambda t, v: setattr(t, "utd", v)),
    "target_entropy": (dict(min=-10.0, max=2.0, help="SAC exploration target; higher explores more"),
                       lambda t: t.agent.target_entropy, lambda t, v: setattr(t.agent, "target_entropy", v)),
    "demo_ratio": (dict(min=0.0, max=0.9, help="share of each batch drawn from your demo laps"),
                   lambda t: t.demo_n / t.batch_size if t.demo is not None else None, _set_demo_ratio),
}


def param_specs(trainer) -> list[ParamSpec]:
    specs = []
    for name, (bounds, get, _) in PARAMS.items():
        try:
            value = get(trainer)
        except (AttributeError, IndexError, TypeError):
            continue   # this trainer has no such setting
        if value is not None:
            specs.append(ParamSpec(name, float(value), **bounds))
    return specs


class DashboardHook(TrainerHooks):
    """Put it in the trainer's HookList, after ShutdownHook:

        HookList(ShutdownHook(cfg), DashboardHook.from_config(cfg))

    It needs nothing from the trainer beyond the hook calls: on_start gives it
    the trainer, and it reads metrics, splits and counters from there and
    applies hyperparameter changes to it directly. Commands it handles:
    pause/resume (should_pause, plus should_reset_episode so a pause takes
    effect mid-lap), reset_track (should_reset_episode), set_param (applied in
    on_step), save_checkpoint (between attempts, in on_episode). The stop
    button does not come through here: the dashboard writes stop.json with
    tmai.run.stop.request_stop, which ShutdownHook already watches."""

    def __init__(self, run_dir: str | Path, port: int, tensorboard: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = None
        self._addr = ("127.0.0.1", port)
        self._sock = udp_socket()
        self._params: dict[str, ParamSpec] = {}
        self._paused = False
        self._reset = False
        self._save = False
        self._stopped = False
        self._set: dict[str, float] = {}
        self._seen_ids: deque[str] = deque(maxlen=256)
        self._last_hello = self._last_tel = self._last_metrics = 0.0
        self._metrics_step = -1
        self._last_tel_obj = None
        self._metrics_file = _open_append(self.run_dir / "metrics.jsonl")
        self._episodes_file = _open_append(self.run_dir / "episodes.jsonl")
        self._tb = _tensorboard_writer(self.run_dir / "tensorboard") if tensorboard else None

    @classmethod
    def from_config(cls, cfg: dict) -> "DashboardHook":
        run, dash = cfg["run"], cfg["dashboard"]
        return cls(Path(run["output_dir"]) / run["name"], dash["link_port"], dash["tensorboard"])

    # ---- TrainerHooks --------------------------------------------------------

    def on_start(self, trainer) -> None:
        self.trainer = trainer
        self._params = {s.name: s for s in param_specs(trainer)}
        self._hello()

    def on_step(self, tel, info=None) -> None:
        self._receive()
        self._last_tel_obj = tel
        if self._set:
            self._apply_params()
        now = time.monotonic()
        if now - self._last_hello >= HELLO_EVERY_S:
            self._hello()
        if now - self._last_metrics >= METRICS_EVERY_S:
            self._last_metrics = now
            self._send_metrics()
        if now - self._last_tel >= TELEMETRY_EVERY_S:
            self._last_tel = now
            self._send_telemetry(tel, info)

    def should_reset_episode(self) -> bool:
        self._receive()
        if self._reset:
            self._reset = False
            return True
        return self._paused

    def should_pause(self) -> bool:
        self._receive()
        if time.monotonic() - self._last_hello >= HELLO_EVERY_S:
            self._hello()
        if self._set:
            self._apply_params()
        return self._paused

    def on_episode(self, row: dict) -> None:
        row = drop_non_finite({"t": round(time.time(), 3), **row})
        splits = self._splits()
        if splits is not None:
            row["splits_ms"] = list(splits.current)
        tel = self._last_tel_obj
        if row.get("end") == "finished" and tel is not None and "lap_ms" not in row:
            row["lap_ms"] = getattr(tel, "last_cp_time_ms", 0) or getattr(tel, "race_time_ms", None)
        _write(self._episodes_file, row)
        self._send({"type": "episode", "row": row})
        if self._tb:
            step = int(row.get("env_steps") or row.get("episode") or 0)
            if row.get("reward") is not None:
                self._tb.add_scalar("episode/reward", float(row["reward"]), step)
            if row.get("lap_ms") is not None:
                self._tb.add_scalar("episode/lap_time_s", row["lap_ms"] / 1000, step)
        self._send_metrics()
        if self._save and self.trainer is not None:
            self._save = False
            self.trainer.save_checkpoint()

    def on_stop(self, trainer=None) -> None:
        self._stopped = True
        self._hello()
        self.close()

    def close(self) -> None:
        for f in (self._metrics_file, self._episodes_file):
            if f and not f.closed:
                f.close()
        if self._tb:
            self._tb.close()
            self._tb = None
        self._sock.close()

    # ---- what gets sent ----------------------------------------------------------

    def _state(self) -> str:
        t = self.trainer
        if self._stopped:
            return "stopped"
        if self._paused:
            return "paused"
        if t is None:
            return "starting"
        try:
            if t.online.transitions < t.warmup:
                return "warming up"
        except AttributeError:
            pass
        return "training"

    def _counters(self) -> dict:
        t, out = self.trainer, {}
        for key, get in (("episodes", lambda: t.episodes), ("updates", lambda: t.agent.updates),
                         ("env_steps", lambda: t.env_steps_before + t.online.transitions),
                         ("best_ms", lambda: t.best_ms)):
            try:
                out[key] = get()
            except AttributeError:
                pass
        return out

    def _hello(self) -> None:
        self._last_hello = time.monotonic()
        self._send({"type": "hello", "run_dir": str(self.run_dir.resolve()), "state": self._state(),
                    "paused": self._paused, "params": [asdict(s) for s in self._params.values()],
                    **self._counters()})

    def _splits(self):
        try:
            return self.trainer.tracker.splits
        except AttributeError:
            return None

    def _send_telemetry(self, tel, info) -> None:
        msg = {"type": "tel"}
        for k in ("speed_kmh", "steer", "gas", "brake", "gear", "rpm", "race_time_ms"):
            v = getattr(tel, k, None)
            if v is not None:
                msg[k] = v
        for k in ("progress", "progress_m", "reward"):
            v = getattr(info, k, None)
            if v is not None:
                msg[k] = v
        splits = self._splits()
        if splits is not None:
            msg["splits_ms"] = list(splits.current)
            msg["best_splits_ms"] = list(splits.best)
        self._send(msg)

    def _send_metrics(self) -> None:
        t = self.trainer
        try:
            step, metrics = t.agent.updates, dict(t.metrics)
        except AttributeError:
            return
        if step == self._metrics_step or not metrics:
            return
        self._metrics_step = step
        values = {}
        for k, v in metrics.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                values[k] = v
        if not values:
            return
        row = {"step": int(step), "t": round(time.time(), 3), **values}
        _write(self._metrics_file, row)
        self._send({"type": "metrics", "step": row["step"], "t": row["t"], "values": values})
        if self._tb:
            for k, v in values.items():
                self._tb.add_scalar(k, v, int(step))

    def _send(self, msg: dict) -> None:
        data = encode(msg)
        if len(data) > MAX_DATAGRAM or self._sock.fileno() < 0:
            return
        try:
            self._sock.sendto(data, self._addr)
        except OSError:
            pass   # dashboard not running; the jsonl files keep the history

    # ---- what comes back -----------------------------------------------------------

    def _apply_params(self) -> None:
        pending, self._set = self._set, {}
        for name, value in pending.items():
            PARAMS[name][2](self.trainer, value)
            self._params[name] = replace(self._params[name], value=float(value))
        self._hello()

    def _receive(self) -> None:
        while self._sock.fileno() >= 0:
            try:
                data, addr = self._sock.recvfrom(65536)
            except OSError:   # nothing waiting (BlockingIOError) or socket closed
                return
            msg = decode(data)
            if msg and msg["type"] == "cmd":
                self._handle(msg, addr)

    def _handle(self, msg: dict, addr) -> None:
        cid = str(msg.get("id", ""))
        if cid in self._seen_ids:
            self._ack(cid, addr, True)   # our first ack was lost
            return
        kind, name, value = msg.get("kind"), msg.get("name"), msg.get("value")
        error, note = "", ""
        if kind == "pause":
            self._paused = True
        elif kind == "resume":
            self._paused = False
        elif kind == "reset_track":
            self._reset = True
        elif kind == "save_checkpoint":
            self._save = True
            note = "saves when this attempt ends"
        elif kind == "set_param":
            spec = self._params.get(name)
            if spec is None or self.trainer is None:
                error = f"{name!r} is not adjustable in this run"
            else:
                try:
                    self._set[name] = spec.coerce(value)
                except (TypeError, ValueError) as exc:
                    error = str(exc)
        else:
            error = f"unknown command {kind!r}"
        if not error:
            self._seen_ids.append(cid)
        self._ack(cid, addr, not error, error, note)
        if not error and kind in ("pause", "resume"):
            self._hello()

    def _ack(self, cid: str, addr, ok: bool, error: str = "", note: str = "") -> None:
        try:
            self._sock.sendto(encode({"type": "ack", "id": cid, "ok": ok, "error": error, "note": note}), addr)
        except OSError:
            pass


def _open_append(path: Path):
    cut_short = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            cut_short = f.read(1) != b"\n"
    f = open(path, "a", encoding="utf-8")
    if cut_short:
        f.write("\n")   # finish a line cut short by a crash
    return f


def _write(f, row: dict) -> None:
    if f and not f.closed:
        f.write(json.dumps(row, default=_plain) + "\n")
        f.flush()


def _plain(v: Any) -> Any:
    if hasattr(v, "item"):        # numpy / torch scalars
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (set, tuple)):
        return list(v)
    raise TypeError(f"cannot send {type(v).__name__}")


def _tensorboard_writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:   # torch or tensorboard missing: the jsonl files have everything
        return None
    return SummaryWriter(str(path))
