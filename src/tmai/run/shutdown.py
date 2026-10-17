"""ShutdownHook: the trainer hook that makes a run safe to stop and resume.

    hooks = ShutdownHook(cfg)            # put it first in a HookList
    trainer.train(..., hooks=hooks)      # however the trainer takes hooks

What it does, by hook point:
  on_start    installs Ctrl+C / SIGTERM / stop.json handling on trainer.stop,
              restores the replay buffer from the last snapshot when the trainer
              resumed from a checkpoint, and marks the run as AI driving for the
              TMAI Ghost plugin
  on_step     checks for a stop request (twice a second) and reports finished
              lap times to the TMAI Ghost plugin
  on_episode  records each new best lap with its splits (best_laps.jsonl), keeps
              a best.pt copy of those weights, and takes periodic replay
              snapshots in the background
  on_stop     final checkpoint, final replay snapshot, run report, best-lap
              replay copy; each step runs even if an earlier one failed

Call `prepare_resume(run_dir)` from tmai.run.checkpoint before building the
Trainer so a damaged latest.pt is swapped for the newest intact copy."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path

from tmai.run import atomic, ghost, snapshot
from tmai.run.checkpoint import CheckpointStore
from tmai.run.hooks import TrainerHooks
from tmai.run.report import build_report
from tmai.run.stop import STOPPED, StopController

AGENT_LAPS_KEEP = 20


def openplanet_dir() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()) / "OpenplanetNext"


class AgentLaps:
    """Tells the TMAI Ghost plugin which finishes were the agent's, through
    OpenplanetNext/PluginStorage/TMAI_Ghost/ai_driving.txt (one lap time in ms
    per line). The file only exists while training runs."""

    def __init__(self, path: Path | None = None):
        self.path = path or openplanet_dir() / "PluginStorage" / "TMAI_Ghost" / "ai_driving.txt"
        self.laps: list[int] = []
        self.enabled = self.path.parent.parent.parent.is_dir()   # Openplanet installed here

    def _write(self) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(f"{ms}\n" for ms in self.laps))   # no fsync: must stay fast
            os.replace(tmp, self.path)
        except OSError:
            pass   # the replay is a bonus; never disturb driving for it

    def start(self) -> None:
        self.laps = []
        self._write()

    def add(self, lap_ms: int) -> None:
        self.laps = (self.laps + [int(lap_ms)])[-AGENT_LAPS_KEEP:]
        self._write()

    def stop(self) -> None:
        if self.enabled:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


class ShutdownHook(TrainerHooks):
    def __init__(self, cfg: dict, log=print, agent_laps: AgentLaps | None = None, replays_root: Path | None = None):
        self.cfg, self.log = cfg, log
        run = cfg["run"]
        self.run_dir = Path(run["output_dir"]) / run["name"]
        self.track = run["track"]
        self.track_dir = Path(run["output_dir"]) / "tracks" / self.track
        self.replay_dir = self.run_dir / "replay"
        self.snapshot_every = cfg["replay"]["snapshot_every_min"] * 60
        self.checkpoints = CheckpointStore(self.run_dir)
        self.agent_laps = agent_laps or AgentLaps()
        self.replays_root = replays_root
        self.controller: StopController | None = None
        self.trainer = None
        self._best_ms: int | None = None
        self._last_snapshot = time.monotonic()
        self._snap_thread: threading.Thread | None = None
        self._snap_error: BaseException | None = None
        self.steps: list[dict] = []

    # -- start -------------------------------------------------------------------
    def on_start(self, trainer) -> None:
        self.trainer = trainer
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.controller = StopController(self.run_dir, trainer.stop, self.log)
        try:
            self.controller.install()
        except ValueError:   # not on the main thread (tests, embedded use): stop.json still works
            self.controller.set_status("running", phase="training")
        self._best_ms = trainer.best_ms
        resumed = getattr(trainer, "env_steps_before", 0) > 0
        if resumed and trainer.online.transitions == 0:
            self._restore_replay(trainer)
        self.agent_laps.start()

    def _restore_replay(self, trainer) -> None:
        try:
            manifest = snapshot.load_replay(trainer.online, self.replay_dir, log=self.log)
        except Exception as exc:   # a bad snapshot must not stop training from starting
            self.log(f"replay snapshot not restored: {exc}")
            return
        if manifest is None:
            return
        # Count only this session's transitions, so env_steps in the checkpoint
        # stays right; skip warm-up because the buffer is already full of data.
        trainer.online.transitions = 0
        trainer.warmup = 0
        self.log(f"restored {trainer.online.size:,} replay entries from {time.ctime(manifest['created'])}")

    # -- during the run ------------------------------------------------------------
    def on_step(self, tel, info) -> None:
        if self.controller is not None:
            self.controller.poll()
        if info is not None and info.done == "finished" and tel is not None:
            self.agent_laps.add(tel.last_cp_time_ms or tel.race_time_ms)

    def on_episode(self, row: dict) -> None:
        t = self.trainer
        if t is not None and t.best_ms is not None and (self._best_ms is None or t.best_ms < self._best_ms):
            self._best_ms = t.best_ms
            self._record_best(row)
        if time.monotonic() - self._last_snapshot >= self.snapshot_every:
            self._snapshot_in_background()

    def _record_best(self, row: dict) -> None:
        t = self.trainer
        splits = list(getattr(t.tracker.splits, "current", []) or [])
        entry = {"episode": int(row.get("episode", t.episodes)), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "wall_time": time.time(), "lap_ms": int(t.best_ms), "splits": [int(x) for x in splits],
                 "env_steps": row.get("env_steps"), "updates": row.get("updates")}
        path = self.run_dir / "best_laps.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            self.checkpoints.archive("best", t.save_checkpoint(), lap_ms=entry["lap_ms"], episode=entry["episode"])
        except Exception as exc:   # keep training; the next best lap tries again
            self.log(f"could not keep best-lap weights: {exc}")

    def _snapshot_in_background(self) -> None:
        if self._snap_thread is not None and self._snap_thread.is_alive():
            return
        self._last_snapshot = time.monotonic()
        t = self.trainer

        def work():
            try:
                started = time.monotonic()
                snapshot.save_replay(t.online, self.replay_dir, lock=t.replay_lock, step=t.online.transitions)
                self.log(f"replay snapshot saved in {time.monotonic() - started:.0f} s")
            except BaseException as exc:   # reported, retried at the next interval
                self._snap_error = exc
                self.log(f"replay snapshot failed: {exc}")

        self._snap_thread = threading.Thread(target=work, name="replay-snapshot", daemon=True)
        self._snap_thread.start()

    # -- stop --------------------------------------------------------------------
    def _step(self, name: str, fn) -> object:
        if self.controller is not None:
            self.controller.set_status("stopping", phase=name, steps=self.steps)
        started = time.monotonic()
        entry = {"name": name, "ok": True, "detail": ""}
        result = None
        try:
            result = fn()
            if isinstance(result, (str, Path)):
                entry["detail"] = str(result)
        except Exception as exc:
            entry.update(ok=False, detail=f"{type(exc).__name__}: {exc}")
            self.log(f"shutdown step '{name}' failed: {exc}")
            self.log(traceback.format_exc(limit=3))
        entry["seconds"] = time.monotonic() - started
        self.steps.append(entry)
        return result

    def on_stop(self, trainer) -> None:
        self.trainer = trainer
        trainer.stop.set()
        if self.controller is not None and self.controller.reason is None:
            self.controller.reason, self.controller.source = "training ended", "trainer"
        self.agent_laps.stop()

        def final_checkpoint():
            path = trainer.save_checkpoint()
            self.checkpoints.archive("final", path, episodes=trainer.episodes, best_ms=trainer.best_ms)
            self.checkpoints.archive("history", path)
            return path

        def final_snapshot():
            if self._snap_thread is not None:
                self._snap_thread.join()
            if trainer.online.size == 0:
                return "replay buffer is empty; nothing to save"
            # If the learner thread outlived its join timeout it may still read
            # the buffer, so keep taking the lock; the driver has stopped.
            return snapshot.save_replay(trainer.online, self.replay_dir, lock=trainer.replay_lock,
                                        step=trainer.online.transitions)

        replay_info: dict = {}

        def best_replay():
            replay_info.update(ghost.preserve_best_replay(self.run_dir, self.track, root=self.replays_root,
                                                          best_ms=trainer.best_ms))
            return replay_info.get("copy") or replay_info.get("detail")

        def report():
            extra = {"track": self.track, "replay": replay_info, "shutdown": self.steps,
                     "stop_reason": self.controller.reason if self.controller else None,
                     "trainer_best_ms": trainer.best_ms}
            return build_report(self.run_dir, self.track_dir, extra)

        self.log("saving: final checkpoint, replay snapshot, run report")
        self._step("final checkpoint", final_checkpoint)
        self._step("replay snapshot", final_snapshot)
        self._step("best lap replay", best_replay)
        report_path = self._step("run report", report)
        if self.controller is not None:
            self.controller.set_status(STOPPED, phase="done", steps=self.steps,
                                       report=str(report_path) if report_path else None)
            self.controller.finish()
        for st in self.steps:
            self.log(f"  {'ok  ' if st['ok'] else 'FAIL'} {st['name']}: {st['detail']}")
        if report_path:
            self.log(f"report: {report_path}")
