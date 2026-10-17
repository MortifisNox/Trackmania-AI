"""A stand-in trainer for `tmai dashboard --demo`: drives an imaginary car
round an imaginary track and learns at a made-up rate. It has the attributes
DashboardHook reads from tmai.rl.train.Trainer and calls the hooks at the
same points, so the dashboard can be tried and tested without the game."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tmai.dashboard.link import DashboardHook, read_jsonl
from tmai.run.stop import STOPPED, StopController
from tmai.track.splits import SplitTracker

TRACK_M = 1200.0
CHECKPOINTS_M = (240.0, 480.0, 720.0, 960.0)
SPEEDUP = 4            # simulated seconds per real second, so laps come quickly


class _Opt:
    def __init__(self, lr: float):
        self.param_groups = [{"lr": lr}]


class FakeTrainer:
    def __init__(self, cfg: dict, run_dir: Path, step_hz: float | None = None):
        t = cfg["train"]
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.hz = step_hz or cfg["env"]["step_hz"]
        self.rng = np.random.default_rng(cfg["run"]["seed"])
        done = read_jsonl(self.run_dir / "episodes.jsonl")   # carry on from an earlier demo
        last = done[-1] if done else {}
        # The Trainer attributes DashboardHook uses.
        self.agent = SimpleNamespace(updates=last.get("updates", 0), target_entropy=-3.0,
                                     critic_opt=_Opt(t["learning_rate"]), actor_opt=_Opt(t["learning_rate"]),
                                     alpha_opt=_Opt(t["learning_rate"]))
        self.utd, self.batch_size, self.warmup = t["updates_per_step"], t["batch_size"], t["warmup_steps"] // 10
        self.demo = object()
        self.demo_n = int(round(self.batch_size * t["demo_ratio"]))
        self.episodes = last.get("episode", 0)
        self.env_steps_before = last.get("env_steps", 0)
        self.online = SimpleNamespace(transitions=0)
        self.best_ms = None
        self.metrics: dict = {}
        self.tracker = SimpleNamespace(splits=SplitTracker())
        self.stop = threading.Event()
        self.checkpoints_saved = 0
        self.skill = 0.0

        self.hooks = DashboardHook(self.run_dir, cfg["dashboard"]["link_port"])
        self.stopper = StopController(self.run_dir, self.stop, log=lambda *_: None)
        self._thread = threading.Thread(target=self.train, name="fake-trainer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.stop.set()
        self._thread.join(timeout=5)

    def save_checkpoint(self) -> None:
        self.checkpoints_saved += 1

    # The loop has the shape of Trainer.train / run_episode.
    def train(self) -> None:
        h = self.hooks
        (self.run_dir / "control" / "stop.json").unlink(missing_ok=True)
        self.stopper.set_status("running", phase="training")
        h.on_start(self)
        try:
            while not self.stopper.poll():
                if h.should_pause():
                    time.sleep(0.1)
                    continue
                self.run_episode()
        finally:
            for phase in ("stopping the driver", "saving checkpoint", "writing report"):
                self.stopper.set_status("stopping", phase=phase)
                time.sleep(0.5)
            self.stopper.set_status(STOPPED, phase="done")
            self.stopper.finish()
            h.on_stop(self)

    def run_episode(self) -> None:
        h, splits = self.hooks, self.tracker.splits
        splits.reset()
        dt_ms = 1000.0 / self.hz * SPEEDUP
        race_ms, pos, speed, reward, end = 0.0, 0.0, 0.0, 0.0, None
        crash_p = 0.002 * (1.0 - self.skill) + 0.0002
        while end is None and not self.stopper.poll():
            target = (120 + 180 * self.skill) * (0.75 + 0.25 * math.cos(pos / TRACK_M * 6 * math.pi))
            speed = max(0.0, speed + (target - speed) * 0.15 + self.rng.normal(0, 3))
            step_m = speed / 3.6 * dt_ms / 1000
            pos += step_m
            race_ms += dt_ms
            r = 0.1 * step_m - 0.01
            splits.update(sum(pos >= cp for cp in CHECKPOINTS_M) + (pos >= TRACK_M), int(race_ms))
            if pos >= TRACK_M:
                end, r = "finished", r + 10.0
            elif self.rng.random() < crash_p:
                end, r = ("stuck" if self.rng.random() < 0.5 else "wrong_way"), r - 1.0
            elif race_ms >= self.cfg["env"]["episode_timeout_s"] * 1000:
                end = "timeout"
            reward += r
            self.online.transitions += 1
            self.learn()
            tel = SimpleNamespace(speed_kmh=speed, steer=float(np.tanh(self.rng.normal(0, 0.4))),
                                  gas=1.0 if speed < target else 0.3, brake=0.0,
                                  gear=min(5, 1 + int(speed // 60)), rpm=6000 + 4000 * ((speed % 60) / 60),
                                  race_time_ms=int(race_ms),
                                  last_cp_time_ms=int(race_ms) if end == "finished" else 0)
            h.on_step(tel, SimpleNamespace(progress=min(pos / TRACK_M, 1.0), progress_m=pos, reward=r))
            if end is None and h.should_reset_episode():
                end = "reset"
            time.sleep(1.0 / self.hz)

        self.episodes += 1
        if end == "finished":
            splits.finish()
            if self.best_ms is None or race_ms < self.best_ms:
                self.best_ms = int(race_ms)
        h.on_episode({"episode": self.episodes, "wall_time": round(time.time(), 1),
                      "steps": int(race_ms / dt_ms), "end": end or "stopped",
                      "race_time_s": round(race_ms / 1000, 2), "progress": round(min(pos / TRACK_M, 1.0), 4),
                      "reward": round(reward, 2), "env_steps": self.env_steps_before + self.online.transitions,
                      "updates": self.agent.updates, **{k: round(v, 4) for k, v in self.metrics.items()}})

    def learn(self) -> None:
        if self.online.transitions < self.warmup or self.utd == 0:
            return
        self.agent.updates += self.utd
        lr = self.agent.actor_opt.param_groups[0]["lr"]
        self.skill = min(1.0, self.skill + 2e-5 * self.utd * math.log10(lr / 1e-6) / 4)
        decay = math.exp(-self.agent.updates / 40_000)
        noise = self.rng.normal(0, 0.05)
        self.metrics = {
            "critic_loss": 0.2 + 1.5 * decay + abs(noise),
            "actor_loss": -5 - 20 * (1 - decay) + noise * 3,
            "alpha": 0.02 + 0.1 * decay,
            "q": 5 + 25 * (1 - decay) + noise * 2,
            "entropy": self.agent.target_entropy + noise,
        }
