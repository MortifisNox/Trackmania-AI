"""`tmai train`: the driving loop runs in real time on the main thread while a
learner thread keeps updating SAC from the replay buffers. Ctrl+C stops safely
and writes a checkpoint."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np
import torch

from tmai.rl.demos import load_demo
from tmai.rl.env import TERMINAL, TMEnv
from tmai.rl.replay import ReplayBuffer, concat_batches
from tmai.rl.sac import Policy, SACAgent
from tmai.track.progress import ProgressTracker
from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker
from tmai.track.store import TrackStore

EPISODE_FIELDS = ["episode", "wall_time", "steps", "end", "race_time_s", "progress", "reward",
                  "env_steps", "updates", "critic_loss", "actor_loss", "alpha", "q", "entropy",
                  "late_p99_ms"]
SYNC_EVERY = 20   # learner updates between refreshes of the driving policy


def pick_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


class Trainer:
    def __init__(self, cfg: dict, game=None, device: torch.device | None = None, fresh: bool = False,
                 log=print):
        self.cfg, self.log = cfg, log
        run, env, t = cfg["run"], cfg["env"], cfg["train"]
        self.device = device or pick_device()
        if self.device.type == "cuda":
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["max_vram_fraction"], index)
        torch.manual_seed(run["seed"])
        self.rng = np.random.default_rng(run["seed"])

        self.store = TrackStore(run["output_dir"], run["track"])
        if not self.store.reference.exists():
            raise FileNotFoundError(f"no reference line for {self.store.name!r}; run tmai record-demo first")
        ref = ReferenceLine.load(self.store.reference)
        self.tracker = ProgressTracker(ref, cfg, SplitTracker.load(self.store.splits))

        self.run_dir = Path(run["output_dir"]) / run["name"]
        self.ckpt = self.run_dir / "checkpoints" / "latest.pt"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.demo = self._load_demos()
        self.online = ReplayBuffer(cfg["replay"]["capacity"], env["frame_size"], env["frame_stack"])
        self.agent = SACAgent(cfg, self.device)
        self.policy = Policy(cfg, self.device)
        self.episodes, self.best_ms, self.env_steps_before = 0, None, 0
        if self.ckpt.exists() and not fresh:
            self._load_checkpoint()
        self.policy.load(self.agent.policy_state())

        self.game = game
        self.batch_size, self.utd, self.warmup = t["batch_size"], t["updates_per_step"], t["warmup_steps"]
        self.demo_n = int(round(self.batch_size * t["demo_ratio"])) if self.demo is not None else 0
        self.stop = threading.Event()
        self.replay_lock = threading.Lock()
        self.policy_lock = threading.Lock()
        self.agent_lock = threading.Lock()
        self._pending_policy = None
        self.metrics: dict = {}
        self.run_updates = 0
        self.learner_error: BaseException | None = None

    # -- setup -----------------------------------------------------------------
    def _load_demos(self) -> ReplayBuffer | None:
        files = self.store.demo_files()
        sizes = []
        for f in files:
            with np.load(f) as d:
                sizes.append(len(d["actions"]))
        if not files:
            self.log("no demo laps found: training starts from scratch (slower)")
            return None
        env = self.cfg["env"]
        buf = ReplayBuffer(sum(sizes) + len(files) + 1, env["frame_size"], env["frame_stack"])
        demo_tracker = ProgressTracker(self.tracker.ref, self.cfg, SplitTracker(self.tracker.splits.best))
        for f in files:
            n, total = load_demo(f, buf, demo_tracker)
            self.log(f"demo {f.name}: {n} transitions, reward {total:.1f}")
        return buf if buf.transitions else None

    def _load_checkpoint(self) -> None:
        sd = torch.load(self.ckpt, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(sd["agent"])
        self.episodes, self.best_ms = sd["episodes"], sd["best_ms"]
        self.env_steps_before = sd["env_steps"]
        self.log(f"resumed from {self.ckpt} ({self.episodes} episodes, {self.agent.updates} updates)")

    def save_checkpoint(self) -> Path:
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt.with_suffix(".tmp")
        with self.agent_lock:
            agent = self.agent.state_dict()
        torch.save({"agent": agent, "episodes": self.episodes, "best_ms": self.best_ms,
                    "env_steps": self.env_steps_before + self.online.transitions}, tmp)
        tmp.replace(self.ckpt)
        return self.ckpt

    # -- learner -----------------------------------------------------------------
    def updates_allowed(self) -> int:
        steps = self.online.transitions - self.warmup
        return max(0, steps * self.utd)

    def sample(self) -> dict:
        n_online = self.batch_size - self.demo_n
        with self.replay_lock:
            batch = self.online.sample(n_online, self.rng)
        if self.demo_n:
            batch = concat_batches(batch, self.demo.sample(self.demo_n, self.rng))
        return batch

    def learn_once(self) -> bool:
        if self.run_updates >= self.updates_allowed() or self.online.transitions < self.batch_size:
            return False
        batch = self.sample()
        with self.agent_lock:
            self.metrics = self.agent.update(batch)
        self.run_updates += 1
        if self.run_updates % SYNC_EVERY == 0:
            with self.policy_lock:
                self._pending_policy = self.agent.policy_state()
        return True

    def _learner(self) -> None:
        try:
            while not self.stop.is_set():
                if not self.learn_once():
                    time.sleep(0.002)
        except BaseException as exc:   # surfaced by the driving loop
            self.learner_error = exc
            self.stop.set()

    # -- driving -------------------------------------------------------------------
    def _refresh_policy(self) -> None:
        with self.policy_lock:
            pending, self._pending_policy = self._pending_policy, None
        if pending is not None:
            self.policy.load(pending)

    def run_episode(self, env: TMEnv, hooks=None) -> dict:
        obs = env.reset()
        with self.replay_lock:
            self.online.start_episode()
        total, steps, reason, info = 0.0, 0, None, None
        env.lateness.clear()
        while reason is None and not self.stop.is_set():
            self._refresh_policy()
            action = self.policy.act(obs.frames, obs.state)
            nxt, reward, reason, info = env.step(action)
            if hooks is not None:
                hooks.on_step(env.last_tel, info)
                if reason is None and hooks.should_reset_episode():
                    reason = "reset"   # dashboard: reset track, or pause mid-lap
                    env.game.pad.release()
            with self.replay_lock:
                self.online.add(obs.frame, obs.state, action, reward, terminal=reason in TERMINAL)
            total += reward
            steps += 1
            obs = nxt
        with self.replay_lock:
            self.online.add(obs.frame, obs.state)
        tel = env.last_tel
        lap_ms = None
        if reason == "finished":
            lap_ms = tel.last_cp_time_ms or tel.race_time_ms
        late = sorted(env.lateness) or [0.0]
        return {"end": reason or "stopped", "steps": steps, "reward": total, "lap_ms": lap_ms,
                "race_time_s": tel.race_time_ms / 1000, "progress": info.progress if info else 0.0,
                "late_p99_ms": late[int(0.99 * (len(late) - 1))] * 1000}

    def _log_episode(self, ep: dict, writer, f) -> dict:
        self.episodes += 1
        m = self.metrics
        row = {"episode": self.episodes, "wall_time": round(time.time(), 1), "steps": ep["steps"],
               "end": ep["end"], "race_time_s": round(ep["race_time_s"], 2), "progress": round(ep["progress"], 4),
               "reward": round(ep["reward"], 2), "env_steps": self.env_steps_before + self.online.transitions,
               "updates": self.agent.updates, "late_p99_ms": round(ep["late_p99_ms"], 1),
               **{k: round(m.get(k, float("nan")), 4) for k in ("critic_loss", "actor_loss", "alpha", "q", "entropy")}}
        writer.writerow(row)
        f.flush()
        note = ""
        if ep["lap_ms"] is not None:
            if self.best_ms is None or ep["lap_ms"] < self.best_ms:
                self.best_ms = ep["lap_ms"]
                note = "  NEW BEST"
            note = f"  lap {ep['lap_ms'] / 1000:.3f}s{note}"
        learning = "warming up" if self.online.transitions < self.warmup else f"{self.agent.updates} updates"
        self.log(f"ep {self.episodes:5d}  {ep['end']:9s}  {ep['race_time_s']:6.2f}s  "
                 f"progress {ep['progress'] * 100:5.1f}%  reward {ep['reward']:7.1f}  {learning}  "
                 f"alpha {m.get('alpha', float('nan')):.3f}  late p99 {ep['late_p99_ms']:.0f} ms{note}")
        return row

    def _wait_while_paused(self, game, hooks) -> None:
        if not hooks.should_pause():
            return
        game.pad.release()
        self.log("paused")
        while hooks.should_pause() and not self.stop.is_set():
            time.sleep(0.2)
        self.log("resumed")

    def train(self, minutes: float | None = None, hooks=None) -> int:
        """Drive and learn until stopped. `hooks` (tmai.run.hooks.TrainerHooks)
        get on_start / on_step / on_episode / should_pause / on_stop; without
        hooks the trainer only saves its checkpoint on the way out."""
        from tmai.game.interface import GameInterface
        from tmai.run.hooks import TrainerHooks

        own_save = hooks is None
        hooks = hooks or TrainerHooks()

        game = self.game or GameInterface(self.cfg, capture=True)
        env = TMEnv(self.cfg, game, self.tracker)
        learner = threading.Thread(target=self._learner, name="learner", daemon=True)
        csv_path = self.run_dir / "episodes.csv"
        new_file = not csv_path.exists()
        deadline = time.monotonic() + minutes * 60 if minutes else None
        last_ckpt = time.monotonic()
        every = self.cfg["train"]["checkpoint_every_min"] * 60
        self.log(f"training on {self.store.name!r} with {self.device}; Ctrl+C stops and saves")
        hooks.on_start(self)
        learner.start()
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
                if new_file:
                    writer.writeheader()
                while not self.stop.is_set():
                    self._wait_while_paused(game, hooks)
                    if self.stop.is_set():
                        break
                    try:
                        ep = self.run_episode(env, hooks)
                    except TimeoutError as exc:
                        self.log(f"game not responding ({exc}); retrying")
                        time.sleep(2.0)
                        continue
                    hooks.on_episode(self._log_episode(ep, writer, f))
                    if time.monotonic() - last_ckpt >= every:
                        self.save_checkpoint()
                        last_ckpt = time.monotonic()
                    if deadline and time.monotonic() >= deadline:
                        break
        except KeyboardInterrupt:
            self.log("stopping...")
        finally:
            self.stop.set()
            learner.join(timeout=30)
            game.pad.release()
            if self.game is None:
                game.close()
            if own_save:
                self.log(f"saved {self.save_checkpoint()}")
            else:
                hooks.on_stop(self)
        if self.learner_error is not None:
            raise self.learner_error
        return 0
