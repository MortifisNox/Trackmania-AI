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

from tmai.game.interface import CheckpointRespawn
from tmai.rl.demos import add_lap, load_demo
from tmai.rl.env import TERMINAL, TMEnv
from tmai.rl.replay import ReplayBuffer, concat_batches
from tmai.rl.sac import Policy, SACAgent
from tmai.track.baseline import PaceBaseline
from tmai.track.progress import ProgressTracker
from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker
from tmai.track.store import TrackStore

EPISODE_FIELDS = ["episode", "wall_time", "steps", "end", "race_time_s", "progress", "reward",
                  "env_steps", "updates", "critic_loss", "actor_loss", "alpha", "q", "entropy",
                  "late_p99_ms"]
EVAL_FIELDS = ["episode", "wall_time", "env_steps", "end", "lap_ms", "race_time_s", "progress", "best_eval_ms"]
STALL_WARN_S = 0.1   # longest race-clock gap beyond one step that the log points out
SYNC_EVERY = 20   # learner updates between refreshes of the driving policy
NO_INPUT = np.array([0.0, -1.0, -1.0], dtype=np.float32)   # the previous action at the start


def anneal(start: float, end: float, steps: int, done: int) -> float:
    """Linear schedule from start to end over `steps`, then held at end."""
    if steps <= 0:
        return end
    return start + (end - start) * min(1.0, done / steps)


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
        if ref.times is None and self.store.demo_files():
            # Lines built before pace rewards existed: rebuild with lap timing.
            from tmai.track.commands import build_reference

            build_reference(cfg)
            ref = ReferenceLine.load(self.store.reference)
        rw = cfg["reward"]
        self.baseline = PaceBaseline(ref, rw["pace_baseline_laps"], rw["pace_baseline_min_laps"])
        self.tracker = ProgressTracker(ref, cfg, SplitTracker.load(self.store.splits), self.baseline)

        self.run_dir = Path(run["output_dir"]) / run["name"]
        self.ckpt = self.run_dir / "checkpoints" / "latest.pt"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.best_laps: list[dict] = []   # agent's fastest finished laps, fastest first
        self.demo = self._load_demos()
        self.online = ReplayBuffer(cfg["replay"]["capacity"], env["frame_size"], env["frame_stack"])
        self.agent = SACAgent(cfg, self.device)
        self.policy = Policy(cfg, self.device)
        self.episodes, self.best_ms, self.env_steps_before = 0, None, 0
        self.best_eval_ms = None
        if self.ckpt.exists() and not fresh:
            self._load_checkpoint()
        self.policy_state_used = self.agent.policy_state()
        self.policy.load(self.policy_state_used)

        self.game = game
        self.batch_size, self.utd, self.warmup = t["batch_size"], t["updates_per_step"], t["warmup_steps"]
        self.n_step, self.gamma = t["n_step"], t["gamma"]
        # Schedules run until the dashboard sets the value by hand (see _apply_schedules).
        self._demo_sched: int | None = None
        self._entropy_sched: float | None = None
        self.demo_n = 0
        self._apply_schedules()
        self.stop = threading.Event()
        self.replay_lock = threading.Lock()
        self.policy_lock = threading.Lock()
        self.agent_lock = threading.Lock()
        self._pending_policy = None
        self.metrics: dict = {}
        self.run_updates = 0
        self.learner_error: BaseException | None = None

    # -- setup -----------------------------------------------------------------
    def _load_demos(self, quiet: bool = False) -> ReplayBuffer | None:
        """Guide laps: your demo laps plus the agent's fastest finished laps, with
        rewards recomputed against the current reference line."""
        files = self.store.demo_files()
        sizes = []
        for f in files:
            with np.load(f) as d:
                sizes.append(len(d["actions"]))
        sizes += [len(lap["tels"]) for lap in self.best_laps]
        if not sizes:
            self.log("no demo laps found: training starts from scratch (slower)")
            return None
        env = self.cfg["env"]
        buf = ReplayBuffer(sum(sizes) + len(sizes) + 1, env["frame_size"], env["frame_stack"])
        demo_tracker = ProgressTracker(self.tracker.ref, self.cfg, SplitTracker(self.tracker.splits.best),
                                       self.baseline)
        for f in files:
            n, total = load_demo(f, buf, demo_tracker)
            if not quiet:
                self.log(f"demo {f.name}: {n} transitions, reward {total:.1f}")
        for lap in self.best_laps:
            add_lap(buf, demo_tracker, lap["frames"], lap["tels"], lap["actions"], NO_INPUT)
        return buf if buf.transitions else None

    def keep_best_lap(self, ep: dict) -> bool:
        """Keep a finished lap among the train.best_laps_kept fastest as a guide
        lap, so batches learn from the agent's best driving, not only the demos."""
        keep = self.cfg["train"]["best_laps_kept"]
        if not keep or ep["lap_ms"] is None or "lap" not in ep:
            return False
        if len(self.best_laps) >= keep and ep["lap_ms"] >= self.best_laps[-1]["lap_ms"]:
            return False
        self.best_laps = sorted(self.best_laps + [{"lap_ms": ep["lap_ms"], **ep["lap"]}],
                                key=lambda lap: lap["lap_ms"])[:keep]
        self.demo = self._load_demos(quiet=True)
        return True

    def env_steps(self) -> int:
        return self.env_steps_before + self.online.transitions

    def _apply_schedules(self) -> None:
        """Move the guide-lap share and the entropy target along their schedules.
        A value that no longer matches what the schedule last set was changed
        from the dashboard; that schedule then stops for the rest of the run."""
        t, done = self.cfg["train"], self.env_steps()
        if self._demo_sched is None or self.demo_n == self._demo_sched:
            ratio = anneal(t["demo_ratio"], t["demo_ratio_end"], t["demo_anneal_steps"], done)
            self.demo_n = self._demo_sched = int(round(self.batch_size * ratio)) if self.demo is not None else 0
        else:
            self._demo_sched = -1   # never matches again
        if self._entropy_sched is None or self.agent.target_entropy == self._entropy_sched:
            self._entropy_sched = anneal(t["target_entropy"], t["target_entropy_end"],
                                         t["entropy_anneal_steps"], done)
            self.agent.target_entropy = self._entropy_sched
        else:
            self._entropy_sched = float("nan")

    def _load_checkpoint(self) -> None:
        sd = torch.load(self.ckpt, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(sd["agent"])
        self.episodes, self.best_ms = sd["episodes"], sd["best_ms"]
        self.env_steps_before = sd["env_steps"]
        self.best_eval_ms = sd.get("best_eval_ms")
        self.log(f"resumed from {self.ckpt} ({self.episodes} episodes, {self.agent.updates} updates)")

    def save_checkpoint(self) -> Path:
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt.with_suffix(".tmp")
        with self.agent_lock:
            agent = self.agent.state_dict()
        torch.save({"agent": agent, "episodes": self.episodes, "best_ms": self.best_ms,
                    "best_eval_ms": self.best_eval_ms, "env_steps": self.env_steps()}, tmp)
        tmp.replace(self.ckpt)
        return self.ckpt

    def save_best(self) -> Path:
        """Driving weights of the best test lap, for evaluation and the run report."""
        path = self.ckpt.with_name("best_test_lap.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"policy": self.policy_state_used, "lap_ms": self.best_eval_ms,
                    "episodes": self.episodes, "env_steps": self.env_steps()}, tmp)
        tmp.replace(path)
        return path

    # -- learner -----------------------------------------------------------------
    def updates_allowed(self) -> int:
        steps = self.online.transitions - self.warmup
        return max(0, steps * self.utd)

    def pace_reward(self, u0, u1, dt):
        """The pace term of stored transitions, against today's baseline."""
        return self.cfg["reward"]["pace_per_s"] * self.baseline.pace(u0, u1, dt)

    def sample(self) -> dict:
        demo, demo_n = self.demo, self.demo_n
        if demo is None:
            demo_n = 0
        n_online = self.batch_size - demo_n
        with self.replay_lock:
            batch = self.online.sample(n_online, self.rng, self.n_step, self.gamma, self.pace_reward)
        if demo_n:
            batch = concat_batches(batch, demo.sample(demo_n, self.rng, self.n_step, self.gamma, self.pace_reward))
        return batch

    def learn_once(self) -> bool:
        if self.run_updates >= self.updates_allowed() or self.online.transitions < self.batch_size:
            return False
        self._apply_schedules()
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
            self.policy_state_used = pending

    def run_episode(self, env: TMEnv, hooks=None, deterministic: bool = False) -> dict:
        """One attempt. deterministic=True drives a test lap without exploration
        noise (its transitions still go to the replay buffer)."""
        self.tracker.best_lap_ms = self.best_ms   # for the personal-best bonus at the finish
        obs = env.reset()
        frames, tels, actions = [obs.frame], [env.last_tel], []
        trace = [(env.last_tel.pos, env.last_tel.race_time_ms, env.last_tel.cp_count, env.last_tel.last_cp_time_ms)]
        with self.replay_lock:
            self.online.start_episode()
        total, steps, reason, info = 0.0, 0, None, None
        env.lateness.clear()
        while reason is None and not self.stop.is_set():
            if not deterministic:   # a test lap keeps one set of weights from start to finish
                self._refresh_policy()
            action = self.policy.act(obs.frames, obs.state, deterministic)
            nxt, reward, reason, info = env.step(action)
            t = env.last_tel
            frames.append(nxt.frame)
            tels.append(t)
            actions.append(np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0))
            trace.append((t.pos, t.race_time_ms, t.cp_count, t.last_cp_time_ms))
            if hooks is not None:
                hooks.on_step(env.last_tel, info)
                if reason is None and hooks.should_reset_episode():
                    reason = "reset"   # dashboard: reset track, or pause mid-lap
                    env.game.pad.release()
            with self.replay_lock:
                # The pace term is left out and scored again at sampling time.
                self.online.add(obs.frame, obs.state, action, reward - info.terms.get("pace", 0.0),
                                terminal=reason in TERMINAL, pace=info.pace_inputs)
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
        # Race time the clock ran without a driving step (game or loop hitch, slow restart).
        clock = [0] + [max(t, 0) for _, t, *_ in trace]
        stall_s = max(0.0, max(np.diff(clock), default=0) / 1000.0 - 1.0 / self.cfg["env"]["step_hz"])
        ep = {"end": reason or "stopped", "steps": steps, "reward": total, "lap_ms": lap_ms,
              "race_time_s": tel.race_time_ms / 1000, "progress": info.progress if info else 0.0,
              "late_p99_ms": late[int(0.99 * (len(late) - 1))] * 1000, "trace": trace,
              "deterministic": deterministic, "stall_s": stall_s}
        if lap_ms is not None:
            ep["lap"] = {"frames": frames, "tels": tels, "actions": actions}
        return ep

    def add_baseline_lap(self, ep: dict) -> None:
        """Finished laps join the pace baseline (the median of recent laps)."""
        if ep["lap_ms"] is None:
            return
        pos = np.array([p for p, *_ in ep["trace"]], dtype=np.float64)
        times = np.array([t for _, t, *_ in ep["trace"]], dtype=np.float64) / 1000.0
        self.baseline.add_lap(pos, times)

    def is_eval_episode(self) -> bool:
        every = self.cfg["train"]["eval_every"]
        return bool(every) and self.online.transitions >= self.warmup and (self.episodes + 1) % every == 0

    def record_eval(self, ep: dict, row: dict) -> bool:
        """Log a test lap to eval.csv; a new best test lap saves checkpoints/best_test_lap.pt."""
        better = ep["lap_ms"] is not None and (self.best_eval_ms is None or ep["lap_ms"] < self.best_eval_ms)
        if better:
            self.best_eval_ms = ep["lap_ms"]
            self.save_best()
            self.log(f"test lap {ep['lap_ms'] / 1000:.3f}s is the best so far: saved {self.ckpt.with_name('best_test_lap.pt')}")
        path = self.run_dir / "eval.csv"
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow({"episode": row["episode"], "wall_time": row["wall_time"], "env_steps": row["env_steps"],
                             "end": ep["end"], "lap_ms": ep["lap_ms"] if ep["lap_ms"] is not None else "",
                             "race_time_s": row["race_time_s"], "progress": row["progress"],
                             "best_eval_ms": self.best_eval_ms if self.best_eval_ms is not None else ""})
        return better

    def maybe_adopt_lap(self, ep: dict) -> bool:
        """Make a finished lap the new reference line when it beats the current
        one by track.adopt_best_gain_ms, so progress, lookahead and pace follow
        the agent's faster line instead of staying anchored to the demo."""
        gain_ms = self.cfg["track"]["adopt_best_gain_ms"]
        ref = self.tracker.ref
        if not gain_ms or ep["lap_ms"] is None or ref.times is None:
            return False
        ref_ms = ref.times[-1] * 1000.0
        if ref_ms - ep["lap_ms"] < gain_ms:
            return False
        pos = np.array([p for p, *_ in ep["trace"]], dtype=np.float64)
        times = np.array([t for _, t, *_ in ep["trace"]], dtype=np.float64) / 1000.0
        cps, last = [], 0
        for _, _, count, cp_ms in ep["trace"]:
            if count > last:
                cps.append(int(cp_ms))
                last = count
        start = int(np.argmax(times >= 0))   # drop the countdown before the clock starts
        new = ReferenceLine.from_positions(pos[start:], self.cfg["track"]["reference_spacing_m"], times[start:])
        backup = self.store.root / "reference_demo.npz"
        if not backup.exists():
            ref.save(backup)
        new.save(self.store.reference)
        SplitTracker(cps).save(self.store.splits)
        self.tracker.ref = new
        self.baseline.set_ref(new)
        self.tracker.splits.best = cps
        self.log(f"new reference line: {ep['lap_ms'] / 1000:.3f}s lap replaces {ref_ms / 1000:.3f}s")
        self.demo = self._load_demos()   # demo rewards follow the new line
        return True

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
        if ep.get("deterministic"):
            note += "  (test lap)"
        if ep.get("stall_s", 0.0) >= STALL_WARN_S:
            note += f"  STALL {ep['stall_s']:.2f}s (race clock ran without a driving step)"
        if self.baseline.from_recent_laps:
            note += f"  baseline {self.baseline.lap_time_s:.2f}s"
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
                        ep = self.run_episode(env, hooks, deterministic=self.is_eval_episode())
                    except CheckpointRespawn as exc:
                        self.log(f"stopping: {exc}")
                        break
                    except TimeoutError as exc:
                        self.log(f"game not responding ({exc}); retrying")
                        time.sleep(2.0)
                        continue
                    row = self._log_episode(ep, writer, f)
                    if ep["deterministic"]:
                        self.record_eval(ep, row)
                    self.add_baseline_lap(ep)
                    self.keep_best_lap(ep)
                    self.maybe_adopt_lap(ep)
                    hooks.on_episode(row)
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
