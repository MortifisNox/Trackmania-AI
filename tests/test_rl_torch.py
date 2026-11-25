import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tmai.budget import STATE_DIM  # noqa: E402
from tmai.game import protocol  # noqa: E402
from test_rl import LENGTH, PLAYING, make_track, small_cfg, tel  # noqa: E402


def test_sac_update_runs(tmp_path):
    from tmai.rl.sac import Policy, SACAgent

    cfg = small_cfg(tmp_path)
    agent = SACAgent(cfg, "cpu")
    rng = np.random.default_rng(0)
    batch = {
        "frames": rng.integers(0, 255, (16, 4, 64, 64), dtype=np.uint8),
        "next_frames": rng.integers(0, 255, (16, 4, 64, 64), dtype=np.uint8),
        "state": rng.standard_normal((16, STATE_DIM)).astype(np.float32),
        "next_state": rng.standard_normal((16, STATE_DIM)).astype(np.float32),
        "action": rng.uniform(-1, 1, (16, 3)).astype(np.float32),
        "reward": rng.standard_normal(16).astype(np.float32),
        "terminal": np.zeros(16, np.float32),
    }
    for _ in range(3):
        m = agent.update(batch)
    assert all(np.isfinite(v) for v in m.values()) and agent.updates == 3
    policy = Policy(cfg, "cpu")
    policy.load(agent.policy_state())
    a = policy.act(batch["frames"][0], batch["state"][0])
    assert a.shape == (3,) and np.all(np.abs(a) <= 1)
    clone = SACAgent(cfg, "cpu")
    clone.load_state_dict(agent.state_dict())
    assert clone.updates == 3


@pytest.mark.parametrize("camera", [True, False])
def test_sac_autocasts_only_image_encoders(tmp_path, monkeypatch, camera):
    from tmai.rl.features import state_dim
    from tmai.rl.sac import SACAgent

    cfg = small_cfg(tmp_path)
    cfg["env"]["observation"] = "image" if camera else "state"
    cfg["model"]["critic_head"] = "scalar"
    cfg["train"]["critic_loss"] = "mse"
    agent = SACAgent(cfg, "cpu")
    # The CPU supports bf16 autocast too, so this exercises the precision
    # boundary without requiring a CUDA device.
    agent.use_bf16 = camera
    observed = []

    def check(name, autocast):
        def hook(module, args, output):
            assert torch.is_autocast_enabled("cpu") == autocast
            if not autocast:
                outputs = output if isinstance(output, tuple) else (output,)
                assert all(value.dtype == torch.float32 for value in outputs)
            observed.append(name)
        return hook

    for name in ("encoder", "encoder_target", "actor", "critic", "critic_target"):
        getattr(agent, name).register_forward_hook(check(name, camera and name.startswith("encoder")))
    mse = torch.nn.functional.mse_loss

    def fp32_loss(q, target):
        assert not torch.is_autocast_enabled("cpu")
        assert q.dtype == target.dtype == torch.float32
        observed.append("loss")
        return mse(q, target)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", fp32_loss)
    size = 64 if camera else 0
    frames = np.zeros((2, 4, size, size), np.uint8)
    state = np.zeros((2, state_dim(cfg["env"])))
    agent.update({"frames": frames, "next_frames": frames, "state": state, "next_state": state,
                  "action": np.zeros((2, 3)), "reward": np.zeros(2), "discount": np.full(2, 0.99)})
    assert set(observed) == {"encoder", "encoder_target", "actor", "critic", "critic_target", "loss"}


def test_checkpoint_can_serialize_after_learner_mutates_live_weights(tmp_path, monkeypatch):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, warmup_steps=0, updates_per_step=1)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    trainer.run_episode(TMEnv(cfg, trainer.game, trainer.tracker))
    assert trainer.learn_once()
    name, parameter = next(iter(trainer.agent.critic.named_parameters()))
    moment = trainer.agent.critic_opt.state[parameter]["exp_avg"]
    expected_weights, expected_moment = parameter.detach().clone(), moment.clone()
    expected_alpha = trainer.agent.log_alpha.item()
    save = torch.save

    def save_after_update(checkpoint, path):
        assert not trainer.agent_lock.locked()
        with trainer.agent_lock, torch.no_grad():
            parameter.add_(3)
            moment.add_(3)
            trainer.agent.log_alpha.add_(3)
        save(checkpoint, path)

    monkeypatch.setattr(torch, "save", save_after_update)
    state = torch.load(trainer.save_checkpoint(), weights_only=False)["agent"]
    assert torch.equal(state["critic"][name], expected_weights)
    group = trainer.agent.critic_opt.param_groups[0]["params"]
    index = next(i for i, p in enumerate(group) if p is parameter)
    assert torch.equal(state["critic_opt"]["state"][index]["exp_avg"], expected_moment)
    assert state["log_alpha"].item() == expected_alpha


class FakePad:
    def __init__(self):
        self.last = (0.0, 0.0, 0.0)
        self.releases = []

    def release(self):
        self.releases.append(self.last)
        self.last = (0.0, 0.0, 0.0)


class FakeGame:
    """A car on a straight line that moves at a speed set by the gas input."""

    def __init__(self):
        self.pad = FakePad()
        self.x, self.t = 0.0, 0

    def restart(self):
        self.x, self.t = 0.0, 0
        return self._tel()

    def act(self, controls):
        self.pad.last = controls
        return controls

    def _tel(self):
        flags = PLAYING | (protocol.FLAG_FINISHED if self.x >= LENGTH else 0)
        speed = 5.0 + 20.0 * self.pad.last[1]
        return tel(pos=(self.x, 0.0, 0.0), speed=speed, vel=(speed, 0.0, 0.0), race_time_ms=self.t, flags=flags)

    def observe(self):
        self.x += 0.05 * (5.0 + 20.0 * self.pad.last[1])
        self.t += 50
        return np.full((64, 64), int(self.x) % 255, np.uint8), self._tel()


def test_trainer_runs_episodes_and_resumes(tmp_path):
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, warmup_steps=20, updates_per_step=1)
    make_track(tmp_path, cfg)
    lines = []
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lines.append)
    assert trainer.demo is not None and trainer.demo_n == 8
    env_steps = 0
    from tmai.rl.env import TMEnv

    env = TMEnv(cfg, trainer.game, trainer.tracker)
    for _ in range(3):
        ep = trainer.run_episode(env)
        env_steps += ep["steps"]
        assert ep["end"] in ("finished", "timeout")
        while trainer.learn_once():
            pass
    assert trainer.online.transitions == env_steps
    assert trainer.agent.updates == (env_steps - 20) * 1
    trainer.episodes = 3
    trainer.save_checkpoint()
    again = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lines.append)
    assert again.agent.updates == trainer.agent.updates and again.episodes == 3
    assert any("resumed" in line for line in lines)


def test_trainer_train_loop_writes_csv(tmp_path):
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1)
    make_track(tmp_path, cfg, with_demo=False)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert trainer.demo is None and trainer.demo_n == 0
    trainer.train(minutes=0.02)
    rows = (tmp_path / "default" / "episodes.csv").read_text().splitlines()
    assert rows[0].startswith("episode,") and len(rows) >= 2
    assert (tmp_path / "default" / "checkpoints" / "latest.pt").exists()


def test_resumed_episode_log_preserves_legacy_rows_and_reports_q_clipping(tmp_path, monkeypatch):
    import csv

    from tmai.rl.train import EPISODE_FIELDS, Trainer
    from tmai.run.hooks import TrainerHooks
    from tmai.run.report import build_report, read_episodes, summarize

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg, with_demo=False)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    path = trainer.run_dir / "episodes.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS[:-1])
        writer.writeheader()
        writer.writerow({"episode": 1, "end": "stuck", "env_steps": 10, "updates": 1, "q": 10})
        f.write("2,partial")   # a crash cut the row short
    assert "c-q_clip_frac" not in build_report(trainer.run_dir).read_text()

    trainer.episodes, trainer.agent.updates = 1, 2
    trainer.metrics = {"q_clip_frac": 0.125, "q": 11.0}
    monkeypatch.setattr(trainer, "_learner", lambda: None)
    hooks = TrainerHooks()
    monkeypatch.setattr(hooks, "on_episode", lambda row: trainer.stop.set())
    trainer.train(hooks=hooks)

    with path.open(newline="") as f:
        old, new = list(csv.DictReader(f))
    assert old["episode"] == "1" and old["q_clip_frac"] == ""
    assert new["episode"] == "2" and new["q_clip_frac"] == "0.125"
    assert len(read_episodes(path)) == 2
    assert summarize(trainer.run_dir)["stability"]["q_clip_frac"]["n"] == 1
    assert 'id="c-q_clip_frac"' in build_report(trainer.run_dir).read_text()


@pytest.mark.parametrize("exit_path", ["startup", "interrupt", "respawn", "learner", "deadline", "join_interrupt"])
def test_train_releases_pad_before_stopping_and_joining_learner(tmp_path, monkeypatch, exit_path):
    import threading

    from tmai.game.interface import CheckpointRespawn, FULL_GAS_CONTROLS
    from tmai.rl import train as train_module
    from tmai.rl.train import Trainer
    from tmai.run.hooks import TrainerHooks

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg, with_demo=False)
    game = FakeGame()
    trainer = Trainer(cfg, game=game, device=torch.device("cpu"), log=lambda *_: None)
    hooks = TrainerHooks()

    def start(trainer):
        game.act(FULL_GAS_CONTROLS)
        if exit_path == "startup":
            raise KeyboardInterrupt

    def attempt(*args, **kwargs):
        if exit_path == "respawn":
            raise CheckpointRespawn("fake checkpoint respawn")
        raise KeyboardInterrupt

    def fail_learning():
        raise RuntimeError("fake learner failure")

    monkeypatch.setattr(hooks, "on_start", start)
    monkeypatch.setattr(trainer, "run_episode", attempt)
    if exit_path == "learner":
        monkeypatch.setattr(trainer, "learn_once", fail_learning)
        prepare_log = train_module._prepare_episode_log

        def wait_for_learner_error(path):
            assert trainer.stop.wait(5)
            return prepare_log(path)

        monkeypatch.setattr(train_module, "_prepare_episode_log", wait_for_learner_error)
    else:
        monkeypatch.setattr(trainer, "_learner", lambda: None)
        stop = trainer.stop.set

        def check_stop():
            assert game.pad.releases == [FULL_GAS_CONTROLS]
            stop()

        monkeypatch.setattr(trainer.stop, "set", check_stop)

    join = threading.Thread.join
    joined = []

    def check_join(thread, timeout=None):
        assert game.pad.releases == [FULL_GAS_CONTROLS]
        assert trainer.stop.is_set()
        joined.append(thread.name)
        join(thread, timeout)
        if exit_path == "join_interrupt":
            raise KeyboardInterrupt

    monkeypatch.setattr(threading.Thread, "join", check_join)
    if exit_path == "learner":
        with pytest.raises(RuntimeError, match="fake learner failure"):
            trainer.train(hooks=hooks)
    elif exit_path == "join_interrupt":
        with pytest.raises(KeyboardInterrupt):
            trainer.train(hooks=hooks)
    else:
        assert trainer.train(minutes=0 if exit_path == "deadline" else None, hooks=hooks) == 0
    assert joined == ([] if exit_path == "startup" else ["learner"])
    assert game.pad.releases == [FULL_GAS_CONTROLS]
    assert game.pad.last == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("exit_path", ["reset_interrupt", "step_interrupt", "respawn", "deadline", "stop"])
def test_run_episode_releases_held_input_on_exit(tmp_path, monkeypatch, exit_path):
    from types import SimpleNamespace

    from tmai.game.interface import CheckpointRespawn, FULL_GAS_CONTROLS
    from tmai.rl import train as train_module
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    class HoldingGame(FakeGame):
        def restart(self):
            self.act(FULL_GAS_CONTROLS)
            if exit_path == "reset_interrupt":
                raise KeyboardInterrupt
            if exit_path == "respawn":
                raise CheckpointRespawn("fake checkpoint respawn")
            return super().restart()

        def observe(self):
            if self.t:
                if exit_path == "step_interrupt":
                    raise KeyboardInterrupt
                if exit_path == "stop":
                    trainer.stop.set()
            return super().observe()

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg, with_demo=False)
    game = HoldingGame()
    trainer = Trainer(cfg, game=game, device=torch.device("cpu"), log=lambda *_: None)
    env = TMEnv(cfg, game, trainer.tracker)
    monkeypatch.setattr(trainer.policy, "act", lambda *args: np.array([0.0, 1.0, -1.0]))
    if exit_path == "deadline":
        monkeypatch.setattr(train_module, "time", SimpleNamespace(monotonic=lambda: game.t / 1000))
        ep = trainer.run_episode(env, deadline=0.1)
        assert ep["end"] == "stopped" and ep["steps"] == 1
    elif exit_path == "stop":
        assert trainer.run_episode(env)["end"] == "stopped"
    else:
        with pytest.raises(CheckpointRespawn if exit_path == "respawn" else KeyboardInterrupt):
            trainer.run_episode(env)
    assert game.pad.releases[-1] == FULL_GAS_CONTROLS
    assert game.pad.last == (0.0, 0.0, 0.0)


def test_train_loop_survives_restart_errors(tmp_path, monkeypatch):
    """A restart that times out is retried; one that respawns at a checkpoint
    stops training cleanly (both go through the loop's except clauses)."""
    from tmai.game.interface import CheckpointRespawn
    from tmai.rl import train as train_mod
    from tmai.rl.train import Trainer

    class FlakyGame(FakeGame):
        calls = 0

        def restart(self):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("menu up")
            if self.calls == 4:
                raise CheckpointRespawn("respawned at a checkpoint")
            return super().restart()

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1)
    make_track(tmp_path, cfg, with_demo=False)
    lines = []
    game = FlakyGame()
    trainer = Trainer(cfg, game=game, device=torch.device("cpu"), log=lines.append)
    monkeypatch.setattr(train_mod.time, "sleep", lambda s: None)
    trainer.train(minutes=1.0)
    assert game.calls == 4
    assert any("retrying" in line for line in lines) and any("stopping: respawned" in line for line in lines)


def test_trainer_runs_shutdown_hooks(tmp_path):
    from tmai.rl.train import Trainer
    from tmai.run.hooks import HookList, TrainerHooks
    from tmai.run.shutdown import AgentLaps, ShutdownHook

    class Recorder(TrainerHooks):
        def __init__(self):
            self.calls = []

        def on_start(self, trainer):
            self.calls.append("start")

        def on_step(self, tel, info):
            if "step" not in self.calls:
                self.calls.append("step")

        def on_episode(self, row):
            self.calls.append(("episode", row["end"]))

        def on_stop(self, trainer):
            self.calls.append("stop")

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1)
    make_track(tmp_path, cfg)
    rec = Recorder()
    shutdown = ShutdownHook(cfg, log=lambda *_: None, agent_laps=AgentLaps(tmp_path / "op" / "a" / "b" / "ai.txt"),
                            replays_root=tmp_path / "replays")
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    trainer.train(minutes=0.02, hooks=HookList(shutdown, rec))
    assert rec.calls[:2] == ["start", "step"] and rec.calls[-1] == "stop"
    assert ("episode", "finished") in rec.calls
    run_dir = tmp_path / "default"
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "best_laps.jsonl").exists()


def test_stop_request_reaches_a_paused_trainer(tmp_path):
    """`tmai stop` and the dashboard's stop button must work while paused."""
    import threading
    import time

    from tmai.rl.train import Trainer
    from tmai.run.hooks import HookList, TrainerHooks
    from tmai.run.shutdown import AgentLaps, ShutdownHook
    from tmai.run.stop import read_status, request_stop

    class PausedFor30s(TrainerHooks):
        started = None

        def should_pause(self):
            self.started = self.started or time.monotonic()
            return time.monotonic() - self.started < 30

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    shutdown = ShutdownHook(cfg, log=lambda *_: None, agent_laps=AgentLaps(tmp_path / "op" / "a" / "b" / "ai.txt"),
                            replays_root=tmp_path / "replays")
    threading.Timer(0.5, request_stop, (tmp_path / "default", "test")).start()
    started = time.monotonic()
    trainer.train(hooks=HookList(PausedFor30s(), shutdown))   # pausing hook first: every hook is still asked
    assert time.monotonic() - started < 10
    assert read_status(tmp_path / "default")["state"] == "stopped"


def test_faster_lap_becomes_reference(tmp_path):
    from tmai.rl.train import Trainer
    from tmai.track.reference import ReferenceLine

    cfg = small_cfg(tmp_path)
    store = make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert trainer.tracker.ref.times is not None              # rebuilt with lap timing
    ref_ms = trainer.tracker.ref.times[-1] * 1000
    trace = [((3.0 * i, 0.0, 0.0), 50 * i, 0, 0) for i in range(21)]
    slow = {"lap_ms": int(ref_ms) + 100, "trace": trace}
    assert not trainer.maybe_adopt_lap(slow)
    fast = {"lap_ms": 1000, "trace": trace}
    assert trainer.maybe_adopt_lap(fast)
    assert (store.root / "reference_demo.npz").exists()
    assert ReferenceLine.load(store.reference).times[-1] == pytest.approx(1.0)
    assert trainer.tracker.ref.times[-1] == pytest.approx(1.0) and trainer.demo is not None


def test_anneal_schedule():
    from tmai.rl.train import anneal

    assert anneal(0.5, 0.1, 100, 0) == 0.5
    assert anneal(0.5, 0.1, 100, 50) == pytest.approx(0.3)
    assert anneal(0.5, 0.1, 100, 1000) == pytest.approx(0.1)
    assert anneal(0.5, 0.1, 0, 0) == 0.1


def test_schedules_follow_steps_until_set_by_hand(tmp_path):
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, demo_anneal_steps=100, entropy_anneal_steps=100, target_entropy_end=-6.0)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert trainer.demo_n == 8 and trainer.agent.target_entropy == -3.0
    trainer.env_steps_before = 50
    trainer._apply_schedules()
    assert trainer.demo_n == 5 and trainer.agent.target_entropy == pytest.approx(-4.5)   # 0.3 * 16 rounds to 5
    trainer.demo_n, trainer.agent.target_entropy = 12, -1.0      # what the dashboard does
    trainer.env_steps_before = 100
    trainer._apply_schedules()
    assert trainer.demo_n == 12 and trainer.agent.target_entropy == -1.0


def test_best_laps_join_the_guide_laps(tmp_path):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, best_laps_kept=2)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    demo_only = trainer.demo.transitions
    env = TMEnv(cfg, trainer.game, trainer.tracker)
    ep = trainer.run_episode(env)
    assert ep["end"] == "finished" and len(ep["lap"]["actions"]) == ep["steps"]
    assert trainer.keep_best_lap(ep)
    assert trainer.demo.transitions == demo_only + ep["steps"]
    np.testing.assert_allclose(trainer.demo.action[demo_only + 1], ep["lap"]["actions"][0])
    slower = {**ep, "lap_ms": ep["lap_ms"] + 10}
    assert trainer.keep_best_lap(slower) and not trainer.keep_best_lap({**ep, "lap_ms": ep["lap_ms"] + 20})
    assert [lap["lap_ms"] for lap in trainer.best_laps] == [ep["lap_ms"], ep["lap_ms"] + 10]
    batch = trainer.sample()
    assert batch["reward"].shape == (16,) and "discount" in batch
    trainer.save_checkpoint()
    resumed = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert resumed.demo.transitions == trainer.demo.transitions
    assert [lap["lap_ms"] for lap in resumed.best_laps] == [ep["lap_ms"], ep["lap_ms"] + 10]


def test_timed_out_attempt_keeps_last_observation(tmp_path):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    class FlakyStep(FakeGame):
        def observe(self):
            if self.t >= 100:
                raise TimeoutError("telemetry paused")
            return super().observe()

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FlakyStep(), device=torch.device("cpu"), log=lambda *_: None)
    env = TMEnv(cfg, trainer.game, trainer.tracker)
    with pytest.raises(TimeoutError):
        trainer.run_episode(env)
    assert trainer.online.size == 2 and trainer.online.transitions == 1
    sample = trainer.online.sample(4, np.random.default_rng(0))
    assert (sample["discount"] > 0).all()
    np.testing.assert_array_equal(sample["next_state"][0], trainer.online.state[1])


def test_eval_refreshes_current_policy_once_and_uses_it_for_the_whole_lap(tmp_path, monkeypatch):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    parameter = next(trainer.agent.actor.parameters())
    initial = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1)
    calls = []

    def drive(frames, state, deterministic):
        assert deterministic
        calls.append(next(trainer.policy.actor.parameters()).clone())
        with torch.no_grad():
            parameter.add_(1)
        trainer._pending_policy = trainer.agent.policy_state()
        return np.array([0, 1, -1])

    monkeypatch.setattr(trainer.policy, "act", drive)
    ep = trainer.run_episode(TMEnv(cfg, trainer.game, trainer.tracker), deterministic=True)
    assert ep["end"] == "finished" and len(calls) > 1
    assert all(torch.equal(weight, initial + 1) for weight in calls)


def test_interpolated_finish_time_reaches_eval_guide_and_adopted_reference(tmp_path, monkeypatch):
    import csv
    from io import StringIO

    from tmai.rl.env import TMEnv
    from tmai.rl.train import EPISODE_FIELDS, Trainer

    cfg = small_cfg(tmp_path, best_laps_kept=1)
    cfg["track"]["adopt_best_gain_ms"] = 1
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    trainer.tracker.ref.times *= 2
    monkeypatch.setattr(trainer.policy, "act", lambda *a: np.array([0, 1.0, -1.0]))
    ep = trainer.run_episode(TMEnv(cfg, trainer.game, trainer.tracker), deterministic=True)
    assert ep["lap_ms"] == 2440 and ep["trace"][-1][1] == 2440
    assert ep["lap"]["tels"][-1].race_time_ms == 2450 and ep["race_time_s"] == 2.44
    log = StringIO()
    row = trainer._log_episode(ep, csv.DictWriter(log, fieldnames=EPISODE_FIELDS), log)
    assert row["lap_ms"] == trainer.best_ms == 2440
    assert trainer.record_eval(ep, row)
    assert trainer.keep_best_lap(ep) and trainer.best_eval_ms == trainer.best_laps[0]["lap_ms"] == 2440
    assert trainer.maybe_adopt_lap(ep) and trainer.tracker.ref.times[-1] == pytest.approx(2.44)


@pytest.mark.parametrize("paused", [False, True])
def test_minutes_expires_during_pause_or_timeout_retry(tmp_path, monkeypatch, paused):
    from types import SimpleNamespace

    from tmai.rl import train as train_module
    from tmai.rl.train import Trainer
    from tmai.run.hooks import TrainerHooks
    from test_game_logic import FakeTime

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    clock = FakeTime()
    monkeypatch.setattr(train_module, "time", SimpleNamespace(monotonic=clock.now))
    monkeypatch.setattr(trainer.stop, "wait", clock.sleep)
    monkeypatch.setattr(trainer, "_learner", lambda: None)
    attempts = []

    def attempt(*args, **kwargs):
        attempts.append(clock.now())
        raise TimeoutError("telemetry paused")

    hooks = TrainerHooks()
    monkeypatch.setattr(hooks, "should_pause", lambda: paused)
    monkeypatch.setattr(trainer, "run_episode", attempt)
    assert trainer.train(minutes=0.01, hooks=hooks) == 0
    assert clock.now() == pytest.approx(0.6)
    assert attempts == ([] if paused else [0.0])


def test_test_laps_drive_without_noise_and_keep_the_best(tmp_path):
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1, eval_every=2)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    trainer.train(minutes=0.03)
    run_dir = tmp_path / "default"
    rows = (run_dir / "eval.csv").read_text().splitlines()
    assert rows[0].startswith("episode,") and len(rows) >= 2
    assert trainer.best_eval_ms is not None
    best = torch.load(run_dir / "checkpoints" / "best_test_lap.pt", weights_only=False)
    assert best["lap_ms"] == trainer.best_eval_ms and "actor" in best["policy"]


def test_finished_laps_feed_the_pace_baseline(tmp_path):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path)
    cfg["reward"]["pace_baseline_min_laps"] = 2
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert trainer.tracker.baseline is trainer.baseline and not trainer.baseline.from_recent_laps
    env = TMEnv(cfg, trainer.game, trainer.tracker)
    laps = []
    for _ in range(2):
        ep = trainer.run_episode(env)
        assert ep["end"] == "finished"
        trainer.add_baseline_lap(ep)
        laps.append(ep["lap_ms"] / 1000)
    assert trainer.baseline.from_recent_laps
    assert trainer.baseline.lap_time_s == pytest.approx(np.mean(laps), abs=0.15)   # within a step or two of the line's end
    # stored rewards leave the pace term out; sampling adds it back from the inputs
    stored = ~np.isnan(trainer.online.pace_dt[:trainer.online.size])
    assert stored.any()
    batch = trainer.sample()
    assert np.isfinite(batch["reward"]).all()
    # a resumed run starts from the same baseline, not from scratch
    trainer.save_checkpoint()
    resumed = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert resumed.baseline.lap_count == 2
    assert resumed.baseline.lap_time_s == pytest.approx(trainer.baseline.lap_time_s)


def test_stop_or_pause_during_restart_aborts_without_holding_gas(tmp_path):
    from tmai.game.interface import RestartAborted
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg)
    trainer = Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
    env = TMEnv(cfg, trainer.game, trainer.tracker)
    episodes_before = trainer.online._episode

    class PauseHooks:
        def on_step(self, *_):
            pass

        def should_reset_episode(self):
            return False

        def should_pause(self):
            return True

    trainer.game.pad.last = (0.0, 1.0, 0.0)   # as if gas were held from the countdown
    with pytest.raises(RestartAborted):
        trainer.run_episode(env, PauseHooks())
    assert trainer.game.pad.last == (0.0, 0.0, 0.0)          # pad let go
    assert trainer.online._episode == episodes_before         # no attempt was started

    trainer.stop.set()
    with pytest.raises(RestartAborted):
        trainer.run_episode(env)
    assert trainer.game.pad.last == (0.0, 0.0, 0.0)


def test_training_refuses_the_cpu_unless_allowed(tmp_path, monkeypatch):
    from tmai.rl import train as train_mod

    cfg = small_cfg(tmp_path)
    make_track(tmp_path, cfg)
    monkeypatch.setattr(train_mod, "pick_device", lambda: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        train_mod.Trainer(cfg, game=FakeGame(), log=lambda *_: None)
    cfg["gpu"]["allow_cpu"] = True
    trainer = train_mod.Trainer(cfg, game=FakeGame(), log=lambda *_: None)
    assert trainer.device.type == "cpu"
