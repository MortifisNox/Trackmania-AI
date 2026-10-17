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


class FakePad:
    def __init__(self):
        self.last = (0.0, 0.0, 0.0)

    def release(self):
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

    cfg = small_cfg(tmp_path, demo_anneal_steps=100, entropy_anneal_steps=100)
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
