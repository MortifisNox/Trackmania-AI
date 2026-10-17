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

    def take_input_interruption(self):
        return None

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


class LidarGame(FakeGame):
    """FakeGame that also reports lidar beams, as GameInterface does."""

    beams = np.linspace(0.2, 1.0, 5).astype(np.float32)


def test_trainer_with_lidar_and_no_camera(tmp_path):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer
    from tmai.track.demo import LapBuffer
    from tmai.track.store import TrackStore

    cfg = small_cfg(tmp_path, warmup_steps=10, updates_per_step=1)
    make_track(tmp_path, cfg, with_demo=False)
    buf = LapBuffer()
    for i in range(31):
        x = 2.0 * i
        flags = PLAYING | (protocol.FLAG_FINISHED if x >= LENGTH else 0)
        buf.add(np.full((64, 64), i, np.uint8),
                tel(pos=(x, 0.0, 0.0), speed=40.0, vel=(40.0, 0.0, 0.0), race_time_ms=50 * i,
                    steer=0.0, gas=1.0, flags=flags), LidarGame.beams)
    buf.save(TrackStore(cfg["run"]["output_dir"], cfg["run"]["track"]).demos, 1500)

    cfg["env"]["lidar_beams"] = 5
    cfg["env"]["camera"] = False
    trainer = Trainer(cfg, game=LidarGame(), device=torch.device("cpu"), log=lambda *_: None)
    assert trainer.demo is not None and trainer.demo.state.shape[1] == STATE_DIM + 5
    env = TMEnv(cfg, trainer.game, trainer.tracker)
    obs = env.reset()
    assert obs.state.shape == (STATE_DIM + 5,)
    assert not any(True for _ in trainer.agent.encoder.parameters())   # no CNN without the camera
    ep = trainer.run_episode(env)
    assert ep["steps"] > 0
    while trainer.learn_once():
        pass
    trainer.save_checkpoint()

    cfg["env"]["lidar_beams"] = 0
    with pytest.raises(RuntimeError, match="different inputs"):
        Trainer(cfg, game=FakeGame(), device=torch.device("cpu"), log=lambda *_: None)
