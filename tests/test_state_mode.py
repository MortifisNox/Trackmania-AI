"""env.observation=state: no camera, the policy drives from game state alone."""

import numpy as np
import pytest

from test_rl import LENGTH, PLAYING, make_track, small_cfg, tel
from tmai.config import ConfigError, load_config
from tmai.game import protocol
from tmai.rl.features import frame_size, lookahead_distances, state_dim, state_vector, uses_camera
from tmai.rl.replay import ReplayBuffer
from tmai.run import snapshot

STATE_SETS = ["env.observation=state", "env.lookahead_points=20", "env.lookahead_spacing_m=5",
              "env.lookahead_growth=1.1"]


def state_cfg(tmp_path, **train):
    return load_config(None, [f"run.output_dir={tmp_path.as_posix()}", "run.track=T", "replay.capacity=2000",
                              "train.batch_size=16", "model.hidden_dim=32", "env.step_hz=1000",
                              "env.episode_timeout_s=5", *STATE_SETS,
                              *(f"train.{k}={v}" for k, v in train.items())])


def test_state_mode_sizes_and_vector():
    env = load_config(None, STATE_SETS)["env"]
    assert not uses_camera(env) and frame_size(env) == 0
    assert state_dim(env) == 18 + 60 + 3
    far = lookahead_distances(env)
    assert far[0] == pytest.approx(5.0) and 250 < far[-1] < 300   # reaches ~286 m up the road
    level = tel(speed=50.0, vel=(0.0, 0.0, 50.0), dir=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0))
    points = np.tile([100.0, 0.0, 0.0], 20)
    s = state_vector(level, points, (0.0, 1.0, -1.0), env)
    assert s.shape == (state_dim(env),) and np.all(np.abs(s) <= 5.0)
    assert s[-3:] == pytest.approx([0.0, 0.0, 1.0])     # level car: world up is the car's up
    nose_down = tel(dir=(0.0, -1.0, 1.0), up=(0.0, 1.0, 1.0))
    assert state_vector(nose_down, points, (0, 0, 0), env)[-3] < -0.5   # nose down: world up leans back


def test_image_mode_is_unchanged():
    env = load_config(None, [])["env"]
    assert uses_camera(env) and frame_size(env) == 64 and state_dim(env) == 48
    with pytest.raises(ConfigError):
        load_config(None, ["env.lookahead_points=20"])
    with pytest.raises(ConfigError):
        load_config(None, ["env.observation=pixels"])


def test_empty_frame_buffer_samples_and_snapshots(tmp_path):
    env = load_config(None, STATE_SETS)["env"]
    buf = ReplayBuffer(100, frame_size(env), 4, state_dim(env))
    for ep in range(3):
        buf.start_episode()
        for i in range(10):
            buf.add(np.zeros((0, 0), np.uint8), np.full(state_dim(env), i, np.float32), np.zeros(3), 1.0)
        buf.add(np.zeros((0, 0), np.uint8), np.zeros(state_dim(env)))
    batch = buf.sample(8, np.random.default_rng(0), n_step=3, gamma=0.99)
    assert batch["frames"].shape == (8, 4, 0, 0) and batch["state"].shape == (8, state_dim(env))
    snapshot.save_replay(buf, tmp_path)
    fresh = ReplayBuffer(100, 0, 4, state_dim(env))
    assert snapshot.load_replay(fresh, tmp_path) is not None
    np.testing.assert_array_equal(fresh.state, buf.state)


torch = pytest.importorskip("torch")


class StateGame:
    """FakeGame without a camera: observe() returns no frame."""

    def __init__(self):
        from test_rl_torch import FakePad

        self.pad = FakePad()
        self.x, self.t = 0.0, 0

    def restart(self):
        self.x, self.t = 0.0, 0

    def act(self, controls):
        self.pad.last = controls
        return controls

    def observe(self):
        self.x += 0.05 * (5.0 + 20.0 * self.pad.last[1])
        self.t += 50
        flags = PLAYING | (protocol.FLAG_FINISHED if self.x >= LENGTH else 0)
        speed = 5.0 + 20.0 * self.pad.last[1]
        return None, tel(pos=(self.x, 0.0, 0.0), speed=speed, vel=(speed, 0.0, 0.0), race_time_ms=self.t, flags=flags)


def test_trainer_learns_without_a_camera_and_uses_old_demos(tmp_path):
    from tmai.rl.env import TMEnv
    from tmai.rl.train import Trainer

    cfg = state_cfg(tmp_path, warmup_steps=20, updates_per_step=1)
    make_track(tmp_path, cfg)          # demo recorded with camera frames: still used
    lines = []
    trainer = Trainer(cfg, game=StateGame(), device=torch.device("cpu"), log=lines.append)
    assert trainer.demo is not None and trainer.demo_n > 0
    assert trainer.online.frames.shape[1:] == (0, 0)
    env = TMEnv(cfg, trainer.game, trainer.tracker)
    steps = 0
    for _ in range(3):
        ep = trainer.run_episode(env)
        steps += ep["steps"]
        while trainer.learn_once():
            pass
    assert trainer.agent.updates == steps - 20
    trainer.save_checkpoint()
    again = Trainer(cfg, game=StateGame(), device=torch.device("cpu"), log=lines.append)
    assert again.agent.updates == trainer.agent.updates

    image_cfg = small_cfg(tmp_path)    # same run folder, other mode: refused with a clear message
    with pytest.raises(FileNotFoundError, match="observation"):
        Trainer(image_cfg, game=StateGame(), device=torch.device("cpu"), log=lines.append)
