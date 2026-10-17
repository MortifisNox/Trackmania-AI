import numpy as np
import pytest

from tmai.budget import STATE_DIM
from tmai.config import load_config
from tmai.game import protocol
from tmai.rl.features import from_controls, state_vector, to_controls
from tmai.rl.replay import ReplayBuffer
from tmai.track.demo import LapBuffer
from tmai.track.progress import ProgressTracker
from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker
from tmai.track.store import TrackStore

PLAYING = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING | protocol.FLAG_GROUND
LENGTH = 60.0


def tel(**kw):
    kw.setdefault("flags", PLAYING)
    kw.setdefault("dir", (1.0, 0.0, 0.0))
    kw.setdefault("up", (0.0, 1.0, 0.0))
    return protocol.decode(protocol.encode(**kw))


def small_cfg(tmp_path, **train):
    over = [f"run.output_dir={tmp_path.as_posix()}", "run.track=T", "replay.capacity=2000",
            "train.batch_size=16", "model.hidden_dim=32", "model.encoder_channels=[8,8,8,8]",
            "env.step_hz=1000", "env.episode_timeout_s=5", "env.camera=true", "env.lidar_beams=0"]
    over += [f"train.{k}={v}" for k, v in train.items()]
    return load_config(None, over)


def make_track(tmp_path, cfg, with_demo=True):
    store = TrackStore(cfg["run"]["output_dir"], cfg["run"]["track"])
    xs = np.linspace(0, LENGTH, 30)
    ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0).save(store.reference)
    SplitTracker().save(store.splits)
    if with_demo:
        buf = LapBuffer()
        for i in range(31):
            x = 2.0 * i
            flags = PLAYING | (protocol.FLAG_FINISHED if x >= LENGTH else 0)
            buf.add(np.full((64, 64), i, np.uint8),
                    tel(pos=(x, 0.0, 0.0), speed=40.0, vel=(40.0, 0.0, 0.0), race_time_ms=50 * i,
                        steer=0.0, gas=1.0, flags=flags))
        buf.save(store.demos, 1500)
    return store


def test_state_vector_and_controls():
    s = state_vector(tel(speed=50.0, vel=(0.0, 0.0, 50.0), dir=(0.0, 0.0, 1.0)), np.zeros(30), (0.0, 1.0, -1.0))
    assert s.shape == (STATE_DIM,) and s.dtype == np.float32
    assert s[3] == pytest.approx(1.0)                 # forward velocity in the car frame
    assert to_controls((0.5, 1.0, -1.0)) == (0.5, 1.0, 0.0)
    assert np.allclose(from_controls(*to_controls((0.3, -0.2, 0.6))), (0.3, -0.2, 0.6), atol=1e-6)


def test_replay_stacks_within_episode():
    buf = ReplayBuffer(8, 4, stack=3, state_dim=STATE_DIM)
    for ep in range(2):
        buf.start_episode()
        for i in range(3):
            v = 10 * (ep + 1) + i
            buf.add(np.full((4, 4), v, np.uint8), np.zeros(STATE_DIM), np.zeros(3), 1.0)
        buf.add(np.full((4, 4), 10 * (ep + 1) + 3, np.uint8), np.zeros(STATE_DIM))
    # entry 4 is the first observation of episode 2: the stack repeats it
    assert list(buf.stacked(4)[:, 0, 0]) == [20, 20, 20]
    assert list(buf.stacked(6)[:, 0, 0]) == [20, 21, 22]
    batch = buf.sample(64, np.random.default_rng(0))
    last, nxt = batch["frames"][:, -1, 0, 0], batch["next_frames"][:, -1, 0, 0]
    assert np.all(nxt == last + 1)                    # never crosses an episode end
    assert not np.isin(last, [13, 23]).any()          # final observations are not starts
    # wrap around: overwritten episodes do not leak into stacks
    buf.start_episode()
    for i in range(3):
        buf.add(np.full((4, 4), 30 + i, np.uint8), np.zeros(STATE_DIM), np.zeros(3), 1.0)
    batch = buf.sample(64, np.random.default_rng(1))
    assert np.all(batch["next_frames"][:, -1, 0, 0] == batch["frames"][:, -1, 0, 0] + 1)


def test_demo_loads_with_rewards(tmp_path):
    from tmai.rl.demos import load_demo

    cfg = small_cfg(tmp_path)
    store = make_track(tmp_path, cfg)
    buf = ReplayBuffer(100, 64, 4)
    tracker = ProgressTracker(ReferenceLine.load(store.reference), cfg)
    n, total = load_demo(store.demo_files()[0], buf, tracker)
    assert n == 30 and total > 5.0
    assert buf.terminal[n - 1] == 1.0 and buf.reward[n - 1] > cfg["reward"]["finish_bonus"] - 1
    assert np.allclose(buf.action[0], (0.0, 1.0, -1.0))
