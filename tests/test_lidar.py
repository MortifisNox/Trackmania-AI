import numpy as np
import pytest

from tmai.budget import STATE_DIM, state_dim
from tmai.config import ConfigError, load_config
from tmai.game.lidar import Lidar, from_config, wide_size
from tmai.rl.demos import load_demo
from tmai.rl.features import state_vector
from tmai.rl.replay import ReplayBuffer
from tmai.track.progress import ProgressTracker
from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker
from tmai.track.store import TrackStore

from test_rl import make_track, small_cfg, tel


def road(h=144, w=256, left=40, right=216):
    img = np.full((h, w), 200, np.uint8)
    img[:, :left] = 10
    img[:, right:] = 10
    return img


def test_beams_stop_at_dark_borders():
    lidar = Lidar(3, origin=(0.5, 0.9))
    img = road()
    left, ahead, right = lidar.measure(img)
    oy = 0.9 * 143
    assert ahead == pytest.approx(round(oy) / 144, abs=0.02)        # clear road up to the top
    assert left == pytest.approx((128 - 40) / 144, abs=0.02)        # stops at the left border
    assert right == pytest.approx((216 - 128) / 144, abs=0.02)
    img[:50, :] = 10                                                # a wall ahead
    assert lidar.measure(img)[1] == pytest.approx((round(oy) - 49) / 144, abs=0.02)


def test_single_dark_pixels_do_not_stop_a_beam():
    lidar = Lidar(1, origin=(0.5, 0.9))
    img = road()
    img[60, 128] = 0                                                # noise on the road
    assert lidar.measure(img)[0] > 0.8


def test_endpoints_land_on_the_beam_end():
    lidar = Lidar(3, origin=(0.5, 0.9))
    img = road()
    ends = lidar.endpoints(img, lidar.measure(img))
    assert ends[0][1][0] == pytest.approx(40, abs=2) and ends[2][1][0] == pytest.approx(215, abs=2)


def test_lidar_config(tmp_path):
    cfg = small_cfg(tmp_path)
    assert from_config(cfg["env"]) is None and state_dim(cfg["env"]) == STATE_DIM
    cfg = load_config()
    assert not cfg["env"]["camera"]                                 # default: beams, no image
    assert from_config(cfg["env"]).beams == 19 and state_dim(cfg["env"]) == STATE_DIM + 19
    cfg = load_config(None, ["env.lidar_far_origin=[0.5, 0.52]"])
    assert from_config(cfg["env"]).beams == 38 and state_dim(cfg["env"]) == STATE_DIM + 38   # near + far fan
    assert wide_size(1920, 1080, 256) == (256, 144)
    with pytest.raises(ConfigError):
        load_config(None, ["env.lidar_origin=[0.5, 1.5]"])


def test_state_vector_appends_beams():
    s = state_vector(tel(), np.zeros(30), (0.0, -1.0, -1.0), np.array([0.25, 1.0], np.float32))
    assert s.shape == (STATE_DIM + 2,) and s[-2] == pytest.approx(0.25)


def test_demo_without_beams_is_skipped_when_lidar_is_on(tmp_path):
    cfg = small_cfg(tmp_path)
    store = make_track(tmp_path, cfg)
    tracker = ProgressTracker(ReferenceLine.load(store.reference), cfg, SplitTracker())
    demo = TrackStore(cfg["run"]["output_dir"], cfg["run"]["track"]).demo_files()[0]
    assert load_demo(demo, ReplayBuffer(100, 64, 4, STATE_DIM + 5), tracker) == (0, 0.0)
    n, _ = load_demo(demo, ReplayBuffer(100, 64, 4), tracker)
    assert n > 0


def test_far_fan_sees_a_wall_before_the_near_fan():
    cfg = load_config(None, ["env.lidar_far_origin=[0.5, 0.52]"])
    lidar = from_config(cfg["env"])
    img = road()
    img[:72, :] = 10                                                # a wall halfway up the picture
    lengths = lidar.measure(img)
    near, far = lengths[:19], lengths[19:]
    assert far[9] < 0.05 < near[9]                                  # straight ahead: far fan starts at the wall
    assert len(lidar.endpoints(img, lengths)) == 2
