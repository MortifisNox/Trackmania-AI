import numpy as np
import pytest

from tmai.config import load_config
from tmai.game import protocol
from tmai.track.demo import LapBuffer, record_laps
from tmai.track.progress import ProgressTracker
from tmai.track.reference import ReferenceLine, resample
from tmai.track.splits import SplitTracker
from tmai.track.store import TrackStore

PLAYING = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING


def tel(**kw):
    kw.setdefault("flags", PLAYING)
    kw.setdefault("dir", (1.0, 0.0, 0.0))
    kw.setdefault("up", (0.0, 1.0, 0.0))
    return protocol.decode(protocol.encode(**kw))


def straight(length=200.0):
    xs = np.linspace(0, length, 50)
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def test_resample_spacing_and_duplicates():
    pts = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 10]], dtype=float)
    out = resample(pts, 1.0)
    gaps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(gaps, 1.0) and np.allclose(out[-1], [10, 0, 10])
    with pytest.raises(ValueError):
        resample(np.zeros((5, 3)), 1.0)


def test_project_straight_line():
    ref = ReferenceLine.from_positions(straight(), 1.0)
    assert ref.length == pytest.approx(200.0)
    idx, s, lat = ref.project((50.4, 0.0, 3.0))
    assert s == pytest.approx(50.4) and lat == pytest.approx(3.0)


def test_hint_keeps_the_right_branch_where_the_track_crosses_itself():
    # an out-and-back loop: points at s=20 and s~180 are both near (20, 0, 0)
    t = np.linspace(0, np.pi, 100)
    loop = np.stack([80 * np.sin(t) * 0 + 20 + 60 * (1 - np.cos(t)), np.zeros_like(t), 30 * np.sin(t)], axis=1)
    back = loop[::-1].copy()
    back[:, 2] = -back[:, 2] * 0.01
    path = np.vstack([loop, back[1:]])
    ref = ReferenceLine.from_positions(path, 1.0)
    first, s_first, _ = ref.project(path[0])
    _, s_late, _ = ref.project(path[-1] + (0.1, 0, 0), hint=len(ref.points) - 10)
    assert s_first < 5 and s_late > ref.length - 5


def test_lookahead_in_car_frame():
    ref = ReferenceLine.from_positions(straight(), 1.0)
    pts = ref.lookahead((10, 0, 0), (1, 0, 0), (0, 1, 0), 10.0, count=2, spacing=5.0).reshape(2, 3)
    assert np.allclose(pts[0], [5, 0, 0]) and np.allclose(pts[1], [10, 0, 0])
    # car facing +z with y up: left = up x forward = +x, where the line continues
    pts = ref.lookahead((10, 0, 0), (0, 0, 1), (0, 1, 0), 10.0, count=1, spacing=5.0)
    assert pts[0] == pytest.approx(0) and pts[1] == pytest.approx(5)


def test_reference_save_load(tmp_path):
    ref = ReferenceLine.from_positions(straight(), 2.0)
    ref.save(tmp_path / "r.npz")
    back = ReferenceLine.load(tmp_path / "r.npz")
    assert back.spacing == 2.0 and np.allclose(back.points, ref.points)


def test_splits_deltas_and_best(tmp_path):
    sp = SplitTracker([10_000, 20_000])
    assert sp.update(0, 0) is None
    assert sp.update(1, 9_500) == -500
    assert sp.update(1, 9_500) is None  # same checkpoint again
    assert sp.update(2, 20_400) == 400
    assert sp.finish() is False         # slower at the finish
    sp.reset()
    sp.update(1, 9_000)
    sp.update(2, 19_000)
    assert sp.finish() is True and sp.best == [9_000, 19_000]
    sp.save(tmp_path / "s.json")
    assert SplitTracker.load(tmp_path / "s.json").best == [9_000, 19_000]
    assert SplitTracker.load(tmp_path / "missing.json").best == []


def make_tracker(best=None):
    cfg = load_config()
    ref = ReferenceLine.from_positions(straight(), 1.0)
    return ProgressTracker(ref, cfg, SplitTracker(best)), cfg


def test_progress_reward_terms():
    tr, cfg = make_tracker()
    tr.reset(tel(pos=(0.0, 0.0, 0.0), speed=0.0))
    info = tr.step(tel(pos=(5.0, 0.0, 1.0), speed=20.0))
    assert info.delta_m == pytest.approx(5.0)
    assert info.reward == pytest.approx(0.5 - cfg["reward"]["time_penalty"])
    assert info.lookahead.shape == (cfg["env"]["lookahead_points"] * 3,)
    crash = tr.step(tel(pos=(5.5, 0.0, 1.0), speed=20.0 - 50 / 3.6))
    assert crash.crashed and "crash" in crash.terms


def test_teleport_gets_no_credit_and_wrong_way_ends():
    tr, _ = make_tracker()
    tr.reset(tel(pos=(0.0, 0.0, 0.0)))
    assert tr.step(tel(pos=(150.0, 0.0, 0.0))).delta_m == 0.0
    tr.reset(tel(pos=(100.0, 0.0, 0.0)))
    for x in (95.0, 90.0, 85.0, 80.0):
        info = tr.step(tel(pos=(x, 0.0, 0.0)))
    assert info.done is None
    assert tr.step(tel(pos=(74.0, 0.0, 0.0))).done == "wrong_way"


def test_split_bonus_and_finish():
    tr, cfg = make_tracker(best=[10_000])
    tr.reset(tel(pos=(0.0, 0.0, 0.0)))
    info = tr.step(tel(pos=(1.0, 0.0, 0.0), cp_count=1, last_cp_time_ms=9_000))
    assert info.split_delta_ms == -1000 and info.terms["split"] == pytest.approx(1.0)
    fin = tr.step(tel(pos=(2.0, 0.0, 0.0), flags=protocol.FLAG_PLAYER | protocol.FLAG_FINISHED))
    assert fin.done == "finished" and fin.terms["finish"] == cfg["reward"]["finish_bonus"]
    after = tr.step(tel(pos=(40.0, 30.0, 0.0), flags=protocol.FLAG_PLAYER | protocol.FLAG_FINISHED))
    assert after.done == "finished" and after.reward == 0.0
    tr.reset(tel(pos=(0.0, 0.0, 0.0)))
    assert tr.step(tel(pos=(1.0, 0.0, 0.0))).done is None


def test_track_store(tmp_path):
    with pytest.raises(ValueError):
        TrackStore(tmp_path, "../escape")
    store = TrackStore(tmp_path, "My Map 1")
    with pytest.raises(FileNotFoundError):
        store.best_demo()
    for ms in (30_000, 25_000):
        buf = LapBuffer()
        buf.add(np.zeros((64, 64), np.uint8), tel(pos=(0.0, 0.0, 0.0)))
        buf.add(np.zeros((64, 64), np.uint8), tel(pos=(5.0, 0.0, 0.0), cp_count=1, last_cp_time_ms=ms))
        path = buf.save(store.demos, ms)
        path.rename(path.with_name(f"{ms}.npz"))
    assert store.best_demo().name == "25000.npz"
    with np.load(store.best_demo()) as d:
        assert d["frames"].shape == (2, 64, 64) and list(d["cp_times"]) == [25_000]


class FakeGame:
    def __init__(self, packets):
        self.packets = iter(packets)

    def observe(self):
        return np.zeros((64, 64), np.uint8), next(self.packets)


def test_record_laps_drops_restarted_attempts(tmp_path):
    fin = protocol.FLAG_PLAYER | protocol.FLAG_FINISHED
    run = lambda n: [tel(race_time_ms=i * 50, pos=(float(i), 0.0, 0.0)) for i in range(n)]
    packets = (run(30)                                   # attempt 1: restarted
               + run(40)                                 # attempt 2: finishes
               + [tel(flags=fin, race_time_ms=2000, cp_count=1, last_cp_time_ms=2000, pos=(40.0, 0.0, 0.0))])
    saved = record_laps(FakeGame(packets), tmp_path, laps=1, step_hz=1000, log=lambda *_: None)
    assert len(saved) == 1
    with np.load(saved[0]) as d:
        assert int(d["lap_time_ms"]) == 2000 and len(d["actions"]) == 41


def test_reference_times_roundtrip_and_pace_reward(tmp_path):
    xs = np.linspace(0, 100, 101)
    pts = np.stack([xs, 0 * xs, 0 * xs], 1)
    ref = ReferenceLine.from_positions(pts, 1.0, xs / 10.0)   # reference lap: 10 m/s
    assert ref.time_at(50.0) == pytest.approx(5.0)
    ref.save(tmp_path / "r.npz")
    assert ReferenceLine.load(tmp_path / "r.npz").time_at(80.0) == pytest.approx(8.0)
    cfg = load_config(None, ["reward.pace_per_s=10"])
    tr = ProgressTracker(ref, cfg)
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=0))
    fast = tr.step(tel(pos=(2.0, 0.0, 0.0), race_time_ms=100))    # 20 m/s: 0.1 s gained
    assert fast.terms["pace"] == pytest.approx(1.0)
    slow = tr.step(tel(pos=(2.5, 0.0, 0.0), race_time_ms=200))    # 5 m/s: 0.05 s lost
    assert slow.terms["pace"] == pytest.approx(-0.5)
    plain = ProgressTracker(ReferenceLine.from_positions(pts, 1.0), cfg)
    plain.reset(tel(pos=(0.0, 0.0, 0.0)))
    assert "pace" not in plain.step(tel(pos=(2.0, 0.0, 0.0), race_time_ms=100)).terms


def test_stalls_and_late_starts_cost_the_time_they_lose():
    xs = np.linspace(0, 100, 101)
    ref = ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0, xs / 10.0)   # 10 m/s
    cfg = load_config(None, ["reward.pace_per_s=10", "reward.time_penalty=0.01", "env.step_hz=20"])
    tr = ProgressTracker(ref, cfg)
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=0))
    normal = tr.step(tel(pos=(0.5, 0.0, 0.0), race_time_ms=50))        # on the reference pace
    assert normal.terms["time"] == pytest.approx(-0.01) and "pace" not in normal.terms
    stall = tr.step(tel(pos=(1.0, 0.0, 0.0), race_time_ms=1050))       # 1 s on the clock, one step driven
    assert stall.terms["pace"] == pytest.approx(10 * (0.05 - 1.0))     # the whole 0.95 s lost, not capped at 0.2
    assert stall.terms["time"] == pytest.approx(-0.01 * 20)
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=1000))              # clock already at 1 s when driving starts
    late = tr.step(tel(pos=(0.5, 0.0, 0.0), race_time_ms=1050))
    assert late.terms["pace"] == pytest.approx(10 * (0.05 - 1.05))
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=-1500))             # countdown frames (demos) cost nothing extra
    countdown = tr.step(tel(pos=(0.0, 0.0, 0.0), race_time_ms=-1450))
    assert "pace" not in countdown.terms and countdown.terms["time"] == pytest.approx(-0.01)


def test_pace_baseline_follows_the_average_of_recent_laps():
    from tmai.track.baseline import PaceBaseline

    xs = np.linspace(0, 100, 101)
    line = np.stack([xs, 0 * xs, 0 * xs], 1)
    ref = ReferenceLine.from_positions(line, 1.0, xs / 10.0)              # reference lap: 10 s at 10 m/s
    base = PaceBaseline(ref, laps=3, min_laps=2)
    assert not base.from_recent_laps and base.lap_time_s == pytest.approx(10.0)
    assert base.pace(0.0, 0.1, 1.0) == pytest.approx(0.0)                # on reference pace
    base.add_lap(line, xs / 8.0)                                          # 12.5 s lap: one is not enough
    assert base.lap_time_s == pytest.approx(10.0)
    base.add_lap(line, xs / 5.0)                                          # 20 s lap
    assert base.from_recent_laps and base.lap_time_s == pytest.approx(16.25)
    # 10 m in 1 s: 1.625 s for the baseline, but gains are capped at 0.2 s a step
    assert base.pace(0.0, 0.1, 1.0) == pytest.approx(0.2)
    assert base.pace(0.0, 0.01, 0.1) == pytest.approx(0.1625 - 0.1)
    for lap in (xs / 10.0,) * 3:                                          # only the last 3 laps count
        base.add_lap(line, lap)
    assert base.lap_count == 3 and base.lap_time_s == pytest.approx(10.0)
    np.testing.assert_allclose(base.pace(np.array([0.0, 0.5]), np.array([0.01, 0.5]), np.array([0.05, 0.05])),
                               [0.05, -0.05])
    longer = ReferenceLine.from_positions(np.stack([2 * xs, 0 * xs, 0 * xs], 1), 1.0, xs / 5.0)
    base.set_ref(longer)                                                  # laps measured along the new line
    assert base.lap_count == 3 and base.lap_time_s == pytest.approx(10.0)


def test_personal_best_bonus_at_the_finish():
    cfg = load_config(None, ["reward.pb_bonus_per_s=20", "reward.pb_bonus_cap=5"])
    tr = ProgressTracker(ReferenceLine.from_positions(straight(), 1.0), cfg)
    fin = protocol.FLAG_PLAYER | protocol.FLAG_FINISHED
    for best, lap, bonus in ((None, 20_000, None), (21_000, 21_500, None), (21_000, 20_900, 2.0), (21_000, 20_000, 5.0)):
        tr.best_lap_ms = best
        tr.reset(tel(pos=(0.0, 0.0, 0.0)))
        info = tr.step(tel(pos=(1.0, 0.0, 0.0), flags=fin, race_time_ms=lap))
        assert info.terms.get("personal_best") == (None if bonus is None else pytest.approx(bonus))
