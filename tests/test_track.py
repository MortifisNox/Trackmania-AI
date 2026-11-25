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
    tr.reset(tel(pos=(0.0, 0.0, 0.0)))
    for x in (25, 50, 75, 100):
        tr.step(tel(pos=(x, 0, 0)))
    for x in (95.0, 90.0, 85.0, 80.0):
        info = tr.step(tel(pos=(x, 0.0, 0.0)))
    assert info.done is None
    assert tr.step(tel(pos=(74.0, 0.0, 0.0))).done == "wrong_way"


def test_start_projection_ignores_neighboring_finish():
    ref = ReferenceLine.from_positions([(0, 0, 0), (100, 0, 0), (100, 0, 5), (0, 0, 5)], 1)
    tr = ProgressTracker(ref, load_config())
    spawn = tel(pos=(0, 0, 4))
    assert ref.project(spawn.pos)[1] > 200
    tr.reset(spawn)
    assert tr.step(tel(pos=(2, 0, 0))).delta_m == pytest.approx(2)


def test_hinted_projection_stays_local_and_progress_resyncs_after_ten_jumps():
    cfg = load_config()
    ref = ReferenceLine.from_positions(straight(), 1.0, np.linspace(0, 20, 50))
    tr = ProgressTracker(ref, cfg)
    tr.reset(tel())
    for x in range(20, 200, 20):
        tr.step(tel(pos=(x, 0, 0)))
    old_idx = tr._idx
    assert ref.project((30, 0, 0), hint=old_idx)[1] > 150
    for _ in range(9):
        info = tr.step(tel(pos=(30, 0, 0)))
        assert info.done is None and info.delta_m == 0 and info.progress_m == 180
        assert tr._idx == old_idx and info.pace_inputs[:2] == (0.9, 0.9)
    # One good sample breaks the streak.
    assert tr.step(tel(pos=(181, 0, 0))).delta_m == 1
    for _ in range(9):
        assert tr.step(tel(pos=(30, 0, 0))).progress_m == 181
    synced = tr.step(tel(pos=(30, 0, 0)))
    assert synced.done is None and synced.delta_m == 0 and synced.progress_m == 30
    assert tr.step(tel(pos=(31, 0, 0))).delta_m == 1


def test_hitch_credits_local_forward_progress_and_keeps_lookahead_current():
    ref = ReferenceLine.from_positions(straight(), 1.0, np.linspace(0, 200 / 136, 50))
    cfg = load_config(overrides=["reward.pace_per_s=10", "env.step_hz=20"])
    tr = ProgressTracker(ref, cfg)
    tr.reset(tel())

    speed = 480 / 3.6
    hitch = tr.step(tel(pos=(34, 0, 0), speed=speed, race_time_ms=250))
    assert hitch.delta_m == pytest.approx(34)
    assert hitch.pace_inputs == pytest.approx((0, 34 / 200, 0.25))
    assert hitch.terms.get("pace", 0.0) == pytest.approx(0.0)
    assert hitch.lookahead[0] == pytest.approx(cfg["env"]["lookahead_spacing_m"])
    next_step = tr.step(tel(pos=(41, 0, 0), speed=speed, race_time_ms=300))
    assert next_step.delta_m == pytest.approx(7)


def test_hitch_keeps_the_strict_limit_for_local_backward_snaps():
    cfg = load_config(overrides=["track.max_progress_jump_m=10"])
    tr = ProgressTracker(ReferenceLine.from_positions(straight(), 1.0), cfg)
    tr.reset(tel())
    for i in range(1, 6):
        tr.step(tel(pos=(i * 10, 0, 0), race_time_ms=i * 50))

    backward = tr.step(tel(pos=(35, 0, 0), speed=133, race_time_ms=500))
    assert backward.delta_m == 0 and backward.progress_m == 50
    assert backward.lateral_m == 0


def test_hitch_rejects_a_far_forward_branch_and_preserves_local_lateral():
    tr, _ = make_tracker()
    tr.reset(tel())
    pos = (150, 0, 2)
    local_lateral = tr.ref.project(pos, hint=tr._idx)[2]
    assert local_lateral > 15
    for i in range(1, 10):
        jump = tr.step(tel(pos=pos, speed=133, race_time_ms=i * 1000))
        assert jump.progress_m == jump.delta_m == 0
        assert jump.lateral_m == pytest.approx(local_lateral)

    synced = tr.step(tel(pos=pos, speed=133, race_time_ms=10000))
    assert synced.progress_m == 150 and synced.delta_m == 0
    assert synced.lateral_m == pytest.approx(2)


@pytest.mark.parametrize(("finish_cp", "finish_ms", "flag", "expected"), [
    (1, 9000, 0, 20_370), (2, 20_361, 0, 20_361), (1, 20_361, protocol.FLAG_FINISH_TIME, 20_361),
])
def test_finish_timing_prefers_exact_time_otherwise_interpolates(finish_cp, finish_ms, flag, expected):
    tr, cfg = make_tracker()
    tr.best_lap_ms = 20_400
    tr.reset(tel())
    for x in range(20, 200, 20):
        tr.step(tel(pos=(x, 0, 0), cp_count=1, last_cp_time_ms=9000))
    tr.step(tel(pos=(197, 0, 0), race_time_ms=20_340, cp_count=1, last_cp_time_ms=9000))
    finish = tr.step(tel(pos=(202, 0, 0), race_time_ms=20_390, cp_count=finish_cp,
                         last_cp_time_ms=finish_ms, flags=protocol.FLAG_PLAYER | protocol.FLAG_FINISHED | flag))
    assert finish.lap_ms == expected
    assert finish.terms["personal_best"] == pytest.approx(cfg["reward"]["pb_bonus_per_s"] * (20_400 - expected) / 1000)


@pytest.mark.parametrize(("pos", "vel", "step_hz", "expected"), [
    (198, (100, 0, 0), 20, 20_410),   # velocity along the final segment
    (198, (0, 0, 0), 20, 20_410),     # progress rate when velocity is missing
    (198.5, (20, 0, 0), 20, 20_440),  # cap at 50 ms instead of extrapolating 75 ms
    (199, (20, 0, 0), 60, 20_406),    # stay within the fractional step period
    (194, (20, 0, 0), 20, 20_390),    # still too far from the endpoint
    (198, (-100, 0, 0), 20, 20_390),  # moving away from the endpoint
])
def test_finish_packet_just_short_of_reference_uses_bounded_extrapolation(pos, vel, step_hz, expected):
    cfg = load_config(overrides=[f"env.step_hz={step_hz}"])
    tr = ProgressTracker(ReferenceLine.from_positions(straight(), 1.0), cfg)
    tr.reset(tel())
    for x in range(20, 200, 20):
        tr.step(tel(pos=(x, 0, 0)))
    tr.step(tel(pos=(193, 0, 0), race_time_ms=20_340))
    finish = tr.step(tel(pos=(pos, 0, 0), vel=vel, race_time_ms=20_390,
                         flags=protocol.FLAG_PLAYER | protocol.FLAG_FINISHED))
    assert finish.lap_ms == expected


def test_skipped_and_duplicate_splits_keep_waypoint_indices(tmp_path):
    splits = SplitTracker([1000, 2000, 3000])
    lap = LapBuffer()
    for count, ms, delta in ((1, 900, -100), (1, 910, None), (3, 2950, -50), (2, 1900, None)):
        lap.add(None, tel(cp_count=count, last_cp_time_ms=ms))
        assert splits.update(count, ms) == delta
    assert splits.current == lap.cp_times() == [900, 0, 2950]
    with np.load(lap.save(tmp_path, 2950)) as recorded:
        assert recorded["cp_times"].tolist() == [900, 0, 2950]


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
    cfg = load_config(None, ["reward.pace_per_s=10", "reward.pace_loss_cap=-1"])   # raw pace maths, no cap
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
    from tmai.track.baseline import MAX_PACE_LOSS_S

    xs = np.linspace(0, 100, 101)
    ref = ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0, xs / 10.0)   # 10 m/s
    cfg = load_config(None, ["reward.pace_per_s=10", "reward.time_penalty=0.01", "env.step_hz=20",
                             "reward.pace_loss_cap=-1"])   # raw pace maths, no cap
    tr = ProgressTracker(ref, cfg)
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=0))
    normal = tr.step(tel(pos=(0.5, 0.0, 0.0), race_time_ms=50))        # on the reference pace
    assert normal.terms["time"] == pytest.approx(-0.01) and "pace" not in normal.terms
    stall = tr.step(tel(pos=(1.0, 0.0, 0.0), race_time_ms=1050))       # 1 s on the clock, one step driven
    # Per-step pace loss is bounded so single-step hitches stay within critic Q support,
    # while time_penalty still charges the full elapsed race time.
    assert stall.terms["pace"] == pytest.approx(-10 * MAX_PACE_LOSS_S)
    assert stall.terms["time"] == pytest.approx(-0.01 * 20)
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=1000))              # clock already at 1 s when driving starts
    late = tr.step(tel(pos=(0.5, 0.0, 0.0), race_time_ms=1050))
    assert late.terms["pace"] == pytest.approx(-10 * MAX_PACE_LOSS_S)
    assert late.terms["time"] == pytest.approx(-0.01 * 21)
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=-1500))             # countdown frames (demos) cost nothing extra
    countdown = tr.step(tel(pos=(0.0, 0.0, 0.0), race_time_ms=-1450))
    assert "pace" not in countdown.terms and countdown.terms["time"] == pytest.approx(-0.01)


def test_pace_baseline_follows_the_average_of_recent_laps():
    from tmai.track.baseline import PaceBaseline

    xs = np.linspace(0, 100, 101)
    line = np.stack([xs, 0 * xs, 0 * xs], 1)
    ref = ReferenceLine.from_positions(line, 1.0, xs / 10.0)              # reference lap: 10 s at 10 m/s
    base = PaceBaseline(ref, laps=3, min_laps=2, percentile=50)
    assert not base.from_recent_laps and base.lap_time_s == pytest.approx(10.0)
    assert base.pace(0.0, 0.1, 1.0) == pytest.approx(0.0)                # on reference pace
    base.add_lap(line, xs / 8.0)                                          # 12.5 s lap: one is not enough
    assert base.lap_time_s == pytest.approx(10.0)
    base.add_lap(line, xs / 5.0)                                          # 20 s lap
    assert base.from_recent_laps and base.lap_time_s == pytest.approx(16.25)   # median of two = mean
    # 10 m in 1 s: 1.625 s for the baseline, but gains are capped at 0.1 s a step
    assert base.pace(0.0, 0.1, 1.0) == pytest.approx(0.1)
    assert base.pace(0.0, 0.01, 0.1) == pytest.approx(0.1625 - 0.1)
    base.add_lap(line, xs / 10.0)                                         # laps of 12.5, 20 and 10 s
    assert base.lap_time_s == pytest.approx(12.5)
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


def test_one_bad_lap_does_not_move_the_pace_baseline():
    from tmai.track.baseline import PaceBaseline

    xs = np.linspace(0, 100, 101)
    line = np.stack([xs, 0 * xs, 0 * xs], 1)
    base = PaceBaseline(ReferenceLine.from_positions(line, 1.0, xs / 10.0), laps=20, min_laps=5)
    for _ in range(10):
        base.add_lap(line, xs / 10.0)                                     # 10 s laps
    crash = np.where(xs < 50, xs / 10.0, xs / 10.0 + 6.0)                 # lost 6 s halfway
    base.add_lap(line, crash)
    assert base.lap_time_s == pytest.approx(10.0)
    assert base.pace(0.4, 0.6, 2.0) == pytest.approx(0.0)


def test_pace_baseline_state_round_trips():
    from tmai.track.baseline import PaceBaseline

    xs = np.linspace(0, 100, 101)
    line = np.stack([xs, 0 * xs, 0 * xs], 1)
    ref = ReferenceLine.from_positions(line, 1.0, xs / 10.0)
    base = PaceBaseline(ref, laps=20, min_laps=2)
    for extra in (0.0, 1.0, 2.0):
        base.add_lap(line, xs / 10.0 + extra * xs / 100.0)
    again = PaceBaseline(ref, laps=20, min_laps=2)
    again.load_state(base.state())
    assert again.lap_count == 3 and again.lap_time_s == pytest.approx(base.lap_time_s)
    again.load_state(None)
    assert again.lap_count == 0 and not again.from_recent_laps


def paced_tracker(lap_s=10.0):
    """A 200 m straight whose reference lap takes lap_s seconds (20 m/s)."""
    cfg = load_config()
    xs = np.linspace(0, 200.0, 201)
    ref = ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0, xs / 200.0 * lap_s)
    return ProgressTracker(ref, cfg, SplitTracker()), cfg


def drive(tr, metres_per_step, steps, dt_ms=50):
    tr.reset(tel(pos=(0.0, 0.0, 0.0), speed=0.0, race_time_ms=0))
    total, x, t = 0.0, 0.0, 0
    for _ in range(steps):
        x, t = x + metres_per_step, t + dt_ms
        info = tr.step(tel(pos=(x, 0.0, 0.0), speed=metres_per_step * 20, race_time_ms=t))
        total += info.reward
    return total, info


def test_driving_further_never_scores_worse_than_standing_still():
    # Regression: the pace term charged all time lost to the baseline at
    # pace_per_s, so a slow car lost more the further it drove and standing
    # still (or ending the attempt) looked better than driving on.
    tr, cfg = paced_tracker()
    time_pen = cfg["reward"]["time_penalty"]
    still, _ = drive(tr, 0.0, 100)
    assert still == pytest.approx(-100 * time_pen)          # only the time penalty
    crawl, info = drive(tr, 0.2, 100)                        # 4 m/s, far below the 20 m/s baseline
    share = cfg["reward"]["pace_loss_cap"]
    assert info.terms["progress"] + info.terms.get("pace", 0.0) == pytest.approx((1 - share) * info.terms["progress"])
    assert crawl > still
    further, _ = drive(tr, 0.2, 400)
    assert further > crawl                                    # more progress, more reward


def test_faster_still_pays_more_and_near_baseline_pace_is_unchanged():
    tr, cfg = paced_tracker()
    rewards = [drive(tr, v, 40)[0] for v in (0.2, 0.5, 0.9, 1.0, 1.1)]
    assert rewards == sorted(rewards)                         # faster is always better
    # 2% slower than the baseline: the cap does not bind, the full pace loss applies.
    _, info = drive(tr, 0.98, 10)
    expected = cfg["reward"]["pace_per_s"] * (0.98 / 20.0 - 0.05)
    assert info.terms["pace"] == pytest.approx(expected, rel=1e-3)   # float32 telemetry positions
