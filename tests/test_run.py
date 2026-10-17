import csv
import json
import os
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tmai.config import load_config
from tmai.rl.replay import ReplayBuffer
from tmai.run import atomic, ghost, snapshot, stop
from tmai.run.checkpoint import CheckpointStore, prepare_resume
from tmai.run.hooks import HookList, TrainerHooks
from tmai.run.report import build_report, stability
from tmai.run.shutdown import AgentLaps, ShutdownHook

# Same columns as tmai.rl.train.EPISODE_FIELDS (that module needs torch, which CI lacks).
EPISODE_FIELDS = ["episode", "wall_time", "steps", "end", "race_time_s", "progress", "reward",
                  "env_steps", "updates", "critic_loss", "actor_loss", "alpha", "q", "entropy",
                  "late_p99_ms"]


def filled_buffer(capacity=500, steps=380, frame=8, episode_len=40):
    """Entry for global step k carries k in its frame, state and action, so a
    torn entry (fields from different steps) is detectable."""
    buf = ReplayBuffer(capacity, frame, 4)
    k = 0
    while k < steps:
        buf.start_episode()
        for _ in range(episode_len - 1):
            add(buf, k)
            k += 1
        add(buf, k, last=True)
        k += 1
    return buf


def add(buf, k, last=False):
    frame = np.full(buf.frames.shape[1:], k % 256, dtype=np.uint8)
    state = np.full(buf.state.shape[1], k, dtype=np.float32)
    if last:
        buf.add(frame, state)
    else:
        buf.add(frame, state, np.array([k, 0, 0], dtype=np.float32), reward=float(k))


# -- atomic writes ----------------------------------------------------------------
def test_atomic_write_keeps_old_file_on_error(tmp_path):
    p = tmp_path / "a.json"
    atomic.write_json(p, {"v": 1})
    with pytest.raises(RuntimeError):
        with atomic.atomic_open(p, "w") as f:
            f.write('{"v": 2')
            raise RuntimeError("crash mid-write")
    assert atomic.read_json(p) == {"v": 1}
    assert list(tmp_path.glob(".*.tmp")) == []


# -- replay snapshots ---------------------------------------------------------------
def test_snapshot_roundtrip(tmp_path):
    buf = filled_buffer(steps=700)   # wraps the ring
    snapshot.save_replay(buf, tmp_path)
    fresh = ReplayBuffer(500, 8, 4)
    manifest = snapshot.load_replay(fresh, tmp_path)
    assert manifest["torn"] == 0
    for key in snapshot.ARRAYS:
        np.testing.assert_array_equal(getattr(fresh, key), getattr(buf, key))
    assert (fresh.pos, fresh.size, fresh.transitions) == (buf.pos, buf.size, buf.transitions)
    assert fresh._episode == buf._episode + 1
    batch = fresh.sample(32, np.random.default_rng(0))
    assert batch["frames"].shape == (32, 4, 8, 8)


def test_snapshot_from_before_pace_inputs_still_loads(tmp_path):
    buf = filled_buffer(steps=100)
    path = snapshot.save_replay(buf, tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    for key in ("pace_u0", "pace_u1", "pace_dt"):                        # what an older snapshot looks like
        (path / f"{key}.npy").unlink()
        del manifest["files"][key]
    (path / "manifest.json").write_text(json.dumps(manifest))
    fresh = ReplayBuffer(500, 8, 4)
    fresh.pace_dt[:] = 1.0
    assert snapshot.load_replay(fresh, tmp_path) is not None
    assert np.isnan(fresh.pace_dt).all()
    np.testing.assert_array_equal(fresh.reward, buf.reward)


def test_snapshot_while_driving_has_no_torn_transitions(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "CHUNK_BYTES", 2000)   # many small chunks, so writes interleave
    buf = filled_buffer(steps=300)
    lock = threading.Lock()
    done = threading.Event()

    def driver():
        k = 300
        while not done.is_set():
            with lock:
                if k % 40 == 0:
                    buf.start_episode()
                add(buf, k, last=k % 40 == 39)
            k += 1
            time.sleep(0.0002)

    t = threading.Thread(target=driver)
    t.start()
    try:
        snapshot.save_replay(buf, tmp_path, lock=lock)
    finally:
        done.set()
        t.join()
    fresh = ReplayBuffer(500, 8, 4)
    snapshot.load_replay(fresh, tmp_path)
    idx = np.arange(fresh.size)
    valid = idx[fresh._valid(idx)]
    assert len(valid) > 100
    for i in valid:
        k = fresh.state[i, 0]
        nxt = (i + 1) % fresh.capacity
        assert fresh.action[i, 0] == k and fresh.reward[i] == k
        assert fresh.frames[i, 0, 0] == int(k) % 256
        assert fresh.state[nxt, 0] == k + 1 and fresh.frames[nxt, 0, 0] == int(k + 1) % 256


def test_damaged_snapshot_falls_back_to_previous(tmp_path):
    buf = filled_buffer()
    first = snapshot.save_replay(buf, tmp_path)
    time.sleep(0.01)
    add(buf, 999)
    second = snapshot.save_replay(buf, tmp_path)
    with open(second / "state.npy", "r+b") as f:
        f.seek(200)
        f.write(b"garbage")
    fresh = ReplayBuffer(500, 8, 4)
    logs = []
    manifest = snapshot.load_replay(fresh, tmp_path, log=logs.append)
    assert manifest["scalars"]["transitions"] == buf.transitions - 1
    assert any("damaged" in m for m in logs)
    assert first.exists()


def test_snapshot_keeps_two(tmp_path):
    buf = filled_buffer()
    for _ in range(4):
        snapshot.save_replay(buf, tmp_path)
        time.sleep(0.01)
    assert len([p for p in tmp_path.glob("snap-*") if p.is_dir()]) == 2


def test_snapshot_of_other_size_is_not_restored(tmp_path):
    snapshot.save_replay(filled_buffer(), tmp_path)
    assert snapshot.load_replay(ReplayBuffer(400, 8, 4), tmp_path, log=lambda m: None) is None


# -- checkpoints ----------------------------------------------------------------
def test_archive_and_recover_damaged_latest(tmp_path):
    store = CheckpointStore(tmp_path)
    store.dir.mkdir(parents=True)
    store.latest.write_bytes(b"good weights")
    store.archive("final")
    assert store.intact("final")
    store.latest.write_bytes(b"\x00\x00")   # torn by a power cut
    path = prepare_resume(tmp_path, log=lambda m: None, loadable=lambda p: p.read_bytes() == b"good weights")
    assert path.read_bytes() == b"good weights"
    assert list(store.dir.glob("latest.damaged-*.pt"))


def test_history_is_pruned(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path)
    store.dir.mkdir(parents=True)
    store.latest.write_bytes(b"w")
    stamps = iter(f"2026010{i}-000000" for i in range(1, 9))
    monkeypatch.setattr("tmai.run.checkpoint.time.strftime", lambda fmt: next(stamps))
    for _ in range(8):
        store.archive("history")
    assert len(list(store.dir.glob("history-*.pt"))) == 5


# -- stop requests ----------------------------------------------------------------
def test_stop_file_from_another_process_sets_event(tmp_path):
    ctl = stop.StopController(tmp_path, log=lambda m: None).install()
    try:
        assert not ctl.poll(force=True)
        stop.request_stop(tmp_path, "dashboard button", source="dashboard")
        assert ctl.poll(force=True)
        assert ctl.reason == "dashboard button"
        assert stop.read_status(tmp_path)["state"] == stop.STOPPING
    finally:
        ctl.finish()
    assert not (tmp_path / "control" / "stop.json").exists()


def test_ctrl_c_requests_stop_and_third_press_forces(tmp_path):
    ctl = stop.StopController(tmp_path, log=lambda m: None).install()
    try:
        os.kill(os.getpid(), signal.SIGINT)
        assert ctl.event.is_set() and ctl.source == "signal"
        os.kill(os.getpid(), signal.SIGINT)
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    finally:
        ctl.uninstall()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_stale_stop_request_is_cleared_on_start(tmp_path):
    stop.request_stop(tmp_path)
    ctl = stop.StopController(tmp_path, log=lambda m: None).install()
    try:
        assert not ctl.poll(force=True)
    finally:
        ctl.finish()


# -- report ---------------------------------------------------------------------
def write_episodes(run_dir, n=300, q_blowup=False):
    rng = np.random.default_rng(0)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        for i in range(1, n + 1):
            finished = i > 50 and i % 3 == 0
            q = 10 + i * 0.01 + (500 if q_blowup and i > n - 50 else 0)
            w.writerow({"episode": i, "wall_time": 1_700_000_000 + i * 30, "steps": 400, "end":
                        "finished" if finished else "stuck", "race_time_s": 40 - i * 0.02 if finished else 12.0,
                        "progress": 1.0 if finished else 0.4, "reward": i * 0.1 + rng.normal(), "env_steps": i * 400,
                        "updates": i * 1600, "critic_loss": 1 / i, "actor_loss": -q, "alpha": 0.1, "q": q,
                        "entropy": -1.0, "late_p99_ms": 2.0})
        f.write("301,1700009000,40,finis")   # row cut short by a crash


def test_report_from_synthetic_run(tmp_path):
    run_dir, track = tmp_path / "runs" / "r1", tmp_path / "runs" / "tracks" / "t"
    write_episodes(run_dir)
    (track / "demos").mkdir(parents=True)
    np.savez(track / "demos" / "a.npz", lap_time_ms=36000, cp_times=np.array([12000, 24000, 36000]))
    np.savez(track / "demos" / "b.npz", lap_time_ms=38000, cp_times=np.array([13000, 25000, 38000]))
    with open(run_dir / "best_laps.jsonl", "w") as f:
        f.write(json.dumps({"episode": 60, "lap_ms": 38800, "splits": [13000, 26000, 38800]}) + "\n")
        f.write(json.dumps({"episode": 297, "lap_ms": 34060, "splits": [11500, 23000, 34060]}) + "\n")
        f.write('{"episode": 29')   # cut short
    path = build_report(run_dir, track, {"track": "t", "replay": {"found": False, "detail": "no plugin"}})
    text = path.read_text()
    assert "34.060 s" in text and "−1.940 s" in text   # best lap and gap to the 36.000 s demo
    s = json.loads((run_dir / "report" / "summary.json").read_text())
    assert s["episodes"] == 300 and s["env_steps"] == 120_000 and s["updates"] == 480_000
    assert s["vs_demo_ms"] == -1940
    assert [r["delta_ms"] for r in s["splits"]] == [-500, -1000, -1940]
    assert s["splits"][1]["sector_delta_ms"] == -500
    assert s["stability"]["q"]["verdict"] == "stable"
    assert 2.4 < s["training_hours"] < 2.6


def test_report_flags_divergence_and_handles_empty_run(tmp_path):
    write_episodes(tmp_path / "bad", q_blowup=True)
    build_report(tmp_path / "bad")
    s = json.loads((tmp_path / "bad" / "report" / "summary.json").read_text())
    assert "divergence" in s["stability"]["q"]["verdict"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No data" in build_report(empty).read_text()


def test_stability_reports_nan():
    v = np.array([1.0, 2.0, np.nan, 1.5])
    assert stability(v)["verdict"].startswith("NaN")


# -- best lap replay ------------------------------------------------------------
def test_preserve_best_replay_picks_fastest(tmp_path):
    root = tmp_path / "Replays" / "TMAI"
    (root / "TMRL Test").mkdir(parents=True)
    for name in ("41250ms-20261017-101010.Replay.Gbx", "39870ms-20261017-111111.Replay.Gbx", "notes.txt"):
        (root / "TMRL Test" / name).write_bytes(b"gbx")
    info = ghost.preserve_best_replay(tmp_path / "run", "tmrl-test", root=root, best_ms=39870)
    assert info["found"] and info["lap_ms"] == 39870 and "note" not in info
    assert (tmp_path / "run" / "report" / "best-39870ms.Replay.Gbx").exists()
    missing = ghost.preserve_best_replay(tmp_path / "run", "other", root=root, openplanet=tmp_path / "op")
    assert not missing["found"] and "but there are for TMRL Test" in missing["detail"]
    empty = ghost.preserve_best_replay(tmp_path / "run", "other", root=tmp_path / "none", openplanet=tmp_path / "op")
    assert not empty["found"] and "not installed" in empty["detail"]


def test_replay_detail_says_what_the_plugin_did(tmp_path):
    op = tmp_path / "OpenplanetNext"
    (op / "Plugins" / "TMAI_Ghost").mkdir(parents=True)
    assert "never ran" in ghost.plugin_state(op)
    (op / "PluginStorage" / "TMAI_Ghost").mkdir(parents=True)
    (op / "PluginStorage" / "TMAI_Ghost" / "status.txt").write_text("2026-10-17 10:00:00 cannot save local replays\n")
    info = ghost.preserve_best_replay(tmp_path / "run", "Test1", root=tmp_path / "Replays", openplanet=op)
    assert info["detail"].endswith("TMAI Ghost plugin said: 2026-10-17 10:00:00 cannot save local replays")


# -- the trainer hook end to end ----------------------------------------------------
class FakeTrainer:
    def __init__(self, run_dir, resumed=False):
        self.stop = threading.Event()
        self.replay_lock = threading.Lock()
        self.online = ReplayBuffer(500, 8, 4)
        self.best_ms, self.episodes, self.warmup = None, 0, 5000
        self.env_steps_before = 1000 if resumed else 0
        self.tracker = SimpleNamespace(splits=SimpleNamespace(current=[]))
        self.ckpt = run_dir / "checkpoints" / "latest.pt"
        self.saves = 0

    def save_checkpoint(self):
        self.saves += 1
        with atomic.atomic_open(self.ckpt) as f:
            f.write(f"weights {self.saves}".encode())
        return self.ckpt


def make_cfg(tmp_path):
    return load_config(overrides=[f"run.output_dir={tmp_path / 'runs'}", "run.name=r1", "run.track=t"])


def test_shutdown_hook_full_run(tmp_path):
    cfg = make_cfg(tmp_path)
    laps = AgentLaps(tmp_path / "op" / "ai_driving.txt")
    laps.enabled = True
    replays = tmp_path / "Replays" / "TMAI"
    (replays / "t").mkdir(parents=True)
    (replays / "t" / "35000ms-x.Replay.Gbx").write_bytes(b"gbx")
    hook = ShutdownHook(cfg, log=lambda m: None, agent_laps=laps, replays_root=replays)
    run_dir = hook.run_dir
    trainer = FakeTrainer(run_dir)
    hook.on_start(trainer)
    assert laps.path.exists()
    write_episodes(run_dir, n=120)

    for k in range(200):
        if k % 40 == 0:
            trainer.online.start_episode()
        add(trainer.online, k)
    trainer.tracker.splits.current = [12000, 24000, 35000]
    tel = SimpleNamespace(last_cp_time_ms=35000, race_time_ms=35010)
    hook.on_step(tel, SimpleNamespace(done="finished"))
    assert laps.path.read_text() == "35000\n"
    trainer.best_ms, trainer.episodes = 35000, 1
    hook.on_episode({"episode": 1, "env_steps": 200, "updates": 0})
    assert hook.checkpoints.intact("best")

    stop.request_stop(run_dir, "dashboard", source="dashboard")
    hook.on_step(None, None)
    hook.controller.poll(force=True)
    assert trainer.stop.is_set()
    hook.on_stop(trainer)

    status = stop.read_status(run_dir)
    assert status["state"] == "stopped" and status["reason"] == "dashboard"
    assert all(s["ok"] for s in status["steps"]), status["steps"]
    assert hook.checkpoints.intact("final")
    assert (run_dir / "report" / "report.html").exists()
    assert (run_dir / "report" / "best-35000ms.Replay.Gbx").exists()
    assert not laps.path.exists()
    best = json.loads((run_dir / "best_laps.jsonl").read_text().splitlines()[0])
    assert best["splits"] == [12000, 24000, 35000]

    # Next session: the trainer resumed from its checkpoint, so the buffer comes back.
    again = FakeTrainer(run_dir, resumed=True)
    hook2 = ShutdownHook(cfg, log=lambda m: None, agent_laps=laps, replays_root=replays)
    hook2.on_start(again)
    try:
        np.testing.assert_array_equal(again.online.state, trainer.online.state)
        assert again.online.transitions == 0 and again.warmup == 0
    finally:
        hook2.controller.finish()


def test_failed_step_does_not_skip_the_rest(tmp_path):
    cfg = make_cfg(tmp_path)
    hook = ShutdownHook(cfg, log=lambda m: None, agent_laps=AgentLaps(tmp_path / "x" / "y" / "z" / "f.txt"),
                        replays_root=tmp_path)
    trainer = FakeTrainer(hook.run_dir)

    def broken():
        raise OSError("disk full")

    trainer.save_checkpoint = broken
    hook.on_start(trainer)
    hook.on_stop(trainer)
    steps = {s["name"]: s for s in stop.read_status(hook.run_dir)["steps"]}
    assert not steps["final checkpoint"]["ok"] and "disk full" in steps["final checkpoint"]["detail"]
    assert steps["best lap replay"]["ok"]
    assert (hook.run_dir / "report" / "report.html").exists()


def test_hooklist_stops_every_hook_even_if_one_fails():
    calls = []

    class Bad(TrainerHooks):
        def on_stop(self, trainer):
            raise RuntimeError("dashboard broke")

    class Good(TrainerHooks):
        def on_stop(self, trainer):
            calls.append("good")

        def should_pause(self):
            return True

    hooks = HookList(Bad(), Good(), None)
    with pytest.raises(RuntimeError):
        hooks.on_stop(None)
    assert calls == ["good"] and hooks.should_pause()
