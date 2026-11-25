"""Tests for the state-mode LiDAR, pace baseline, replay sizing and countdown launch upgrades."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_geometry import quad, write_export
from test_rl import LENGTH, PLAYING, tel
from test_run import filled_buffer
from tmai import launch
from tmai.config import DEFAULT_CONFIG, load_config
from tmai.game import protocol
from tmai.game.interface import FULL_GAS_CONTROLS, GameInterface
from tmai.rl.env import FULL_GAS, NO_INPUT, TMEnv
from tmai.rl.replay import ReplayBuffer
from tmai.run import snapshot
from tmai.run.shutdown import AgentLaps, ShutdownHook
from tmai.track.baseline import MAX_PACE_GAIN_S, PaceBaseline
from tmai.track.lidar import export_sha256
from tmai.track.progress import ProgressTracker
from tmai.track.reference import ReferenceLine
from tmai.track.store import TrackStore


def _write_settings(tmp_path, **values):
    path = tmp_path / "tmai_settings.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _make_state_cfg(tmp_path, *extra):
    return load_config(overrides=[
        "env.observation=state",
        f"run.output_dir={(tmp_path / 'runs').as_posix()}",
        *extra,
    ])


# -- 1. LiDAR on by default, preset, run_name, geometry gate -------------------

def test_default_yaml_has_upgrade_keys_and_values():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert raw["env"]["lidar"] is True
    assert raw["env"]["launch_hold_gas"] is True
    assert raw["reward"]["pace_per_s"] == 30.0
    assert raw["reward"]["pace_baseline_laps"] == 30
    assert raw["reward"]["pace_baseline_percentile"] == 25
    assert raw["train"]["target_entropy"] == -3.0
    assert raw["train"]["target_entropy_end"] == -3.0
    assert raw["train"]["n_step"] == 5
    assert raw["train"]["updates_per_step"] == 8
    assert raw["train"]["optimizer"] == "adamw"
    assert raw["train"]["weight_decay"] == 1.0e-4
    assert raw["train"]["critic_loss"] == "huber"
    assert raw["train"]["huber_delta"] == 1.0
    assert raw["train"]["reset_every_updates"] == 400000
    assert raw["train"]["reset_shrink"] == 0.8
    assert raw["model"]["actor_layer_norm"] is True
    assert raw["model"]["critic_head"] == "hl_gauss"
    assert raw["model"]["q_bins"] == 141
    assert raw["model"]["q_min"] == -150.0
    assert raw["model"]["q_max"] == 200.0
    assert raw["model"]["hl_sigma_ratio"] == 0.75


def test_state_preset_and_run_name_for_lidar(tmp_path):
    assert launch.STATE_PRESET["env.lidar"] is True
    assert launch.STATE_PRESET["replay.capacity"] == 350_000

    default_state = launch.load_settings(_write_settings(tmp_path, map_name="Fall 2026 - 01", observation="state"))
    assert default_state["lidar"] is True
    assert launch.run_name(default_state, "state") == "Fall 2026 - 01-state-lidar"
    cfg = load_config(overrides=launch.to_overrides(default_state, system="Linux"))
    assert cfg["env"]["lidar"] is True
    assert cfg["replay"]["capacity"] == 350_000
    assert cfg["run"]["name"] == "Fall 2026 - 01-state-lidar"

    no_lidar = launch.load_settings(_write_settings(tmp_path, map_name="Fall 2026 - 01", observation="state",
                                                    lidar=False))
    assert launch.run_name(no_lidar, "state") == "Fall 2026 - 01-state"
    cfg_off = load_config(overrides=launch.to_overrides(no_lidar, system="Linux"))
    assert cfg_off["env"]["lidar"] is False
    assert cfg_off["run"]["name"] == "Fall 2026 - 01-state"

    custom = launch.load_settings(_write_settings(tmp_path, map_name="Fall 2026 - 01", run_name="my-run",
                                                  observation="state"))
    assert launch.run_name(custom, "state") == "my-run"

    with pytest.raises(launch.SettingsError, match="lidar"):
        launch.load_settings(_write_settings(tmp_path, map_name="A", lidar="yes"))


def test_launcher_geometry_gate_blocks_missing_and_invalid_and_allows_valid(tmp_path, capsys):
    track = "Fall 2026 - 01"
    runs_dir = tmp_path / "runs"
    store = TrackStore(runs_dir, track)
    xs = np.linspace(0, 60.0, 30)
    ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0).save(store.reference)

    settings_path = _write_settings(
        tmp_path, map_name=track, observation="state", reinstall_plugin=False,
        wait_for_game=False, start_dashboard=False, open_browser=False,
        extra_settings={"run.output_dir": str(runs_dir)},
    )
    settings = launch.load_settings(settings_path)
    cfg = load_config(overrides=launch.to_overrides(settings, system="Linux"))
    trained = []

    # 1. Missing geometry.json -> stopped before training with clear message.
    rc = launch.start(cfg, settings, settings_path, lambda *a: trained.append(a) or 0)
    out = capsys.readouterr().out
    assert rc == 1 and not trained
    assert "LiDAR is on but the map geometry isn't ready" in out
    assert f'tmai --set "run.track={track}" geometry-check' in out
    assert '"lidar": false' in out

    # 2. Broken geometry.json (missing export_sha256 -> load_lidar raises ValueError) -> stopped.
    export = write_export(tmp_path / "map.tmgeo", [(*quad(0, 32, -0.4, -16, 16), 16)],
                          [("road0", (0, 8, 0), 0, (1, 1, 1), [(0, [])])])
    store.geometry.write_text(json.dumps({"export": str(export), "placement": {"yoff": -64.0}}))
    rc = launch.start(cfg, settings, settings_path, lambda *a: trained.append(a) or 0)
    out = capsys.readouterr().out
    assert rc == 1 and not trained
    assert "LiDAR is on but the map geometry isn't ready" in out
    assert "export_sha256" in out

    # 3. Valid geometry.json -> proceeds to train.
    store.geometry.write_text(json.dumps({
        "export": str(export), "export_sha256": export_sha256(export), "placement": {"yoff": -64.0},
    }))
    rc = launch.start(cfg, settings, settings_path, lambda *a: trained.append(a) or 0)
    assert rc == 0 and len(trained) == 1

    # 4. lidar=false skips geometry gate even when geometry.json is missing.
    store.geometry.unlink()
    settings_off_path = _write_settings(
        tmp_path, map_name=track, observation="state", lidar=False, reinstall_plugin=False,
        wait_for_game=False, start_dashboard=False, open_browser=False,
        extra_settings={"run.output_dir": str(runs_dir)},
    )
    settings_off = launch.load_settings(settings_off_path)
    cfg_off = load_config(overrides=launch.to_overrides(settings_off, system="Linux"))
    rc = launch.start(cfg_off, settings_off, settings_off_path, lambda *a: trained.append(a) or 0)
    assert rc == 0 and len(trained) == 2


# -- 2. Percentile pace baseline and gain cap ----------------------------------

def test_pace_baseline_percentile_50_matches_median_and_25_is_faster():
    xs = np.linspace(0, 100, 101)
    line = np.stack([xs, 0 * xs, 0 * xs], 1)
    ref = ReferenceLine.from_positions(line, 1.0, xs / 10.0)

    lap_durations = [10.0, 11.0, 12.0, 14.0, 20.0]
    p50 = PaceBaseline(ref, laps=10, min_laps=5, percentile=50)
    p25 = PaceBaseline(ref, laps=10, min_laps=5)   # default percentile=25
    assert p25.percentile == 25.0

    for dur in lap_durations:
        times = xs / 100.0 * dur
        p50.add_lap(line, times)
        p25.add_lap(line, times)

    assert p50.lap_time_s == pytest.approx(float(np.median(lap_durations)))
    assert p25.lap_time_s == pytest.approx(float(np.percentile(lap_durations, 25)))
    assert p25.lap_time_s < p50.lap_time_s
    # Point-by-point along the whole lap, p25 <= p50.
    assert np.all(p25.times <= p50.times + 1e-12)
    # Single-step gain clip keeps pace_per_s=30 bounded (<= 3.0 per step).
    assert p25.pace(0.0, 0.5, 0.05) == pytest.approx(MAX_PACE_GAIN_S)
    assert 30.0 * p25.pace(0.0, 0.5, 0.05) == pytest.approx(3.0)

    with pytest.raises(ValueError, match="percentile"):
        PaceBaseline(ref, laps=10, min_laps=5, percentile=120)


# -- 3. Snapshot capacity mismatch ---------------------------------------------

def test_larger_snapshot_does_not_crash_smaller_buffer(tmp_path):
    big = filled_buffer(capacity=500, steps=450, frame=0)
    snapshot.save_replay(big, tmp_path / "replay")

    small = ReplayBuffer(200, 0, 4)
    logs = []
    assert snapshot.load_replay(small, tmp_path / "replay", log=logs.append) is None
    assert small.size == 0 and small.transitions == 0
    assert any("different buffer size" in msg for msg in logs)

    # ShutdownHook._restore_replay also copes cleanly on resume without skipping warmup.
    cfg = _make_state_cfg(tmp_path, "run.name=r1", "run.track=t", "replay.capacity=200")
    run_dir = tmp_path / "runs" / "r1"
    snapshot.save_replay(big, run_dir / "replay")
    hook_logs = []
    hook = ShutdownHook(cfg, log=hook_logs.append, agent_laps=AgentLaps(tmp_path / "no" / "ai.txt"))
    trainer = SimpleNamespace(
        stop=None, online=ReplayBuffer(200, 0, 4), warmup=1000, best_ms=None, env_steps_before=500,
    )
    hook._restore_replay(trainer)
    assert trainer.online.size == 0 and trainer.warmup == 1000
    assert any("different buffer size" in msg for msg in hook_logs)


# -- 4. Countdown full-gas hold and TMEnv.reset prev_action --------------------

def _make_fake_restart_game(frames, launch_hold_gas: bool):
    it = iter(frames)
    events = []

    class Tel:
        def wait(self, timeout, after_seq=None, max_age=None):
            return next(it)

    class Pad:
        def __init__(self):
            self.held = (0.0, 0.0, 0.0)

        def press_restart(self):
            self.held = (0.0, 0.0, 0.0)
            events.append(("press_restart",))

        def press(self, name):
            self.held = (0.0, 0.0, 0.0)
            events.append(("press", name))

        def act(self, controls):
            self.held = controls
            events.append(("act", controls))
            return controls

        def release(self):
            self.held = (0.0, 0.0, 0.0)
            events.append(("release",))

    game = GameInterface.__new__(GameInterface)
    game.telemetry = Tel()
    game.pad = Pad()
    game.restart_button = "BACK"
    game.menu_button = "A"
    game.max_age = 0.1
    game.launch_hold_gas = launch_hold_gas
    return game, events


def test_restart_holds_gas_through_countdown_and_keeps_it_at_go():
    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    frames = [
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=8000, pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=3, flags=playing, race_time_ms=-1000, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=4, flags=playing, race_time_ms=-500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=5, flags=playing, race_time_ms=25, pos=(0.0, 0.0, 0.0))),
    ]
    game, events = _make_fake_restart_game(frames, launch_hold_gas=True)
    result = game.restart()
    assert result.seq == 5
    # Restart pressed first, then full gas held during countdown and still held at return.
    assert events == [("press_restart",), ("act", FULL_GAS_CONTROLS)]
    assert game.pad.held == FULL_GAS_CONTROLS


def test_restart_releases_when_launch_hold_gas_is_off():
    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    frames = [
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=8000, pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=3, flags=playing, race_time_ms=25, pos=(0.0, 0.0, 0.0))),
    ]
    game, events = _make_fake_restart_game(frames, launch_hold_gas=False)
    result = game.restart()
    assert result.seq == 3
    assert events == [("press_restart",), ("release",)]
    assert game.pad.held == (0.0, 0.0, 0.0)


def test_restart_releases_held_gas_if_countdown_times_out():
    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    packets = iter([
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=8000, pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
    ])
    events = []

    class Tel:
        def wait(self, timeout, after_seq=None, max_age=None):
            try:
                return next(packets)
            except StopIteration:
                raise TimeoutError("plugin stalled")

    class Pad:
        def press_restart(self):
            events.append("press_restart")

        def act(self, controls):
            events.append(("act", controls))

        def release(self):
            events.append("release")

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad = Tel(), Pad()
    game.restart_button, game.menu_button, game.max_age = "BACK", "A", 0.1
    game.launch_hold_gas = True
    with pytest.raises(TimeoutError):
        game.restart()
    assert events == ["press_restart", ("act", FULL_GAS_CONTROLS), "release"]


@pytest.mark.parametrize(("hold_gas", "expected_prev"), [
    (True, FULL_GAS),
    (False, NO_INPUT),
])
def test_env_reset_sets_prev_action_and_does_not_release_after_restart(tmp_path, hold_gas, expected_prev):
    cfg = _make_state_cfg(tmp_path, "env.lidar=false", f"env.launch_hold_gas={'true' if hold_gas else 'false'}")
    xs = np.linspace(0, LENGTH, 30)
    ref = ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0)
    tracker = ProgressTracker(ref, cfg)
    order = []

    class Pad:
        def release(self):
            order.append("release")

    class Game:
        pad = Pad()

        def restart(self):
            order.append("restart")

        def observe(self):
            order.append("observe")
            return None, tel(pos=(0.0, 0.0, 0.0), speed=0.0, race_time_ms=25, flags=PLAYING)

    env = TMEnv(cfg, Game(), tracker)
    obs = env.reset()
    assert order == ["release", "restart", "observe"]
    np.testing.assert_array_equal(env._prev_action, expected_prev)
    # Previous action occupies indices 15..17 in state_vector's head.
    np.testing.assert_array_equal(obs.state[15:18], expected_prev)


# -- 5. Regression tests for review findings 1-8 -------------------------------

@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, RuntimeError])
def test_restart_releases_gas_on_any_unsuccessful_exit(exc_cls):
    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    seq_packets = iter([
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=8000, pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
    ])
    events = []

    class Tel:
        def wait(self, timeout, after_seq=None, max_age=None):
            try:
                return next(seq_packets)
            except StopIteration:
                raise exc_cls("injected failure after full gas")

    class Pad:
        def __init__(self):
            self.held = (0.0, 0.0, 0.0)

        def press_restart(self):
            events.append("press_restart")

        def act(self, controls):
            self.held = controls
            events.append(("act", controls))

        def release(self):
            self.held = (0.0, 0.0, 0.0)
            events.append("release")

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad = Tel(), Pad()
    game.restart_button, game.menu_button, game.max_age = "BACK", "A", 0.1
    game.launch_hold_gas = True
    with pytest.raises(exc_cls):
        game.restart()
    assert events == ["press_restart", ("act", FULL_GAS_CONTROLS), "release"]
    assert game.pad.held == (0.0, 0.0, 0.0)


def test_restart_should_abort_during_countdown_and_menu_waits(monkeypatch):
    from tmai.game import interface
    from tmai.game.interface import RestartAborted

    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    # (a) Abort requested during countdown after full gas is held.
    frames = [
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=8000, pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=3, flags=playing, race_time_ms=-1000, pos=(0.0, 0.0, 0.0))),
    ]
    game, events = _make_fake_restart_game(frames, launch_hold_gas=True)
    aborted = {"flag": False}

    def abort_after_gas():
        if ("act", FULL_GAS_CONTROLS) in events:
            aborted["flag"] = True
        return aborted["flag"]

    with pytest.raises(RestartAborted):
        game.restart(should_abort=abort_after_gas)
    assert events == [("press_restart",), ("act", FULL_GAS_CONTROLS), ("release",)]
    assert game.pad.held == (0.0, 0.0, 0.0)

    # (b) Abort requested while waiting on a silent menu (no packets arriving).
    now = 0.0
    monkeypatch.setattr(interface.time, "monotonic", lambda: now)
    polls = 0
    menu_events = []

    class SilentTel:
        def latest(self):
            return protocol.decode(protocol.encode(seq=1, flags=protocol.FLAG_PLAYER | protocol.FLAG_FINISHED,
                                                   race_time_ms=20500))

        def wait(self, timeout, after_seq=None, max_age=None):
            nonlocal now
            now += timeout
            raise TimeoutError("silent menu")

    class Pad:
        def press_restart(self):
            menu_events.append("press_restart")

        def press(self, name):
            menu_events.append(("press", name))

        def release(self):
            menu_events.append("release")

    def abort_on_third_poll():
        nonlocal polls
        polls += 1
        return polls >= 4

    g2 = GameInterface.__new__(GameInterface)
    g2.telemetry, g2.pad = SilentTel(), Pad()
    g2.restart_button, g2.menu_button, g2.max_age = "BACK", "A", 0.1
    g2.launch_hold_gas = True
    with pytest.raises(RestartAborted):
        g2.restart(timeout=15.0, menu_after_s=2.5, should_abort=abort_on_third_poll)
    assert now < 1.0  # aborted promptly instead of waiting 2.5s+
    assert menu_events[-1] == "release"


def test_restart_recovers_when_finish_menu_is_already_silent_with_stale_packet(monkeypatch):
    import time as real_time

    from tmai.game import interface
    from tmai.game.telemetry import TelemetryClient

    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    finished = protocol.FLAG_PLAYER | protocol.FLAG_FINISHED
    now = 100.0
    monkeypatch.setattr(interface.time, "monotonic", lambda: now)

    # Real TelemetryClient seeded with a stale packet (30s old > max_age 0.1s).
    client = TelemetryClient(port=1)
    client._latest = protocol.decode(protocol.encode(seq=10, flags=finished, race_time_ms=20550, pos=(500.0, 0.0, 0.0)))
    client._received_at = real_time.monotonic() - 30.0
    orig_wait = client.wait
    events = []
    fresh_stream = iter([
        protocol.decode(protocol.encode(seq=11, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=12, flags=playing, race_time_ms=25, pos=(0.0, 0.0, 0.0))),
    ])

    def fake_wait(timeout, after_seq=None, max_age=None):
        nonlocal now
        assert max_age == 0.1
        if ("press", "A") not in events:
            now += timeout
            raise TimeoutError("no fresh telemetry")
        return next(fresh_stream)

    client.wait = fake_wait
    del orig_wait

    class Pad:
        def __init__(self):
            self.held = (0.0, 0.0, 0.0)

        def press_restart(self):
            events.append(("press_restart",))

        def press(self, name):
            events.append(("press", name))

        def act(self, controls):
            self.held = controls
            events.append(("act", controls))

        def release(self):
            self.held = (0.0, 0.0, 0.0)
            events.append(("release",))

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad = client, Pad()
    game.restart_button, game.menu_button, game.max_age = "BACK", "A", 0.1
    game.launch_hold_gas = True
    try:
        res = game.restart(timeout=15.0, menu_after_s=1.0)
    finally:
        client.close()
    assert res.seq == 12
    # Pressed restart at once, then menu button A, and only applied gas after fresh countdown arrived.
    assert events == [("press_restart",), ("press", "A"), ("act", FULL_GAS_CONTROLS)]
    assert game.pad.held == FULL_GAS_CONTROLS


def test_checkpoint_respawn_detected_with_nonzero_speed_and_cp_count(monkeypatch):
    from tmai.game import interface
    from tmai.game.interface import CheckpointRespawn

    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    now = 0.0
    monkeypatch.setattr(interface.time, "monotonic", lambda: now)

    # Car jumps back to checkpoint with cp_count=1 and sampled speed 7.2 km/h (2.0 m/s).
    frames = iter([
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=9000, cp_count=1,
                                        pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=9050, cp_count=1,
                                        pos=(50.0, 0.0, 0.0), speed=2.0)),
        protocol.decode(protocol.encode(seq=3, flags=playing, race_time_ms=10600, cp_count=1,
                                        pos=(52.0, 0.0, 0.0), speed=2.0)),
    ])
    events = []

    class Tel:
        def wait(self, timeout, after_seq=None, max_age=None):
            nonlocal now
            t = next(frames)
            if t.seq == 3:
                now += 1.6  # past settle_s=1.5
            else:
                now += 0.05
            return t

    class Pad:
        def press_restart(self):
            events.append("press_restart")

        def release(self):
            events.append("release")

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad = Tel(), Pad()
    game.restart_button, game.menu_button, game.max_age = "B", "A", 0.1
    game.launch_hold_gas = True
    with pytest.raises(CheckpointRespawn):
        game.restart(settle_s=1.5)
    assert events == ["press_restart", "release"]


def test_menu_packet_after_countdown_clears_restart_latch_releases_gas_and_resumes_menu(monkeypatch):
    from tmai.game import interface

    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    menu = protocol.FLAG_PLAYER | protocol.FLAG_FINISHED
    now = 0.0
    monkeypatch.setattr(interface.time, "monotonic", lambda: now)
    events = []

    # Countdown detected -> gas held -> fresh non-playing menu packet arrives ->
    # must clear restart latch, release gas, press menu button A, then recover on fresh countdown+GO.
    pre_menu = iter([
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=8000, pos=(100.0, 0.0, 0.0), speed=20.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=3, flags=menu, race_time_ms=8000, pos=(100.0, 0.0, 0.0))),
    ])
    post_menu = iter([
        protocol.decode(protocol.encode(seq=4, flags=playing, race_time_ms=-1500, pos=(0.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=5, flags=playing, race_time_ms=25, pos=(0.0, 0.0, 0.0))),
    ])

    class Tel:
        def wait(self, timeout, after_seq=None, max_age=None):
            nonlocal now
            if ("press", "A") not in events:
                try:
                    now += 1.0
                    return next(pre_menu)
                except StopIteration:
                    now += timeout
                    raise TimeoutError("menu silent")
            now += 0.1
            return next(post_menu)

    class Pad:
        def __init__(self):
            self.held = (0.0, 0.0, 0.0)

        def press_restart(self):
            events.append(("press_restart",))

        def press(self, name):
            events.append(("press", name))

        def act(self, controls):
            self.held = controls
            events.append(("act", controls))

        def release(self):
            self.held = (0.0, 0.0, 0.0)
            events.append(("release",))

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad = Tel(), Pad()
    game.restart_button, game.menu_button, game.max_age = "BACK", "A", 0.1
    game.launch_hold_gas = True
    res = game.restart(timeout=15.0, menu_after_s=0.5)
    assert res.seq == 5
    assert events == [
        ("press_restart",),
        ("act", FULL_GAS_CONTROLS),
        ("release",),
        ("press", "A"),
        ("act", FULL_GAS_CONTROLS),
    ]
    assert game.pad.held == FULL_GAS_CONTROLS


def test_load_lidar_rejects_zero_collision_triangles(tmp_path):
    from tmai.track.lidar import load_lidar

    # Export with a shape that has 0 triangles (all NotCollidable or empty).
    empty_export = write_export(tmp_path / "empty.tmgeo", [], [])
    fit_path = tmp_path / "geometry.json"
    fit_path.write_text(json.dumps({
        "export": str(empty_export),
        "export_sha256": export_sha256(empty_export),
        "uid": "uid1",
        "placement": {"yoff": -64.0},
    }))
    with pytest.raises(ValueError, match="zero collision triangles"):
        load_lidar(fit_path)
    store = SimpleNamespace(geometry=fit_path)
    err = launch.check_lidar_geometry(store, "EmptyMap")
    assert err is not None and "zero collision triangles" in err


def test_env_lidar_map_uid_mismatch_and_old_plugin_warning(tmp_path):
    from tmai.rl.env import MapMismatchError
    from tmai.track.lidar import load_lidar

    cfg = _make_state_cfg(tmp_path, "env.lidar=true", "env.step_hz=20")
    xs = np.linspace(0, LENGTH, 30)
    ref = ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0)
    export = write_export(tmp_path / "map.tmgeo", [(*quad(0, 32, -0.4, -16, 16), 16)],
                          [("road0", (0, 8, 0), 0, (1, 1, 1), [(0, [])])])
    fit_path = tmp_path / "geometry.json"
    fit_path.write_text(json.dumps({
        "export": str(export),
        "export_sha256": export_sha256(export),
        "uid": "expected_uid",
        "placement": {"yoff": -64.0},
    }))
    lidar = load_lidar(fit_path)
    assert lidar.uid == "expected_uid"

    class FakePad:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    class FakeGame:
        def __init__(self, uid):
            self.pad = FakePad()
            self.uid = uid

        def restart(self):
            pass

        def observe(self):
            return None, tel(pos=(5.0, 0.0, 0.0), race_time_ms=25, flags=PLAYING, map_uid=self.uid)

    # 1. Wrong map UID raises MapMismatchError and releases pad.
    logs = []
    game_wrong = FakeGame("other_map_uid")
    env = TMEnv(cfg, game_wrong, ProgressTracker(ref, cfg), lidar=lidar, log=logs.append)
    with pytest.raises(MapMismatchError, match="wrong map loaded"):
        env.reset()
    assert game_wrong.pad.releases >= 2  # initial release + release on exception

    # 2. Missing map UID (old plugin) logs a one-time warning and continues.
    logs.clear()
    game_old = FakeGame("")
    env_old = TMEnv(cfg, game_old, ProgressTracker(ref, cfg), lidar=lidar, log=logs.append)
    env_old.reset()
    env_old.reset()
    assert len(logs) == 1 and "no map UID" in logs[0]

    # 3. Matching map UID succeeds silently.
    logs.clear()
    game_ok = FakeGame("expected_uid")
    env_ok = TMEnv(cfg, game_ok, ProgressTracker(ref, cfg), lidar=lidar, log=logs.append)
    env_ok.reset()
    assert logs == []


def test_env_step_ends_episode_with_bad_pose_after_sustained_invalid_lidar(tmp_path):
    from tmai.rl.env import TERMINAL
    from tmai.track.lidar import load_lidar

    assert "bad_pose" not in TERMINAL
    cfg = _make_state_cfg(tmp_path, "env.lidar=true", "env.step_hz=20")
    xs = np.linspace(0, LENGTH, 30)
    ref = ReferenceLine.from_positions(np.stack([xs, 0 * xs, 0 * xs], 1), 1.0)
    export = write_export(tmp_path / "map.tmgeo", [(*quad(0, 32, -0.4, -16, 16), 16)],
                          [("road0", (0, 8, 0), 0, (1, 1, 1), [(0, [])])])
    fit_path = tmp_path / "geometry.json"
    fit_path.write_text(json.dumps({
        "export": str(export),
        "export_sha256": export_sha256(export),
        "uid": "uid1",
        "placement": {"yoff": -64.0},
    }))
    lidar = load_lidar(fit_path)

    class FakeClock:
        def reset(self):
            pass

        def tick(self):
            return 0.0

    class FakePad:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    class FakeGame:
        def __init__(self, packets):
            self.pad = FakePad()
            self._it = iter(packets)

        def restart(self):
            pass

        def act(self, controls):
            return controls

        def observe(self):
            return None, next(self._it)

    good = lambda ms: tel(pos=(5.0, 0.0, 0.0), dir=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
                          speed=20.0, race_time_ms=ms, flags=PLAYING, map_uid="uid1")
    bad = lambda ms: tel(pos=(5.0, 0.0, 0.0), dir=(0.0, 0.0, 0.0), up=(0.0, 0.0, 0.0),
                         speed=20.0, race_time_ms=ms, flags=PLAYING, map_uid="uid1")

    # 5 bad steps (0.25s) interrupted by 1 good step -> streak resets; then 21 bad steps (1.05s) -> bad_pose.
    packets = [good(0)]
    t_ms = 50
    for _ in range(5):
        packets.append(bad(t_ms))
        t_ms += 50
    packets.append(good(t_ms))
    t_ms += 50
    for _ in range(25):
        packets.append(bad(t_ms))
        t_ms += 50

    game = FakeGame(packets)
    env = TMEnv(cfg, game, ProgressTracker(ref, cfg), clock=FakeClock(), lidar=lidar)
    env.reset()
    releases_after_reset = game.pad.releases
    reason = None
    steps_after_good = 0
    for idx in range(30):
        _, _, reason, _ = env.step(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        if idx >= 6:
            steps_after_good += 1
        if reason is not None:
            break
    assert reason == "bad_pose"
    assert steps_after_good == 21  # > 1.0s at 20 Hz
    assert game.pad.releases == releases_after_reset + 1


def test_auto_run_name_follows_final_merged_config_and_explicit_name_wins(tmp_path, capsys):
    from tmai import cli

    settings_path = _write_settings(tmp_path, map_name="Fall 2026 - 01", observation="state", lidar=True)
    # 1. Default state+lidar -> -state-lidar
    assert cli.main(["--settings", str(settings_path), "config"]) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["run"]["name"] == "Fall 2026 - 01-state-lidar"

    # 2. --set env.lidar=false updates automatic run name to -state (no collision!)
    assert cli.main(["--settings", str(settings_path), "--set", "env.lidar=false", "config"]) == 0
    cfg_no_lidar = json.loads(capsys.readouterr().out)
    assert cfg_no_lidar["env"]["lidar"] is False
    assert cfg_no_lidar["run"]["name"] == "Fall 2026 - 01-state"

    # 3. --set env.observation=image drops the -state-lidar suffix
    assert cli.main([
        "--settings", str(settings_path),
        "--set", "env.observation=image", "--set", "env.lookahead_points=10",
        "config",
    ]) == 0
    cfg_img = json.loads(capsys.readouterr().out)
    assert cfg_img["run"]["name"] == "Fall 2026 - 01"

    # 4. Explicit --set run.name wins even with --set env.lidar=false
    assert cli.main([
        "--settings", str(settings_path),
        "--set", "env.lidar=false", "--set", "run.name=custom",
        "config",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["name"] == "custom"

    # 5. Explicit run_name in settings wins even with --set env.lidar=false
    named_path = _write_settings(tmp_path, map_name="Fall 2026 - 01", run_name="pinned",
                                 observation="state", lidar=True)
    assert cli.main(["--settings", str(named_path), "--set", "env.lidar=false", "config"]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["name"] == "pinned"


def test_config_validate_types_and_sac_alignment():
    from tmai.config import ConfigError

    # Non-int counts rejected (floats and booleans)
    for bad in (
        "model.q_bins=10.5", "model.q_bins=true",
        "train.n_step=1.5", "train.updates_per_step=2.5",
        "train.batch_size=16.0", "reward.pace_baseline_laps=10.5",
        "train.reset_every_updates=100.5",
    ):
        with pytest.raises(ConfigError):
            load_config(overrides=[bad])

    # Non-boolean switches rejected (quoted "false" must NOT become True)
    for bad in (
        'env.lidar="false"', 'env.launch_hold_gas="false"',
        "env.lidar=1", "env.launch_hold_gas=0",
        'model.actor_layer_norm="false"',
    ):
        with pytest.raises(ConfigError):
            load_config(overrides=[bad])

    # Non-finite numbers rejected
    for bad in (
        "model.q_min=.nan", "model.q_max=.inf", "model.hl_sigma_ratio=.inf",
        "reward.pace_per_s=.nan", "train.gamma=.nan",
    ):
        with pytest.raises(ConfigError):
            load_config(overrides=[bad])

    # Aligned with sac.py: reset_shrink in [0, 1] inclusive of 0
    assert load_config(overrides=["train.reset_shrink=0.0"])["train"]["reset_shrink"] == 0.0
    assert load_config(overrides=["train.reset_shrink=1.0"])["train"]["reset_shrink"] == 1.0
    with pytest.raises(ConfigError):
        load_config(overrides=["train.reset_shrink=-0.1"])

    # Aligned with sac.py: huber_delta validated only when critic_head is scalar (and huber)
    assert load_config(overrides=["model.critic_head=hl_gauss", "train.huber_delta=0"])["model"]["critic_head"] == "hl_gauss"
    assert load_config(overrides=["model.critic_head=scalar", "train.critic_loss=mse", "train.huber_delta=0"])["train"]["critic_loss"] == "mse"
    with pytest.raises(ConfigError):
        load_config(overrides=["model.critic_head=scalar", "train.critic_loss=huber", "train.huber_delta=0"])


def test_pace_rewind_is_zero_and_hitch_loss_is_bounded():
    from tmai.track.baseline import MAX_PACE_LOSS_S

    xs = np.linspace(0, 100, 101)
    line = np.stack([xs, 0 * xs, 0 * xs], 1)
    ref = ReferenceLine.from_positions(line, 1.0, xs / 10.0)   # 10 m/s
    cfg = load_config(overrides=["reward.pace_per_s=30.0", "reward.time_penalty=0.01", "env.step_hz=20"])
    tr = ProgressTracker(ref, cfg)

    # (a) Race-clock rewind (dt < 0) gives 0 pace (must NOT pay +3).
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=0))
    tr.step(tel(pos=(10.0, 0.0, 0.0), race_time_ms=1000))
    rewind = tr.step(tel(pos=(10.5, 0.0, 0.0), race_time_ms=500))  # dt = -0.5s
    assert rewind.terms.get("pace", 0.0) == 0.0
    assert tr.baseline.pace(0.1, 0.2, -0.5) == 0.0
    assert tr.baseline.pace(0.1, 0.2, 0.0) == 0.0

    # (b) 1.82s hitch gives bounded pace (-0.25 * 30 = -7.5, not -54.6),
    # while time_penalty still charges real lost time.
    tr.reset(tel(pos=(0.0, 0.0, 0.0), race_time_ms=0))
    hitch = tr.step(tel(pos=(0.0, 0.0, 0.0), race_time_ms=1820))
    assert hitch.terms["pace"] == pytest.approx(-30.0 * MAX_PACE_LOSS_S)
    assert hitch.terms["pace"] == pytest.approx(-7.5)
    assert hitch.terms["time"] == pytest.approx(-0.01 * 1.82 * 20)

    # Vectorized replay rescoring obeys the same bounds and dt <= 0 -> 0.
    vals = tr.baseline.pace(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.1, 0.1]),
        np.array([0.05, 1.82, -0.5, 0.0]),
    )
    np.testing.assert_allclose(vals, [MAX_PACE_GAIN_S, -MAX_PACE_LOSS_S, 0.0, 0.0])
