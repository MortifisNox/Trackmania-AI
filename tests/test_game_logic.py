import zipfile

import pytest

from tmai.game import maps, plugin, protocol
from tmai.game.clock import RealtimeClock
from tmai.game.episode import EpisodeMonitor
from tmai.game.gamepad import Gamepad, clip_action


class FakeTime:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_clock_paces_and_reanchors_after_overrun():
    ft = FakeTime()
    clock = RealtimeClock(20, now=ft.now, sleep=ft.sleep)
    clock.tick()
    assert clock.tick() == 0.0 and ft.t == pytest.approx(0.05)
    ft.t += 0.2  # a step took far too long
    late = clock.tick()
    assert late == pytest.approx(0.15) and clock.overruns == 1
    clock.tick()
    assert ft.t == pytest.approx(0.30)  # next slot is one period after the overrun, no burst


def tel(**kw):
    kw.setdefault("flags", protocol.FLAG_PLAYER | protocol.FLAG_PLAYING)
    return protocol.decode(protocol.encode(**kw))


def test_episode_monitor_reasons():
    m = EpisodeMonitor(timeout_s=60, stuck_timeout_s=3, stuck_speed_kmh=10)
    assert m.check(tel(race_time_ms=500, speed=0.0)) is None  # start grace
    assert m.check(tel(race_time_ms=5000, speed=1.0)) is None
    assert m.check(tel(race_time_ms=7000, speed=1.0)) is None
    assert m.check(tel(race_time_ms=8000, speed=1.0)) == "stuck"
    m.reset()
    assert m.check(tel(race_time_ms=8000, speed=30.0)) is None
    assert m.check(tel(race_time_ms=60000, speed=30.0)) == "timeout"
    assert m.check(tel(flags=protocol.FLAG_PLAYER | protocol.FLAG_FINISHED, race_time_ms=100)) == "finished"


class FakePad:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, k))


def test_gamepad_clips_and_maps_axes():
    pad = FakePad()
    gp = Gamepad("BACK", pad=pad)
    assert gp.act((-2.0, 1.5, -1.0)) == (-1.0, 1.0, 0.0)
    names = [c[0] for c in pad.calls]
    assert names == ["left_joystick_float", "right_trigger_float", "left_trigger_float", "update"]
    assert clip_action((0.3, 0.2, 0.9)) == (0.3, 0.2, 0.9)
    with pytest.raises(ValueError):
        Gamepad("Z", pad=pad)


def test_install_plugin_copies_folder(tmp_path):
    op = tmp_path / "OpenplanetNext"
    op.mkdir()
    dest = plugin.install_plugin(op)
    assert (dest / "info.toml").is_file() and (dest / "main.as").is_file()
    assert plugin.install_plugin(op) == dest  # reinstall replaces cleanly
    with pytest.raises(RuntimeError):
        plugin.install_plugin(tmp_path / "missing")


def test_install_maps_from_archive(tmp_path):
    archive = tmp_path / "resources.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("resources/tmrl-test.Map.Gbx", b"map-bytes")
    docs = tmp_path / "Documents"
    (docs / "Trackmania").mkdir(parents=True)
    target = maps.my_maps_dir(docs)
    [path] = maps.install_maps(["tmrl-test"], target, url=archive.as_uri())
    assert path.read_bytes() == b"map-bytes" and path.parent.name == "My Maps"
    with pytest.raises(ValueError):
        maps.install_maps(["nope"], target, url=archive.as_uri())


def test_model_frame_shrinks_to_square():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from tmai.game.capture import to_model_frame

    out = to_model_frame(np.full((720, 1280), 200, dtype=np.uint8), 64)
    assert out.shape == (64, 64) and out.dtype == np.uint8 and int(out.mean()) == 200


def test_restart_detection():
    from tmai.game.interface import restart_detected

    before = tel(race_time_ms=8000, pos=(100.0, 10.0, 50.0), speed=20.0)
    clock_reset = tel(race_time_ms=-1500, pos=(100.0, 10.0, 50.0), speed=20.0)
    assert restart_detected(before, before, clock_reset, 0.1)
    # race clock stuck at 0 (editor test drive): detect the teleport instead
    b0 = tel(race_time_ms=0, pos=(100.0, 10.0, 50.0), speed=20.0)
    spawn = tel(race_time_ms=0, pos=(10.0, 10.0, 5.0), speed=0.0)
    assert restart_detected(b0, b0, spawn, 0.1)
    # still driving, nothing happened yet
    moving = tel(race_time_ms=0, pos=(101.0, 10.0, 50.0), speed=20.0)
    assert not restart_detected(b0, b0, moving, 0.5)
    # parked at the start already: accept after a short wait
    parked = tel(race_time_ms=0, pos=(10.0, 10.0, 5.0), speed=0.0)
    assert not restart_detected(parked, parked, parked, 1.0)
    assert restart_detected(parked, parked, parked, 2.5)


def test_crop_to_client_removes_title_bar_and_borders():
    np = pytest.importorskip("numpy")
    from tmai.game.capture import crop_to_client

    frame = np.zeros((760, 1296), dtype=np.uint8)
    bounds = (321, 129, 1617, 889)        # whole window on screen
    client = (329, 168, 1609, 888)        # 1280x720 client area inside it
    out = crop_to_client(frame, bounds, client)
    assert out.shape == (720, 1280)
    # frame captured at 2x scale: offsets scale too
    out2 = crop_to_client(np.zeros((1520, 2592), dtype=np.uint8), bounds, client)
    assert out2.shape == (1440, 2560)
    # nonsense rectangles leave the frame untouched
    assert crop_to_client(frame, (0, 0, 0, 0), client).shape == frame.shape


def test_capture_backend_validated():
    from tmai.config import ConfigError, load_config

    assert load_config()["env"]["capture_backend"] == "window"
    with pytest.raises(ConfigError):
        load_config(overrides=["env.capture_backend=gpu"])


def test_restart_that_respawns_at_a_checkpoint_is_refused():
    from tmai.game import interface, protocol
    from tmai.game.interface import CheckpointRespawn, GameInterface

    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    frames = iter([
        protocol.decode(protocol.encode(seq=1, flags=playing, race_time_ms=9000, pos=(100.0, 0.0, 0.0), speed=30.0)),
        protocol.decode(protocol.encode(seq=2, flags=playing, race_time_ms=9050, pos=(50.0, 0.0, 0.0))),
        protocol.decode(protocol.encode(seq=3, flags=playing, race_time_ms=9100, pos=(50.0, 0.0, 0.0))),
    ])

    class Tel:
        def wait(self, timeout, after_seq=None):
            return next(frames)

    class Pad:
        def press_restart(self):
            pass

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad, game.restart_button = Tel(), Pad(), "B"
    with pytest.raises(CheckpointRespawn):
        game.restart()
    assert interface.CheckpointRespawn is CheckpointRespawn


def test_restart_gets_past_the_end_of_race_menu():
    """After a personal best the game's end-of-race menu ignores the restart
    button; restart() taps the menu button (A, Improve) to start the run again."""
    from tmai.game import protocol
    from tmai.game.interface import GameInterface

    playing = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING
    menu = protocol.FLAG_PLAYER | protocol.FLAG_FINISHED
    pressed = []

    class Tel:
        seq = 0

        def wait(self, timeout, after_seq=None):
            self.seq += 1
            if "A" not in pressed:   # the menu stays up until A is pressed
                return protocol.decode(protocol.encode(seq=self.seq, flags=menu, race_time_ms=20550, pos=(1645.0, 0.0, 0.0)))
            self.clock = getattr(self, "clock", -1600) + 100   # countdown, then the run
            return protocol.decode(protocol.encode(seq=self.seq, flags=playing, race_time_ms=self.clock, pos=(0.0, 0.0, 0.0)))

    class Pad:
        def press_restart(self):
            pressed.append("BACK")

        def press(self, name):
            pressed.append(name)

    game = GameInterface.__new__(GameInterface)
    game.telemetry, game.pad, game.restart_button, game.menu_button = Tel(), Pad(), "BACK", "A"
    tel = game.restart(timeout=5.0, menu_after_s=0.0)
    assert pressed == ["BACK", "A"] and tel.race_time_ms == 0
