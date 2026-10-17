import zipfile

import pytest

from tmai.game import maps, plugin, protocol
from tmai.game.clock import RealtimeClock
from tmai.game.episode import EpisodeMonitor
from tmai.game.gamepad import Gamepad, clip_action
from tmai.game.input_guard import Foreground, InputGuard, blocked_reason


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


def test_input_guard_spots_windows_ui():
    alt_tab = Foreground("MultitaskingViewFrame", "explorer.exe", set())
    game = Foreground("ManiaPlanet", "Trackmania.exe", set())
    shell = Foreground("ConsoleWindowClass", "WindowsTerminal.exe", set())
    assert blocked_reason("shell", alt_tab) == "Alt+Tab / Task View is open"
    assert blocked_reason("shell", Foreground("Windows.UI.Core.CoreWindow", "StartMenuExperienceHost.exe", set()))
    assert blocked_reason("shell", Foreground("x", "GameBar.exe", set())) == "Xbox Game Bar is open"
    assert blocked_reason("shell", game) is None
    assert blocked_reason("shell", shell) is None          # game may train in the background
    assert blocked_reason("shell", Foreground("ManiaPlanet", "Trackmania.exe", {0x12})) == "Alt is held"
    assert blocked_reason("shell", None) == "no window is in front"
    assert blocked_reason("game", shell) == "the game window is not in front"
    assert blocked_reason("game", game) is None
    assert blocked_reason("off", alt_tab) is None
    with pytest.raises(ValueError):
        InputGuard("sometimes")


def test_gamepad_goes_neutral_while_windows_ui_is_in_front():
    front = {"fg": Foreground("ManiaPlanet", "Trackmania.exe", set())}
    pad = FakePad()
    gp = Gamepad("B", pad=pad, guard=InputGuard("shell", probe=lambda: front["fg"]), poll_s=60)
    assert gp.act((0.5, 1.0, 0.0)) == (0.5, 1.0, 0.0)
    assert gp.take_interruption() is None
    front["fg"] = Foreground("MultitaskingViewFrame", "explorer.exe", set())
    pad.calls.clear()
    assert gp.act((-1.0, 1.0, 0.0)) == (0.0, 0.0, 0.0)
    assert gp.act((1.0, 1.0, 0.0)) == (0.0, 0.0, 0.0)
    assert [c[0] for c in pad.calls] == ["reset", "update"]   # neutral once, no steering
    with pytest.raises(TimeoutError, match="Alt\\+Tab"):
        gp.press_restart(wait_s=0.0)
    assert not any(c[0] == "press_button" for c in pad.calls)  # B is Back in menus
    front["fg"] = Foreground("ManiaPlanet", "Trackmania.exe", set())
    assert gp.take_interruption() == "Alt+Tab / Task View is open"
    assert gp.take_interruption() is None
    assert gp.act((0.2, 0.0, 0.0)) == (0.2, 0.0, 0.0)
    gp.press_restart(hold_s=0.0)
    assert ("press_button", {"button": "B"}) in pad.calls
    gp.close()


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
