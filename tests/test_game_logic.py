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
