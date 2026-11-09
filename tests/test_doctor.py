from tmai import doctor
from tmai.config import load_config


def test_python_version_rules():
    assert doctor.check_python((3, 9, 0)).status == doctor.FAIL
    assert doctor.check_python((3, 12, 1)).status == doctor.WARN
    assert doctor.check_python((3, 11, 9)).status == doctor.OK


def test_openplanet_missing_fails(tmp_path):
    checks = doctor.check_openplanet(tmp_path)
    assert checks[0].status == doctor.FAIL


def test_openplanet_with_tmrl_plugin_warns(tmp_path):
    plugins = tmp_path / "OpenplanetNext" / "Plugins"
    plugins.mkdir(parents=True)
    (plugins / "TMRL_GrabData.op").write_bytes(b"")
    op, plugin = doctor.check_openplanet(tmp_path)
    assert op.status == doctor.OK and plugin.status == doctor.WARN


def test_ram_check_warns_when_replay_too_big():
    cfg = load_config(overrides=["replay.capacity=2000000"])
    assert doctor.check_ram(cfg, total_ram=8 * 1024**3).status == doctor.WARN
    cfg = load_config()
    assert doctor.check_ram(cfg, total_ram=32 * 1024**3).status == doctor.OK


def test_report_formats_every_check():
    checks = [doctor.Check("A", doctor.OK, "x"), doctor.Check("Long name", doctor.FAIL, "y")]
    report = doctor.format_report(checks)
    assert "[FAIL] Long name" in report and report.count("\n") == 1


def test_linux_needs_state_mode():
    assert doctor.check_observation(load_config(), "Linux").status == doctor.FAIL
    assert doctor.check_observation(load_config(), "Windows").status == doctor.OK
    state = load_config(overrides=["env.observation=state"])
    assert doctor.check_observation(state, "Linux").status == doctor.OK
    assert doctor.check_os("Linux").status == doctor.OK


def test_uinput_check(tmp_path):
    assert doctor.check_uinput(tmp_path / "uinput").status == doctor.FAIL
    dev = tmp_path / "uinput"
    dev.write_bytes(b"")
    assert doctor.check_uinput(dev).status == doctor.OK


def test_proton_launch_option(tmp_path):
    assert doctor.check_proton_launch_option(tmp_path).status == doctor.WARN
    vdf = tmp_path / ".steam" / "steam" / "userdata" / "123" / "config" / "localconfig.vdf"
    vdf.parent.mkdir(parents=True)
    vdf.write_text('"apps" { "2225070" { "LaunchOptions" "WINEDLLOVERRIDES=\\"dinput8=n,b\\" %command%" } }')
    assert doctor.check_proton_launch_option(tmp_path).status == doctor.OK


def test_state_mode_replay_is_small():
    cfg = load_config(overrides=["env.observation=state", "replay.capacity=2000000"])
    assert doctor.check_ram(cfg, total_ram=16 * 1024**3).status == doctor.OK
