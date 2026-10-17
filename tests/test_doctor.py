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
