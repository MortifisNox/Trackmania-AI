import json

import pytest

from tmai import cli, launch
from tmai.config import load_config


def _write(tmp_path, **values):
    path = tmp_path / "tmai_settings.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_example_settings_resolve_to_a_valid_config():
    settings = launch.load_settings(launch.EXAMPLE_FILE)
    cfg = load_config(overrides=launch.to_overrides(settings))
    assert cfg["run"]["track"] == "Test1"
    assert cfg["run"]["name"] == "Test1"
    assert cfg["env"]["restart_button"] == "BACK"


def test_example_documents_every_setting():
    with open(launch.EXAMPLE_FILE, encoding="utf-8") as f:
        example = json.load(f)
    assert set(example["_help"]) - {"how"} == set(launch.DEFAULTS)
    assert set(example) - {"_help"} == set(launch.DEFAULTS)


def test_auto_restart_button_follows_checkpoints(tmp_path):
    with_cps = launch.load_settings(_write(tmp_path, map_name="A", map_has_checkpoints=True))
    without = launch.load_settings(_write(tmp_path, map_name="A", map_has_checkpoints=False))
    explicit = launch.load_settings(_write(tmp_path, map_name="A", restart_button="start"))
    assert launch.restart_button(with_cps) == "BACK"
    assert launch.restart_button(without) == "B"
    assert launch.restart_button(explicit) == "START"


def test_names_with_spaces_and_extra_settings(tmp_path):
    settings = launch.load_settings(_write(tmp_path, map_name="My Map", run_name="map two",
                                           extra_settings={"train.batch_size": 128}, _note="ignored"))
    cfg = load_config(overrides=launch.to_overrides(settings))
    assert cfg["run"]["track"] == "My Map"
    assert cfg["run"]["name"] == "map two"
    assert cfg["train"]["batch_size"] == 128


def test_bad_settings_are_explained(tmp_path):
    with pytest.raises(launch.SettingsError, match="unknown setting"):
        launch.load_settings(_write(tmp_path, map_name="A", mapname="B"))
    with pytest.raises(launch.SettingsError, match="map_name"):
        launch.load_settings(_write(tmp_path, map_name=""))
    bad = tmp_path / "bad.json"
    bad.write_text('{"map_name": "A",}', encoding="utf-8")
    with pytest.raises(launch.SettingsError, match="not valid JSON"):
        launch.load_settings(bad)


def test_byte_order_mark_is_accepted(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"map_name": "A"}), encoding="utf-8-sig")
    assert launch.load_settings(path)["map_name"] == "A"


def test_settings_apply_to_any_command_and_set_wins(tmp_path, capsys):
    path = _write(tmp_path, map_name="Map2", map_has_checkpoints=False)
    assert cli.main(["--settings", str(path), "--set", "run.name=other", "config"]) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["run"]["track"] == "Map2"
    assert cfg["run"]["name"] == "other"
    assert cfg["env"]["restart_button"] == "B"


def test_bad_button_is_a_config_error(tmp_path, capsys):
    path = _write(tmp_path, map_name="A", restart_button="Z")
    assert cli.main(["--settings", str(path), "config"]) == 2
    assert "restart_button" in capsys.readouterr().out


def test_start_runs_demo_dashboard_and_training_in_order(tmp_path, monkeypatch):
    path = _write(tmp_path, map_name="Map2", reinstall_plugin=False, wait_for_game=False,
                  open_browser=False, train_minutes=5, extra_settings={"run.output_dir": str(tmp_path / "runs")})
    settings = launch.load_settings(path)
    cfg = load_config(overrides=launch.to_overrides(settings))
    calls = []
    monkeypatch.setattr("tmai.track.commands.record_demo", lambda c, laps: calls.append(("demo", laps)) or 0)
    monkeypatch.setattr(launch, "start_dashboard", lambda p, c: calls.append(("dashboard", p)))
    result = launch.start(cfg, settings, path, lambda c, minutes, fresh: calls.append(("train", minutes, fresh)) or 0)
    assert result == 0
    assert calls == [("demo", 3), ("dashboard", path), ("train", 5, False)]


def test_start_stops_without_demo_when_recording_is_off(tmp_path):
    path = _write(tmp_path, map_name="Map2", reinstall_plugin=False, wait_for_game=False,
                  record_demo_if_missing=False, extra_settings={"run.output_dir": str(tmp_path / "runs")})
    settings = launch.load_settings(path)
    cfg = load_config(overrides=launch.to_overrides(settings))
    assert launch.start(cfg, settings, path, lambda *a: pytest.fail("trained without a demo")) == 1


def test_start_creates_the_settings_file_first(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "SETTINGS_FILE", tmp_path / "tmai_settings.json")
    assert cli.main(["start"]) == 1
    assert (tmp_path / "tmai_settings.json").read_text() == launch.EXAMPLE_FILE.read_text()
