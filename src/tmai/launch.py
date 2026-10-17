"""One-click start: read tmai_settings.json, then start the dashboard and training.

The settings file is a short, friendly front for the config system: each
setting becomes a `--set section.key=value` override on top of
configs/default.yaml, so there is still only one set of config rules.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = REPO_ROOT / "tmai_settings.json"
EXAMPLE_FILE = REPO_ROOT / "tmai_settings.example.json"

DEFAULTS = {
    "map_name": "tmrl-test",
    "run_name": "",
    "map_has_checkpoints": True,
    "restart_button": "auto",
    "menu_button": "A",
    "update_code": True,
    "reinstall_plugin": True,
    "wait_for_game": True,
    "record_demo_if_missing": True,
    "demo_laps": 3,
    "start_dashboard": True,
    "open_browser": True,
    "fresh_start": False,
    "train_minutes": None,
    "capture_backend": "window",
    "dashboard_port": 8080,
    "extra_settings": {},
}


class SettingsError(ValueError):
    pass


def load_settings(path: str | Path = SETTINGS_FILE) -> dict:
    """The settings file merged over DEFAULTS. Keys starting with _ are notes."""
    try:
        # utf-8-sig: PowerShell and older Notepad save JSON with a byte-order mark.
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise SettingsError(f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(f"{path} must hold one {{...}} object")
    settings = dict(DEFAULTS)
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key not in DEFAULTS:
            raise SettingsError(f"unknown setting {key!r} in {path}; known ones: {', '.join(DEFAULTS)}")
        settings[key] = value
    if not isinstance(settings["map_name"], str) or not settings["map_name"].strip():
        raise SettingsError("map_name must be the map's name, for example \"Test1\"")
    if not isinstance(settings["extra_settings"], dict):
        raise SettingsError('extra_settings must look like {"train.batch_size": 128}')
    if settings["train_minutes"] is not None and not isinstance(settings["train_minutes"], (int, float)):
        raise SettingsError("train_minutes must be a number of minutes, or null to train until stopped")
    return settings


def restart_button(settings: dict) -> str:
    """"auto" picks Give up (bound to BACK) on maps with checkpoints, B otherwise."""
    button = str(settings["restart_button"]).upper()
    if button == "AUTO":
        return "BACK" if settings["map_has_checkpoints"] else "B"
    return button


def run_name(settings: dict) -> str:
    return (settings["run_name"] or settings["map_name"]).strip()


def to_overrides(settings: dict) -> list[str]:
    """The settings as `section.key=value` config overrides."""
    values = {
        "run.track": settings["map_name"].strip(),
        "run.name": run_name(settings),
        "env.restart_button": restart_button(settings),
        "env.menu_button": str(settings["menu_button"]).upper(),
        "env.capture_backend": settings["capture_backend"],
        "dashboard.port": settings["dashboard_port"],
    }
    values.update(settings["extra_settings"])
    # JSON is valid YAML, so json.dumps keeps strings with spaces as strings.
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def ensure_settings_file(path: Path = SETTINGS_FILE) -> bool:
    """Create the settings file from the example if missing. True if it already existed."""
    if path.exists():
        return True
    shutil.copyfile(EXAMPLE_FILE, path)
    return False


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_dashboard(settings_path: Path, cfg: dict) -> subprocess.Popen | None:
    """Start `tmai dashboard` in its own window; None if one is already running."""
    host, port = cfg["dashboard"]["host"], cfg["dashboard"]["port"]
    if _port_open(host, port):
        print(f"dashboard already running on http://{host}:{port}")
        return None
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)  # Windows: a window of its own
    proc = subprocess.Popen([sys.executable, "-m", "tmai", "--settings", str(settings_path), "dashboard"],
                            cwd=REPO_ROOT, creationflags=flags)
    for _ in range(40):
        if _port_open(host, port) or proc.poll() is not None:
            break
        time.sleep(0.25)
    return proc


def start(cfg: dict, settings: dict, settings_path: Path, train) -> int:
    """Everything after the config is resolved. `train(cfg, minutes, fresh)` runs training."""
    run, env, dash = cfg["run"], cfg["env"], cfg["dashboard"]
    print(f"map {run['track']!r}, run {run['name']!r}, restart button {env['restart_button']}, "
          f"results in {Path(run['output_dir']) / run['name']}")

    if settings["reinstall_plugin"]:
        from tmai.cli import install_plugins

        try:
            install_plugins()
        except RuntimeError as exc:
            print(f"could not install the plugins: {exc}")
            return 1

    from tmai.track.store import TrackStore

    has_reference = TrackStore(run["output_dir"], run["track"]).reference.exists()
    if settings["wait_for_game"]:
        print()
        print("In the game, before you go on:")
        if settings["reinstall_plugin"]:
            print("  - F3 > Developer > (Re)load plugin > TMAI Telemetry, then TMAI Ghost")
        print("  - offline, Openplanet Developer mode, ghosts off")
        print(f"  - start {run['track']} from local play (not the editor), windowed, not minimised")
        if settings["map_has_checkpoints"] and env["restart_button"] == "BACK":
            print("  - Give up bound to View/Back in Settings > Controls")
        input("Press Enter when the map is running... ")

    if not has_reference:
        if not settings["record_demo_if_missing"]:
            print(f"no demo laps for {run['track']!r} yet; set record_demo_if_missing to true, "
                  f"or run tmai --set run.track={json.dumps(run['track'])} record-demo")
            return 1
        from tmai.track.commands import record_demo

        print(f"No demo laps for {run['track']!r} yet: drive {settings['demo_laps']} clean laps yourself now.")
        if record_demo(cfg, int(settings["demo_laps"])) != 0:
            print("no demo laps were recorded, so training cannot start")
            return 1

    if settings["start_dashboard"]:
        start_dashboard(settings_path, cfg)
        if settings["open_browser"]:
            webbrowser.open(f"http://{dash['host']}:{dash['port']}")

    print("Training. Stop with Ctrl+C (once, then wait) or the dashboard's Stop safely button.")
    return train(cfg, settings["train_minutes"], bool(settings["fresh_start"]))
