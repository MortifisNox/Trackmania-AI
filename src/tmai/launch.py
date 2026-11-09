"""One-click start: read tmai_settings.json, then start the dashboard and training.

The settings file is a short, friendly front for the config system: each
setting becomes a `--set section.key=value` override on top of
configs/default.yaml, so there is still only one set of config rules.
"""

from __future__ import annotations

import json
import platform
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
    "observation": "auto",
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
    if str(settings["observation"]).lower() not in ("auto", "image", "state"):
        raise SettingsError('observation must be "auto", "image" or "state"')
    if settings["train_minutes"] is not None and not isinstance(settings["train_minutes"], (int, float)):
        raise SettingsError("train_minutes must be a number of minutes, or null to train until stopped")
    return settings


def restart_button(settings: dict) -> str:
    """"auto" picks Give up (bound to BACK) on maps with checkpoints, B otherwise."""
    button = str(settings["restart_button"]).upper()
    if button == "AUTO":
        return "BACK" if settings["map_has_checkpoints"] else "B"
    return button


# Camera-free driving: 20 line points from 5 m apart growing to ~30 m apart,
# reaching ~290 m up the road, since there is no image to show what is coming.
# Without frames a transition is ~0.35 KB instead of ~4.2 KB, so the replay
# keeps 2M steps (~28 h of driving) in about 0.7 GB.
STATE_PRESET = {"env.lookahead_points": 20, "env.lookahead_spacing_m": 5.0, "env.lookahead_growth": 1.1,
                "replay.capacity": 2_000_000}


def host_system() -> str:
    return platform.system()


def observation(settings: dict, system: str | None = None) -> str:
    """"auto" drives from the camera on Windows and from game state elsewhere,
    since window capture only exists on Windows."""
    mode = str(settings["observation"]).lower()
    if mode == "auto":
        return "image" if (system or host_system()) == "Windows" else "state"
    return mode


def platform_overrides(system: str | None = None) -> list[str]:
    """Overrides for commands run without a settings file: on Linux the camera
    is not available, so they drive from game state like `tmai start` does."""
    if observation({"observation": "auto"}, system) == "image":
        return []
    values = {**STATE_PRESET, "env.observation": "state"}
    return [f"{key}={json.dumps(value)}" for key, value in values.items()]


def run_name(settings: dict, mode: str = "image") -> str:
    """The run_name setting, or the map name. State-mode runs get a -state
    suffix, since a camera run of the same map cannot continue in state mode."""
    if settings["run_name"].strip():
        return settings["run_name"].strip()
    name = settings["map_name"].strip()
    return f"{name}-state" if mode == "state" else name


def to_overrides(settings: dict, system: str | None = None) -> list[str]:
    """The settings as `section.key=value` config overrides."""
    mode = observation(settings, system)
    values = {
        **(STATE_PRESET if mode == "state" else {}),
        "env.observation": mode,
        "run.track": settings["map_name"].strip(),
        "run.name": run_name(settings, mode),
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
    out = None
    if not flags:
        # No new console elsewhere: keep the dashboard's messages out of the training output.
        log = Path(cfg["run"]["output_dir"]) / "dashboard.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        out = open(log, "ab")
        print(f"dashboard messages go to {log}")
    proc = subprocess.Popen([sys.executable, "-m", "tmai", "--settings", str(settings_path), "dashboard"],
                            cwd=REPO_ROOT, creationflags=flags, stdout=out, stderr=out)
    for _ in range(40):
        if _port_open(host, port) or proc.poll() is not None:
            break
        time.sleep(0.25)
    return proc


def start(cfg: dict, settings: dict, settings_path: Path, train) -> int:
    """Everything after the config is resolved. `train(cfg, minutes, fresh)` runs training."""
    run, env, dash = cfg["run"], cfg["env"], cfg["dashboard"]
    print(f"map {run['track']!r}, run {run['name']!r}, drives from {env['observation']}, "
          f"restart button {env['restart_button']}, "
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
