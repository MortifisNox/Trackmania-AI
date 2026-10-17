"""Command-line entry point: `tmai <command>` or `python -m tmai <command>`."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tmai import __version__
from tmai.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmai", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="YAML file merged over configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override one config value (repeatable)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check this machine is ready to train")
    sub.add_parser("config", help="print the resolved configuration")
    sub.add_parser("install-plugin", help="copy the TMAI Telemetry plugin into Openplanet")
    p = sub.add_parser("get-map", help="download training maps into My Maps")
    p.add_argument("names", nargs="*", default=["tmrl-test"])
    p = sub.add_parser("telemetry", help="print live telemetry from the plugin")
    p.add_argument("--seconds", type=float, default=15.0)
    sub.add_parser("gamepad-test", help="steer, brake and restart through the virtual pad")
    sub.add_parser("capture-test", help="measure screen capture and save what the agent sees")
    p = sub.add_parser("drive-test", help="drive full throttle at 20 Hz and report timing")
    p.add_argument("--episodes", type=int, default=3)
    return parser


def _openplanet_dir() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()) / "OpenplanetNext"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2

    if args.command == "config":
        print(json.dumps(cfg, indent=2))
        return 0

    if args.command == "doctor":
        from tmai.doctor import FAIL, format_report, run_checks

        checks = run_checks(cfg)
        print(format_report(checks))
        return 1 if any(c.status == FAIL for c in checks) else 0

    if args.command == "install-plugin":
        from tmai.game.plugin import install_plugin

        dest = install_plugin(_openplanet_dir())
        print(f"installed to {dest}")
        print("In game: F3 > Developer > (Re)load plugin > TMAI Telemetry, or restart the game.")
        return 0

    if args.command == "get-map":
        from tmai.game.maps import install_maps, my_maps_dir

        for path in install_maps(args.names, my_maps_dir()):
            print(f"saved {path}")
        return 0

    from tmai.game import checks

    if args.command == "telemetry":
        return checks.watch_telemetry(cfg, args.seconds)
    if args.command == "gamepad-test":
        return checks.gamepad_test(cfg)
    if args.command == "capture-test":
        return checks.capture_test(cfg, Path(cfg["run"]["output_dir"]))
    if args.command == "drive-test":
        return checks.drive_test(cfg, args.episodes)
    return 2
