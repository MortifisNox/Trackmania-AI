"""Command-line entry point: `tmai <command>` or `python -m tmai <command>`."""

from __future__ import annotations

import argparse
import json

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
    return parser


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

    return 2
