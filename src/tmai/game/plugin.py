"""Installs the TMAI Telemetry plugin into Openplanet's plugin folder."""

from __future__ import annotations

import shutil
from pathlib import Path

PLUGIN_NAME = "TMAI_Telemetry"
PLUGIN_SRC = Path(__file__).resolve().parents[3] / "openplanet" / PLUGIN_NAME


def install_plugin(openplanet_dir: Path, src: Path = PLUGIN_SRC) -> Path:
    plugins = openplanet_dir / "Plugins"
    if not openplanet_dir.is_dir():
        raise RuntimeError(f"{openplanet_dir} not found; install Openplanet and start the game once")
    dest = plugins / PLUGIN_NAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest
