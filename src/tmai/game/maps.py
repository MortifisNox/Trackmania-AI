"""Fetches training maps into the game's My Maps folder."""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

# TMRL's MIT-licensed resources bundle; contains its test and training maps.
TMRL_RESOURCES = "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip"
MAPS = {
    "tmrl-test": "resources/tmrl-test.Map.Gbx",
    "tmrl-train": "resources/tmrl-train.Map.Gbx",
}


def documents_dir() -> Path:
    """The real Documents folder (often redirected to OneDrive)."""
    try:
        from win32com.shell import shell, shellcon

        return Path(shell.SHGetFolderPath(0, shellcon.CSIDL_PERSONAL, None, 0))
    except ImportError:
        return Path(os.path.expanduser("~")) / "Documents"


def my_maps_dir(documents: Path | None = None) -> Path:
    documents = documents or documents_dir()
    for name in ("Trackmania", "Trackmania2020"):
        if (documents / name).is_dir():
            return documents / name / "Maps" / "My Maps"
    raise RuntimeError(f"no Trackmania folder in {documents}; start the game once first")


def install_maps(names: list[str], target: Path, url: str = TMRL_RESOURCES) -> list[Path]:
    unknown = [n for n in names if n not in MAPS]
    if unknown:
        raise ValueError(f"unknown map(s) {unknown}; available: {sorted(MAPS)}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        dest = target / Path(MAPS[name]).name
        dest.write_bytes(archive.read(MAPS[name]))
        written.append(dest)
    return written
