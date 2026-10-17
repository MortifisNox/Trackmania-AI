"""Where each track's demos, reference line and best splits live on disk."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class TrackStore:
    def __init__(self, output_dir: str | Path, name: str):
        if not re.fullmatch(r"[A-Za-z0-9_.\- ]+", name):
            raise ValueError(f"track name {name!r} may only use letters, digits, spaces, - _ .")
        self.name = name
        self.root = Path(output_dir) / "tracks" / name
        self.demos = self.root / "demos"
        self.reference = self.root / "reference.npz"
        self.splits = self.root / "splits.json"

    def demo_files(self) -> list[Path]:
        return sorted(self.demos.glob("*.npz")) if self.demos.is_dir() else []

    def best_demo(self) -> Path:
        files = self.demo_files()
        if not files:
            raise FileNotFoundError(f"no demo laps for track {self.name!r}; run tmai record-demo first")
        def lap_time(p: Path) -> int:
            with np.load(p) as d:
                return int(d["lap_time_ms"])
        return min(files, key=lap_time)
