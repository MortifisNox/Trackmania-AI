"""Named copies of the trainer's checkpoint (best lap, final, history) with
checksums, and recovery of a damaged latest.pt before the trainer loads it.

The trainer keeps writing runs/<name>/checkpoints/latest.pt itself. This module
never replaces that file except in `prepare_resume`, when it cannot be read."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from tmai.run import atomic

LATEST = "latest.pt"
HISTORY_KEEP = 5


class CheckpointStore:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir) / "checkpoints"
        self.manifest_path = self.dir / "manifest.json"

    @property
    def latest(self) -> Path:
        return self.dir / LATEST

    def manifest(self) -> dict:
        return atomic.read_json(self.manifest_path, {"tags": {}})

    def archive(self, tag: str, src: str | Path | None = None, **info) -> Path:
        """Copy `src` (default latest.pt) to <tag>.pt atomically and record its
        checksum. Tag "history" keeps the last few copies by time."""
        src = Path(src) if src else self.latest
        if tag == "history":
            tag = f"history-{time.strftime('%Y%m%d-%H%M%S')}"
        dst = self.dir / f"{tag}.pt"
        with open(src, "rb") as fin, atomic.atomic_open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
        m = self.manifest()
        m["tags"][tag] = {"file": dst.name, "sha256": atomic.sha256(dst), "saved": time.time(), **info}
        history = sorted(t for t in m["tags"] if t.startswith("history-"))
        for old in history[:-HISTORY_KEEP]:
            (self.dir / m["tags"].pop(old)["file"]).unlink(missing_ok=True)
        atomic.write_json(self.manifest_path, m)
        return dst

    def intact(self, tag: str) -> bool:
        entry = self.manifest()["tags"].get(tag)
        if not entry:
            return False
        path = self.dir / entry["file"]
        return path.exists() and atomic.sha256(path) == entry["sha256"]

    def newest_intact(self) -> Path | None:
        tags = sorted(self.manifest()["tags"].items(), key=lambda kv: kv[1]["saved"], reverse=True)
        for tag, entry in tags:
            if self.intact(tag):
                return self.dir / entry["file"]
        return None


def _loadable(path: Path) -> bool:
    try:
        import torch

        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:   # any failure to unpickle means the file is unusable
        return False


def prepare_resume(run_dir: str | Path, log=print, loadable=_loadable) -> Path | None:
    """Call before the trainer loads its checkpoint. Removes temp files a crash
    left behind and, if latest.pt cannot be read, puts back the newest intact
    archived copy (the damaged file is kept as latest.damaged-<time>.pt).
    Returns the checkpoint the trainer will resume from, or None."""
    store = CheckpointStore(run_dir)
    atomic.clean_temp_files(store.dir)
    for leftover in store.dir.glob("*.tmp") if store.dir.is_dir() else []:
        leftover.unlink(missing_ok=True)   # the trainer's own latest.tmp
    latest = store.latest
    if latest.exists() and loadable(latest):
        return latest
    backup = store.newest_intact()
    if latest.exists():
        damaged = latest.with_name(f"latest.damaged-{time.strftime('%Y%m%d-%H%M%S')}.pt")
        atomic.replace(latest, damaged)
        log(f"checkpoint {latest.name} could not be read; kept as {damaged.name}")
    if backup is None:
        return None
    with open(backup, "rb") as fin, atomic.atomic_open(latest, "wb") as fout:
        shutil.copyfileobj(fin, fout, 1 << 20)
    log(f"resuming from {backup.name}")
    return latest
