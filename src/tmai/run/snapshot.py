"""Replay buffer snapshots that survive crashes and can be taken while
training keeps driving.

A snapshot is a folder of .npy files plus a manifest, written under a temp
name and renamed into place; `latest.json` then points at it. The arrays are
copied a few thousand rows at a time, each chunk under the trainer's replay
lock, so the driving loop is only held up for a few milliseconds at a time and
every row is copied whole. Rows the driver wrote while the copy ran may be the
old or the new version; they are marked unusable, so a restored buffer only
samples transitions that were complete before the snapshot started.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path

import numpy as np

from tmai.run import atomic

ARRAYS = ("frames", "state", "action", "reward", "terminal", "has_next", "episode")
SCALARS = ("pos", "size", "_episode", "transitions")
FORMAT = 1
CHUNK_BYTES = 4 << 20    # rows copied per lock hold (about 1 ms of memcpy)
KEEP = 2                 # snapshots kept on disk; older ones are deleted


class SnapshotError(RuntimeError):
    pass


def _ring_range(start: int, end: int, capacity: int, wrapped_all: bool) -> np.ndarray:
    """Indices start, start+1, ... up to (not including) end on a ring."""
    if wrapped_all:
        return np.arange(capacity)
    n = (end - start) % capacity
    return (start + np.arange(n)) % capacity


def save_replay(buffer, folder: str | Path, lock=None, step: int | None = None) -> Path:
    """Write `buffer` (a tmai.rl.replay.ReplayBuffer) as a new snapshot in
    `folder` and return its path. `lock` is the trainer's replay lock, or None
    when nothing else is touching the buffer (for example at shutdown)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    guard = lock if lock is not None else contextlib.nullcontext()
    with guard:
        start_pos, start_total = int(buffer.pos), int(buffer.transitions)
    name = f"snap-{int(time.time() * 1000)}"
    tmp = folder / f".{name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        outs = {}
        for key in ARRAYS:
            src = getattr(buffer, key)
            outs[key] = np.lib.format.open_memmap(tmp / f"{key}.npy", mode="w+", dtype=src.dtype, shape=src.shape)
        capacity = len(buffer.frames)
        row_bytes = sum(getattr(buffer, k)[0:1].nbytes for k in ARRAYS)
        rows = max(1, CHUNK_BYTES // max(1, row_bytes))
        # Reused staging rows: under the lock only a RAM copy into memory that
        # is already mapped; the slow write to the file happens unlocked.
        stage = {key: np.empty((rows,) + getattr(buffer, key).shape[1:], getattr(buffer, key).dtype)
                 for key in ARRAYS}
        for arr in stage.values():
            arr.fill(0)   # fault the pages in now, not while holding the lock
        for lo in range(0, capacity, rows):
            hi = min(capacity, lo + rows)
            with guard:
                for key in ARRAYS:
                    np.copyto(stage[key][:hi - lo], getattr(buffer, key)[lo:hi])
            for key in ARRAYS:
                outs[key][lo:hi] = stage[key][:hi - lo]
        with guard:
            scalars = {k: int(getattr(buffer, k)) for k in SCALARS}
        # Entries written during the copy: drop them from the snapshot.
        added = scalars["transitions"] - start_total
        torn = _ring_range(start_pos, scalars["pos"], capacity, wrapped_all=added >= capacity)
        if len(torn):
            outs["has_next"][torn] = False
            outs["episode"][torn] = -1
            # The entry just before the torn range pointed into it.
            outs["has_next"][(torn[0] - 1) % capacity] = False
        for key in ARRAYS:
            outs[key].flush()
        outs.clear()   # Windows cannot rename a folder with open memory maps
        manifest = {"format": FORMAT, "created": time.time(), "step": step, "torn": int(len(torn)),
                    "capacity": capacity, "frame_shape": list(buffer.frames.shape[1:]),
                    "state_dim": int(buffer.state.shape[1]), "scalars": scalars,
                    "files": {k: atomic.sha256(tmp / f"{k}.npy") for k in ARRAYS}}
        atomic.write_json(tmp / "manifest.json", manifest)
        final = folder / name
        atomic.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    atomic.write_json(folder / "latest.json", {"snapshot": name})
    _prune(folder, keep_name=name)
    return final


def _prune(folder: Path, keep_name: str) -> None:
    snaps = sorted(p for p in folder.glob("snap-*") if p.is_dir())
    for old in snaps[:-KEEP]:
        if old.name != keep_name:
            shutil.rmtree(old, ignore_errors=True)
    for partial in folder.glob(".snap-*.partial"):
        shutil.rmtree(partial, ignore_errors=True)


def _candidates(folder: Path) -> list[Path]:
    """Snapshots newest first, the one latest.json names at the front."""
    snaps = sorted((p for p in folder.glob("snap-*") if p.is_dir()), reverse=True)
    latest = (atomic.read_json(folder / "latest.json") or {}).get("snapshot")
    if latest:
        snaps.sort(key=lambda p: p.name != latest)
    return snaps


def verify(path: Path) -> dict:
    manifest = atomic.read_json(path / "manifest.json")
    if not manifest or manifest.get("format") != FORMAT:
        raise SnapshotError(f"{path.name}: missing or unknown manifest")
    for key, digest in manifest["files"].items():
        f = path / f"{key}.npy"
        if not f.exists() or atomic.sha256(f) != digest:
            raise SnapshotError(f"{path.name}: {key}.npy is damaged")
    return manifest


def load_replay(buffer, folder: str | Path, log=print) -> dict | None:
    """Fill `buffer` from the newest intact snapshot in `folder`. Damaged
    snapshots are skipped. Returns the manifest, or None if nothing fit."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    for path in _candidates(folder):
        try:
            manifest = verify(path)
        except SnapshotError as exc:
            log(f"replay snapshot skipped: {exc}")
            continue
        if (manifest["capacity"] != len(buffer.frames)
                or tuple(manifest["frame_shape"]) != buffer.frames.shape[1:]
                or manifest["state_dim"] != buffer.state.shape[1]):
            log(f"replay snapshot {path.name} has a different buffer size or shape; not restored")
            return None
        for key in ARRAYS:
            src = np.load(path / f"{key}.npy", mmap_mode="r")
            getattr(buffer, key)[:] = src
            del src
        for key, value in manifest["scalars"].items():
            setattr(buffer, key, value)
        # New episodes must not continue an episode id from the snapshot.
        buffer._episode += 1
        return manifest
    return None
