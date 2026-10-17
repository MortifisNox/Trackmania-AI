"""Crash-safe file writes: write a temp file next to the target, flush it to
disk, then rename it over the target. A crash or power cut at any point leaves
either the old file or the new one, never a half-written mix."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

# Windows refuses to replace a file that another program (antivirus, indexer,
# a TensorBoard reader) has open for a moment; retry instead of failing.
REPLACE_RETRIES = 20
REPLACE_WAIT_S = 0.05


def fsync_dir(path: Path) -> None:
    """Persist a rename on POSIX. Windows cannot open directories; NTFS
    journals the rename itself."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace(src: Path, dst: Path) -> None:
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_S)


@contextmanager
def atomic_open(path: str | Path, mode: str = "wb") -> Iterator[IO]:
    """`with atomic_open(p) as f: f.write(...)`. The target only changes if the
    block finishes without an exception."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoding = None if "b" in mode else "utf-8"
    try:
        with open(tmp, mode, encoding=encoding) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def write_bytes(path: str | Path, data: bytes) -> None:
    with atomic_open(path, "wb") as f:
        f.write(data)


def write_json(path: str | Path, obj) -> None:
    with atomic_open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str | Path, default=None):
    """Read JSON written by write_json; a missing or unreadable file gives `default`."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def torch_save(obj, path: str | Path) -> None:
    """torch.save with the same guarantee. Use it for checkpoints."""
    import torch

    with atomic_open(path, "wb") as f:
        torch.save(obj, f)


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def clean_temp_files(folder: str | Path) -> int:
    """Remove temp files a crash left behind. Returns how many were removed."""
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    removed = 0
    for p in folder.glob(".*.tmp"):
        p.unlink(missing_ok=True)
        removed += 1
    return removed
