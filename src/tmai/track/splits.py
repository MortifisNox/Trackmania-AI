"""Checkpoint splits: this run's times against the best run's."""

from __future__ import annotations

import json
from pathlib import Path


class SplitTracker:
    def __init__(self, best: list[int] | None = None):
        self.best: list[int] = list(best or [])
        self.current: list[int] = []

    def reset(self) -> None:
        self.current = []

    def update(self, cp_count: int, last_cp_time_ms: int) -> int | None:
        """Record a newly crossed checkpoint. Returns the delta to the best
        split at that checkpoint in ms (negative = faster), or None."""
        if cp_count <= len(self.current) or cp_count == 0:
            return None
        self.current.append(last_cp_time_ms)
        i = len(self.current) - 1
        return self.current[i] - self.best[i] if i < len(self.best) else None

    def finish(self) -> bool:
        """Call when the run finishes; keeps it if it is the new best."""
        if not self.current:
            return False
        better = (not self.best or len(self.current) > len(self.best)
                  or (len(self.current) == len(self.best) and self.current[-1] < self.best[-1]))
        if better:
            self.best = list(self.current)
        return better

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"best": self.best}))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SplitTracker":
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text()).get("best", []))
