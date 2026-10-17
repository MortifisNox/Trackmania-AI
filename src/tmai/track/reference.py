"""Reference racing line: a polyline resampled at fixed spacing, with fast
projection of the car onto it and lookahead points in the car's frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resample(points: np.ndarray, spacing: float, values: np.ndarray | None = None):
    """Evenly spaced points along the polyline through `points` (N, 3). With
    `values` (N,), also returns them interpolated at the new points."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])  # drop standstill duplicates
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("reference needs at least two distinct points")
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    targets = np.arange(0.0, s[-1], spacing)
    if s[-1] - targets[-1] > 1e-6:
        targets = np.append(targets, s[-1])
    out = np.stack([np.interp(targets, s, pts[:, k]) for k in range(3)], axis=1)
    if values is None:
        return out
    return out, np.interp(targets, s, np.asarray(values, dtype=np.float64)[keep])


class ReferenceLine:
    """`times` (optional, seconds) is when the lap the line came from passed
    each point; it lets the reward compare the car's pace with that lap."""

    def __init__(self, points: np.ndarray, spacing: float, times: np.ndarray | None = None):
        self.points = np.asarray(points, dtype=np.float64)
        self.spacing = spacing
        self.times = None if times is None else np.maximum.accumulate(np.asarray(times, dtype=np.float64))
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    @classmethod
    def from_positions(cls, positions: np.ndarray, spacing: float = 1.0,
                       times: np.ndarray | None = None) -> "ReferenceLine":
        if times is None:
            return cls(resample(positions, spacing), spacing)
        points, point_times = resample(positions, spacing, times)
        return cls(points, spacing, point_times)

    def time_at(self, s: float) -> float:
        return float(np.interp(s, self.s, self.times))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        extra = {} if self.times is None else {"times": self.times}
        np.savez(tmp, points=self.points, spacing=self.spacing, **extra)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ReferenceLine":
        with np.load(path) as data:
            times = data["times"] if "times" in data.files else None
            return cls(data["points"], float(data["spacing"]), times)

    def project(self, pos, hint: int | None = None, window: int = 80) -> tuple[int, float, float]:
        """(segment index, distance along the line in m, distance from the line in m).

        Searches a window around `hint` (the previous index) so a track that
        crosses over itself does not snap to the wrong part; falls back to a
        full search when the car is far from the windowed match (respawn)."""
        p = np.asarray(pos, dtype=np.float64)
        n = len(self.points) - 1
        if hint is None:
            lo, hi = 0, n
        else:
            lo, hi = max(0, hint - window // 4), min(n, hint + window)
        idx, s, dist = self._project_range(p, lo, hi)
        if hint is not None and dist > 15.0:
            full = self._project_range(p, 0, n)
            if full[2] < dist:
                idx, s, dist = full
        return idx, s, dist

    def _project_range(self, p: np.ndarray, lo: int, hi: int) -> tuple[int, float, float]:
        a = self.points[lo:hi]
        b = self.points[lo + 1:hi + 1]
        ab = b - a
        denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        closest = a + ab * t[:, None]
        d = np.linalg.norm(closest - p, axis=1)
        k = int(np.argmin(d))
        idx = lo + k
        s = self.s[idx] + t[k] * (self.s[idx + 1] - self.s[idx])
        return idx, float(s), float(d[k])

    def point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        return np.stack([np.interp(s, self.s, self.points[:, k]) for k in range(3)])

    def lookahead(self, pos, forward, up, s: float, count: int, spacing: float) -> np.ndarray:
        """`count` reference points ahead of `s`, as (forward, left, up) metres in
        the car's frame, flattened to (count * 3,)."""
        f = np.asarray(forward, dtype=np.float64)
        u = np.asarray(up, dtype=np.float64)
        f /= max(np.linalg.norm(f), 1e-9)
        u /= max(np.linalg.norm(u), 1e-9)
        left = np.cross(u, f)  # sign convention only; the policy learns either way
        p = np.asarray(pos, dtype=np.float64)
        out = np.empty((count, 3))
        for i in range(count):
            rel = self.point_at(s + (i + 1) * spacing) - p
            out[i] = (rel @ f, rel @ left, rel @ u)
        return out.reshape(-1)
