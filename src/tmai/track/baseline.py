"""The pace baseline: race time at each point of the lap that the pace reward
compares the car against. It is the median of the agent's last few finished
laps, point by point, so beating its own recent driving pays even on laps that
do not beat the best one, and one lap with a crash or a spin does not drag it
down; until enough laps are in, it is the reference lap itself."""

from __future__ import annotations

from collections import deque

import numpy as np

from tmai.track.reference import ReferenceLine


class PaceBaseline:
    """Times are kept per fraction of the lap (0 at the start, 1 at the finish),
    so transitions stored against an older reference line still line up when a
    faster lap becomes the new line. `laps` = 0 keeps the reference lap."""

    def __init__(self, ref: ReferenceLine, laps: int = 0, min_laps: int = 5):
        self.laps, self.min_laps = laps, max(1, min_laps)
        self._traces: deque = deque(maxlen=max(laps, 1))    # (positions, times) of recent laps
        self._profiles: deque = deque(maxlen=max(laps, 1))  # the same laps as times on the grid
        self.set_ref(ref)

    def set_ref(self, ref: ReferenceLine) -> None:
        """Follow a new reference line; recent laps are measured along it again."""
        self.ref = ref
        self._grid = ref.s / ref.length if ref.length else np.zeros(1)
        self._profiles.clear()
        for pos, times in self._traces:
            self._profiles.append(self._profile(pos, times))
        self._rebuild()

    def add_lap(self, positions, times_s) -> None:
        """A finished lap: car positions and race times (s) at each step."""
        if not self.laps:
            return
        pos = np.asarray(positions, dtype=np.float64)
        times = np.asarray(times_s, dtype=np.float64)
        self._traces.append((pos, times))
        self._profiles.append(self._profile(pos, times))
        self._rebuild()

    @property
    def lap_count(self) -> int:
        return len(self._profiles)

    @property
    def from_recent_laps(self) -> bool:
        return bool(self.laps) and len(self._profiles) >= self.min_laps

    @property
    def times(self) -> np.ndarray | None:
        return self._curve[1]

    @property
    def lap_time_s(self) -> float | None:
        times = self._curve[1]
        return None if times is None else float(times[-1])

    def _profile(self, pos: np.ndarray, times: np.ndarray) -> np.ndarray:
        idx, s = None, np.empty(len(pos))
        for i, p in enumerate(pos):
            idx, s[i], _ = self.ref.project(p, hint=idx)
        u = np.maximum.accumulate(s / self.ref.length)
        keep = np.concatenate([[True], np.diff(u) > 0])   # points where the car got further
        return np.interp(self._grid, u[keep], np.maximum(times[keep], 0.0))

    def _rebuild(self) -> None:
        # (grid, times) swapped in as one object: the learner thread reads it
        # while the driving loop adds laps.
        if self.from_recent_laps:
            # Median, not mean: one lap that lost seconds to a crash would pull a
            # mean up and make the next laps look better than they are.
            self._curve = (self._grid, np.median(np.stack(self._profiles), axis=0))
        elif self.ref.times is not None:
            self._curve = (self._grid, np.asarray(self.ref.times, dtype=np.float64))
        else:
            self._curve = (self._grid, None)

    def pace(self, u0, u1, dt):
        """Seconds gained on the baseline going from lap fraction u0 to u1 in dt
        seconds (arrays or scalars). Gains are capped at 0.2 s a step against
        projection glitches; losses only by the time that really passed."""
        grid, times = self._curve
        if times is None:
            return np.zeros_like(np.asarray(dt, dtype=np.float64))
        base_dt = np.interp(u1, grid, times) - np.interp(u0, grid, times)
        return np.clip(base_dt - dt, -np.maximum(0.2, dt), 0.2)
