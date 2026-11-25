"""The pace baseline: race time at each point of the lap that the pace reward
compares the car against. It is a configurable percentile (default 25th, faster
than the median; 50 = median) of the agent's last few finished laps, point by
point, so beating its own faster recent driving pays even on laps that do not
beat the best one, and one lap with a crash or a spin does not drag it down;
until enough laps are in, it is the reference lap itself."""

from __future__ import annotations

from collections import deque

import numpy as np

from tmai.track.reference import ReferenceLine

# Max seconds of pace gain/loss credited in one step: caps projection glitches
# and hitches (at pace_per_s=30, per-step pace stays within [-7.5, +3.0]) while
# leaving normal 20 Hz driving unclipped; time_penalty still charges full dt.
MAX_PACE_GAIN_S = 0.1
MAX_PACE_LOSS_S = 0.25


class PaceBaseline:
    """Times are kept per fraction of the lap (0 at the start, 1 at the finish),
    so transitions stored against an older reference line still line up when a
    faster lap becomes the new line. `laps` = 0 keeps the reference lap."""

    def __init__(self, ref: ReferenceLine, laps: int = 0, min_laps: int = 5,
                 percentile: float = 25.0):
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
        self.laps, self.min_laps = laps, max(1, min_laps)
        self.percentile = float(percentile)
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

    def state(self) -> list:
        """The recent laps, for the checkpoint, so a resumed run keeps its baseline."""
        return [(pos.copy(), times.copy()) for pos, times in self._traces]

    def load_state(self, laps) -> None:
        self._traces.clear()
        for pos, times in laps or []:
            self._traces.append((np.asarray(pos, dtype=np.float64), np.asarray(times, dtype=np.float64)))
        self.set_ref(self.ref)

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
            idx, s[i], _ = self.ref.project(p, hint=idx, start_only=i == 0)
        u = np.maximum.accumulate(s / self.ref.length)
        u[-1] = 1.0
        keep = np.concatenate([[True], np.diff(u) > 0])   # points where the car got further
        profile = np.interp(self._grid, u[keep], np.maximum(times[keep], 0.0))
        profile[-1] = max(0.0, times[-1])
        return profile

    def _rebuild(self) -> None:
        # (grid, times) swapped in as one object: the learner thread reads it
        # while the driving loop adds laps.
        if self.from_recent_laps:
            # Percentile (default 25th, faster than median; 50 = median), not
            # mean: one lap that lost seconds to a crash cannot pull it up.
            self._curve = (self._grid, np.percentile(np.stack(self._profiles), self.percentile, axis=0))
        elif self.ref.times is not None:
            self._curve = (self._grid, np.asarray(self.ref.times, dtype=np.float64))
        else:
            self._curve = (self._grid, None)

    def pace(self, u0, u1, dt, loss_s_per_m=None):
        """Seconds gained on the baseline going from lap fraction u0 to u1 in dt
        seconds (arrays or scalars). Non-positive dt (countdown or clock rewind)
        gives 0; per-step pace is bounded to [-MAX_PACE_LOSS_S, MAX_PACE_GAIN_S]
        so single-step hitches stay within the critic's Q support.

        With `loss_s_per_m`, a step loses at most that many seconds per metre it
        moved forward, and nothing when it did not move forward. Callers pass
        reward.pace_loss_cap * progress_per_m / pace_per_s, so the pace loss takes
        at most that share of the same step's progress reward: driving on always
        beats standing still or ending the attempt, and faster always pays more."""
        dt_arr = np.asarray(dt, dtype=np.float64)
        grid, times = self._curve
        if times is None:
            return np.zeros_like(dt_arr)
        base_dt = np.interp(u1, grid, times) - np.interp(u0, grid, times)
        clipped = np.clip(base_dt - dt_arr, -MAX_PACE_LOSS_S, MAX_PACE_GAIN_S)
        if loss_s_per_m is not None:
            forward_m = np.maximum((np.asarray(u1, dtype=np.float64) - np.asarray(u0, dtype=np.float64))
                                   * self.ref.length, 0.0)
            clipped = np.maximum(clipped, -loss_s_per_m * forward_m)
        return np.where(dt_arr > 0.0, clipped, 0.0)

    @staticmethod
    def loss_cap(reward_cfg: dict):
        """Seconds of pace loss allowed per metre of progress (see pace), or None
        for the old unbounded behaviour (reward.pace_loss_cap < 0)."""
        pace, share = reward_cfg.get("pace_per_s", 0.0), reward_cfg.get("pace_loss_cap", 0.5)
        if not pace or share < 0:
            return None
        return share * reward_cfg["progress_per_m"] / pace
