"""Turns telemetry into progress along the reference line, reward, and track-based
episode ends (wrong way, off track)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tmai.game.protocol import Telemetry
from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker


@dataclass
class StepInfo:
    reward: float
    progress_m: float          # distance along the reference line
    progress: float            # 0..1 of the reference length
    lateral_m: float           # distance from the reference line
    delta_m: float             # progress made this step
    lookahead: np.ndarray      # reference points ahead, in the car's frame
    done: str | None = None    # finished / wrong_way / off_track
    split_delta_ms: int | None = None
    crashed: bool = False
    terms: dict = field(default_factory=dict)


class ProgressTracker:
    def __init__(self, ref: ReferenceLine, cfg: dict, splits: SplitTracker | None = None):
        self.ref = ref
        self.track = cfg["track"]
        self.rw = cfg["reward"]
        self.env = cfg["env"]
        self.splits = splits or SplitTracker()
        self._idx: int | None = None
        self._s = 0.0
        self._max_s = 0.0
        self._prev_speed = 0.0
        self._done: str | None = None

    def reset(self, tel: Telemetry) -> None:
        self._idx, self._s, _ = self.ref.project(tel.pos)
        self._max_s = self._s
        self._prev_speed = tel.speed_kmh
        self._done = None
        self.splits.reset()

    @property
    def ended(self) -> str | None:
        return self._done

    def step(self, tel: Telemetry) -> StepInfo:
        if self._done:
            # The attempt already ended; the game keeps sending frames until the
            # restart (the finish screen, the car rolling on), which earn nothing.
            return StepInfo(reward=0.0, progress_m=self._s,
                            progress=self._s / self.ref.length if self.ref.length else 0.0,
                            lateral_m=0.0, delta_m=0.0,
                            lookahead=np.zeros(3 * self.env["lookahead_points"]),
                            done=self._done)
        idx, s, lateral = self.ref.project(tel.pos, hint=self._idx)
        delta = s - self._s
        if abs(delta) > self.track["max_progress_jump_m"]:
            delta = 0.0  # teleport (respawn) or a shortcut across the line: no credit
        self._idx, self._s = idx, s
        self._max_s = max(self._max_s, s)

        terms = {
            "progress": self.rw["progress_per_m"] * delta,
            "time": -self.rw["time_penalty"],
        }
        crashed = self._prev_speed - tel.speed_kmh > self.track["crash_speed_drop_kmh"]
        self._prev_speed = tel.speed_kmh
        if crashed:
            terms["crash"] = -self.rw["crash_penalty"]

        split_delta = self.splits.update(tel.cp_count, tel.last_cp_time_ms)
        if split_delta is not None:
            bonus = -split_delta / 1000.0 * self.rw["split_bonus_per_s"]
            terms["split"] = float(np.clip(bonus, -self.rw["split_bonus_cap"], self.rw["split_bonus_cap"]))

        done = None
        if tel.finished:
            done = "finished"
            terms["finish"] = self.rw["finish_bonus"]
        elif self._max_s - s > self.track["wrong_way_m"]:
            done = "wrong_way"
        elif self.track["off_track_m"] > 0 and lateral > self.track["off_track_m"]:
            done = "off_track"

        self._done = done
        lookahead = self.ref.lookahead(tel.pos, tel.dir, tel.up, s,
                                       self.env["lookahead_points"], self.env["lookahead_spacing_m"])
        return StepInfo(
            reward=float(sum(terms.values())), progress_m=s,
            progress=s / self.ref.length if self.ref.length else 0.0,
            lateral_m=lateral, delta_m=delta, lookahead=lookahead, done=done,
            split_delta_ms=split_delta, crashed=crashed, terms=terms,
        )
