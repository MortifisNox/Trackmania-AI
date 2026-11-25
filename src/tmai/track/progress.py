"""Turns telemetry into progress along the reference line, reward, and track-based
episode ends (wrong way, off track)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tmai.game.protocol import Telemetry
from tmai.track.baseline import PaceBaseline
from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker

RESYNC_STEPS = 10
LOCAL_LATERAL_M = 15.0


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
    # (lap fraction before, after, race seconds) for the pace term, so replay can
    # score it again against a newer baseline; None when the step has no pace term.
    pace_inputs: tuple[float, float, float] | None = None
    lap_ms: int | None = None


class ProgressTracker:
    def __init__(self, ref: ReferenceLine, cfg: dict, splits: SplitTracker | None = None,
                 baseline: PaceBaseline | None = None):
        self.ref = ref
        self.baseline = baseline or PaceBaseline(ref)   # default: pace against the reference lap
        self.best_lap_ms: int | None = None              # personal best, for the finish bonus
        self.track = cfg["track"]
        self.rw = cfg["reward"]
        self.env = cfg["env"]
        self.splits = splits or SplitTracker()
        self._idx: int | None = None
        self._s = 0.0
        self._max_s = 0.0
        self._prev_speed = 0.0
        self._prev_time_ms = 0
        self._prev_tel: Telemetry | None = None
        self._rejections = 0
        self._done: str | None = None

    def reset(self, tel: Telemetry) -> None:
        self._idx, self._s, _ = self.ref.project(tel.pos, start_only=True)
        self._max_s = self._s
        self._prev_speed = tel.speed_kmh
        # Count race time from 0, so a run that only starts being driven after the
        # clock started (a slow restart) is charged for the time already gone.
        self._prev_time_ms = min(tel.race_time_ms, 0)
        self._prev_tel = tel
        self._rejections = 0
        self._done = None
        self.splits.reset()

    def lookahead(self, tel: Telemetry) -> np.ndarray:
        """Reference points ahead of the car's current progress, in its frame."""
        return self.ref.lookahead(tel.pos, tel.dir, tel.up, self._s,
                                  self.env["lookahead_points"], self.env["lookahead_spacing_m"],
                                  self.env.get("lookahead_growth", 1.0))

    @property
    def ended(self) -> str | None:
        return self._done

    def step(self, tel: Telemetry, finish_lap_ms: int | None = None) -> StepInfo:
        if self._done:
            # The attempt already ended; the game keeps sending frames until the
            # restart (the finish screen, the car rolling on), which earn nothing.
            return StepInfo(reward=0.0, progress_m=self._s,
                            progress=self._s / self.ref.length if self.ref.length else 0.0,
                            lateral_m=0.0, delta_m=0.0,
                            lookahead=np.zeros(3 * self.env["lookahead_points"]),
                            done=self._done)
        idx, s, lateral = self.ref.project(tel.pos, hint=self._idx)
        local_lateral = lateral
        max_jump = self.track["max_progress_jump_m"]
        forward_limit = max_jump
        prev = self._prev_tel
        if self._idx is not None and lateral <= LOCAL_LATERAL_M and prev is not None:
            elapsed = (tel.race_time_ms - max(prev.race_time_ms, 0)) / 1000.0
            # A fast car can legitimately travel farther than max_jump during a hitch.
            forward_limit = max(max_jump, abs(tel.speed) * max(elapsed, 0.0) * 1.5 + 5.0)
        if lateral > LOCAL_LATERAL_M:
            # A nearby branch can explain a teleport. Keep the local anchor
            # until several consecutive samples support leaving it.
            far = self.ref.project(tel.pos)
            if far[2] <= LOCAL_LATERAL_M and abs(far[1] - self._s) > max_jump:
                idx, s, lateral = far
        delta = s - self._s
        rejected = delta < -max_jump or delta > forward_limit
        had_rejections = self._rejections > 0
        self._rejections = self._rejections + 1 if rejected else 0
        if rejected:
            delta = 0.0
            if self._rejections < RESYNC_STEPS:
                idx, s, lateral = self._idx, self._s, local_lateral
            else:
                self._max_s = s
                self._rejections = 0
        # Race time that passed this step (none during the countdown). Normally one
        # step's worth; more when the game or the loop stalled, and time_penalty
        # charges all of it while the pace term stays bounded per step (dt <= 0 -> 0).
        dt = (max(tel.race_time_ms, 0) - max(self._prev_time_ms, 0)) / 1000.0
        pace, pace_inputs = 0.0, None
        if self.baseline.times is not None and self.rw["pace_per_s"]:
            # Seconds gained on the pace baseline over this stretch of track.
            length = self.ref.length or 1.0
            u0 = self._s / length
            pace_inputs = (u0, s / length if delta else u0, max(dt, 0.0))
            pace = float(self.baseline.pace(*pace_inputs))
        lap_ms = None
        if tel.finished:
            lap_ms = finish_lap_ms if finish_lap_ms is not None else self._finish_time(
                tel, s, lateral, rejected or had_rejections, forward_limit)
        prev_tel, self._prev_tel = self._prev_tel, tel
        self._prev_time_ms = tel.race_time_ms
        self._idx, self._s = idx, s
        self._max_s = max(self._max_s, s)

        terms = {
            "progress": self.rw["progress_per_m"] * delta,
            "time": -self.rw["time_penalty"] * max(1.0, dt * self.env["step_hz"]),
        }
        if pace:
            terms["pace"] = self.rw["pace_per_s"] * pace
        crashed = self._prev_speed - tel.speed_kmh > self.track["crash_speed_drop_kmh"]
        self._prev_speed = tel.speed_kmh
        if crashed:
            terms["crash"] = -self.rw["crash_penalty"]

        split_ms = tel.last_cp_time_ms
        if lap_ms is not None and (prev_tel is None or tel.cp_count > prev_tel.cp_count):
            split_ms = lap_ms
        split_delta = self.splits.update(tel.cp_count, split_ms)
        if split_delta is not None:
            bonus = -split_delta / 1000.0 * self.rw["split_bonus_per_s"]
            terms["split"] = float(np.clip(bonus, -self.rw["split_bonus_cap"], self.rw["split_bonus_cap"]))

        done = None
        if tel.finished:
            done = "finished"
            terms["finish"] = self.rw["finish_bonus"]
            if self.best_lap_ms and lap_ms < self.best_lap_ms:
                gain_s = (self.best_lap_ms - lap_ms) / 1000.0
                terms["personal_best"] = min(self.rw["pb_bonus_cap"], self.rw["pb_bonus_per_s"] * gain_s)
        elif self._max_s - s > self.track["wrong_way_m"]:
            done = "wrong_way"
        elif not rejected and self.track["off_track_m"] > 0 and lateral > self.track["off_track_m"]:
            done = "off_track"

        self._done = done
        lookahead = self.ref.lookahead(tel.pos, tel.dir, tel.up, s,
                                       self.env["lookahead_points"], self.env["lookahead_spacing_m"],
                                       self.env.get("lookahead_growth", 1.0))
        return StepInfo(
            reward=float(sum(terms.values())), progress_m=s,
            progress=s / self.ref.length if self.ref.length else 0.0,
            lateral_m=lateral, delta_m=delta, lookahead=lookahead, done=done,
            split_delta_ms=split_delta, crashed=crashed, terms=terms,
            pace_inputs=pace_inputs, lap_ms=lap_ms,
        )

    def _finish_time(self, tel: Telemetry, s: float, lateral: float,
                     unreliable_projection: bool, forward_limit: float) -> int:
        prev = self._prev_tel
        race_ms = tel.race_time_ms
        if tel.exact_finish_ms is not None:
            return tel.exact_finish_ms
        if tel.last_cp_time_ms > 0 and (prev is None or (
            tel.cp_count > prev.cp_count and tel.last_cp_time_ms > prev.last_cp_time_ms
        )):
            return tel.last_cp_time_ms
        if prev is not None:
            start_ms = max(0, prev.race_time_ms)
            if race_ms <= start_ms:
                return start_ms
            if not unreliable_projection:
                gap = self.ref.length - s
                if gap <= 0:
                    travel = s + self.ref.finish_overshoot(tel.pos) - self._s
                    if 0 < travel <= forward_limit:
                        fraction = (self.ref.length - self._s) / travel
                        return round(start_ms + fraction * (race_ms - start_ms))
                elif lateral <= LOCAL_LATERAL_M and s >= self._s:
                    tangent = self.ref.points[-1] - self.ref.points[-2]
                    rate = float(tangent @ tel.vel) / max(np.linalg.norm(tangent), 1e-12)
                    if not any(tel.vel):
                        rate = (s - self._s) * 1000.0 / (race_ms - start_ms)
                    step_s = 1.0 / self.env["step_hz"]
                    if rate > 0 and gap <= rate * step_s + self.ref.spacing:
                        return race_ms + min(int(step_s * 1000), round(gap / rate * 1000))
        return max(0, race_ms)
