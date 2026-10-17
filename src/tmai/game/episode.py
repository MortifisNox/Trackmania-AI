"""Decides when an attempt is over: finished, stuck, or out of time."""

from __future__ import annotations

from tmai.game.protocol import Telemetry


class EpisodeMonitor:
    def __init__(self, timeout_s: float, stuck_timeout_s: float, stuck_speed_kmh: float,
                 grace_s: float = 2.0):
        self.timeout_ms = timeout_s * 1000
        self.stuck_ms = stuck_timeout_s * 1000
        self.stuck_speed = stuck_speed_kmh
        self.grace_ms = grace_s * 1000   # the car starts from standstill
        self._slow_since: int | None = None

    def reset(self) -> None:
        self._slow_since = None

    def check(self, t: Telemetry) -> str | None:
        if t.finished:
            return "finished"
        if t.race_time_ms >= self.timeout_ms:
            return "timeout"
        if t.race_time_ms < self.grace_ms:
            self._slow_since = None
            return None
        if t.speed_kmh < self.stuck_speed:
            if self._slow_since is None:
                self._slow_since = t.race_time_ms
            elif t.race_time_ms - self._slow_since >= self.stuck_ms:
                return "stuck"
        else:
            self._slow_since = None
        return None
