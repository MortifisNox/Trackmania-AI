"""Fixed-rate loop timing for real-time control."""

from __future__ import annotations

import time


class RealtimeClock:
    """Paces a loop at `hz`. If a step overruns by more than one period the
    schedule re-anchors to now instead of firing a burst of catch-up steps
    (the game moved on; late actions are worse than skipped ones)."""

    def __init__(self, hz: float, now=time.perf_counter, sleep=time.sleep):
        self.period = 1.0 / hz
        self._now, self._sleep = now, sleep
        self._next = None
        self.overruns = 0
        self.steps = 0

    def reset(self) -> None:
        self._next = None

    def tick(self) -> float:
        """Wait for the next slot; returns how late this slot started (s)."""
        now = self._now()
        if self._next is None:
            self._next = now + self.period
            self.steps += 1
            return 0.0
        wait = self._next - now
        if wait > 0:
            self._sleep(wait)
            late = max(0.0, self._now() - self._next)
            self._next += self.period
        else:
            late = -wait
            if late > self.period:
                self.overruns += 1
                self._next = now + self.period
            else:
                self._next += self.period
        self.steps += 1
        return late
