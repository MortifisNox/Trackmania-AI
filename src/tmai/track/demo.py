"""Records human laps (telemetry, frames and the inputs the game applied) at the
agent's control rate. Finished laps seed the reference line and the replay buffer."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tmai.game.clock import RealtimeClock
from tmai.game.protocol import Telemetry

FIELDS = ("pos", "vel", "dir", "up", "speed", "gear", "rpm", "race_time_ms",
          "cp_count", "last_cp_time_ms", "wheel_material", "wheel_slip")


class LapBuffer:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.rows: dict[str, list] = {k: [] for k in FIELDS}
        self.actions: list[tuple] = []

    def add(self, frame: np.ndarray | None, t: Telemetry) -> None:
        if frame is not None:
            self.frames.append(frame)
        for k in FIELDS:
            self.rows[k].append(getattr(t, k))
        self.actions.append((t.steer, t.gas, t.brake))

    def __len__(self) -> int:
        return len(self.actions)

    def cp_times(self) -> list[int]:
        times, last = [], 0
        for count, cp_time in zip(self.rows["cp_count"], self.rows["last_cp_time_ms"]):
            if count > last:
                times.append(int(cp_time))
                last = count
        return times

    def save(self, folder: Path, lap_time_ms: int) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{lap_time_ms}ms.npz"
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        arrays["actions"] = np.asarray(self.actions, dtype=np.float32)
        if self.frames:
            arrays["frames"] = np.stack(self.frames).astype(np.uint8)
        np.savez_compressed(path, lap_time_ms=lap_time_ms, cp_times=np.asarray(self.cp_times()), **arrays)
        return path


def record_laps(game, folder: Path, laps: int, step_hz: float, log=print) -> list[Path]:
    """Record until `laps` finished laps are saved. Unfinished attempts (restart
    before the finish) are dropped."""
    clock = RealtimeClock(step_hz)
    saved: list[Path] = []
    buf: LapBuffer | None = None
    prev: Telemetry | None = None
    log("Drive the map. Each finished lap is saved; restart any time to discard an attempt.")
    while len(saved) < laps:
        clock.tick()
        frame, t = game.observe()
        restarted = prev is not None and t.race_time_ms < prev.race_time_ms - 500
        if restarted and buf is not None and len(buf):
            log(f"  attempt discarded at {prev.race_time_ms / 1000:.2f}s")
            buf = None
        if buf is None and t.playing and 0 <= t.race_time_ms < 1000:
            buf = LapBuffer()
        if buf is not None and t.playing and t.race_time_ms >= 0:
            buf.add(frame, t)
        if buf is not None and t.finished:
            lap_ms = t.last_cp_time_ms or t.race_time_ms
            buf.add(frame, t)
            path = buf.save(folder, lap_ms)
            saved.append(path)
            log(f"  lap {len(saved)}/{laps}: {lap_ms / 1000:.3f}s, {len(buf)} steps -> {path.name}")
            buf = None
        prev = t
    return saved
