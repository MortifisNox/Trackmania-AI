"""The real-time training environment: one step = send an action, wait for the
next 20 Hz slot, read the game. Combines the game interface, track progress
and the episode monitor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from tmai.game.clock import RealtimeClock
from tmai.game.episode import EpisodeMonitor
from tmai.rl.features import state_vector, to_controls
from tmai.track.progress import ProgressTracker, StepInfo

# Episode ends after which the future is worth nothing (no bootstrapping).
# Timeout and stuck are cut-offs, not outcomes, so learning bootstraps through them.
TERMINAL = {"finished", "wrong_way", "off_track"}


@dataclass
class Obs:
    frames: np.ndarray        # (stack, H, W) uint8, newest last
    state: np.ndarray         # (STATE_DIM,) float32
    frame: np.ndarray         # newest frame alone, for the replay buffer


class TMEnv:
    def __init__(self, cfg: dict, game, tracker: ProgressTracker, clock: RealtimeClock | None = None):
        env = cfg["env"]
        self.game, self.tracker = game, tracker
        self.stack = env["frame_stack"]
        self.lidar = bool(env["lidar_beams"])
        self.clock = clock or RealtimeClock(env["step_hz"])
        self.monitor = EpisodeMonitor(env["episode_timeout_s"], env["stuck_timeout_s"], env["stuck_speed_kmh"])
        self._frames: deque = deque(maxlen=self.stack)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self.last_tel = None
        self.lateness: list[float] = []

    def _obs(self, frame, tel, lookahead) -> Obs:
        self._frames.append(frame)
        state = state_vector(tel, lookahead, self._prev_action, self._beams())
        return Obs(np.stack(self._frames), state, frame)

    def _beams(self):
        if not self.lidar:
            return None
        beams = getattr(self.game, "beams", None)
        if beams is None:
            raise RuntimeError("env.lidar_beams is on but the game gave no beam readings")
        return beams

    def reset(self) -> Obs:
        self.game.pad.release()
        self.game.restart()
        self.monitor.reset()
        self.clock.reset()
        self._prev_action = np.array([0.0, -1.0, -1.0], dtype=np.float32)   # no input
        frame, tel = self.game.observe()
        self.tracker.reset(tel)
        self.last_tel = tel
        self._frames.clear()
        for _ in range(self.stack - 1):
            self._frames.append(frame)
        return self._obs(frame, tel, self.tracker.lookahead(tel))

    def step(self, action) -> tuple[Obs, float, str | None, StepInfo]:
        """Returns (observation, reward, end reason or None, progress info)."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.game.act(to_controls(action))
        self.lateness.append(self.clock.tick())
        frame, tel = self.game.observe()
        self.last_tel = tel
        info = self.tracker.step(tel)
        reason = info.done or self.monitor.check(tel)
        self._prev_action = action
        obs = self._obs(frame, tel, info.lookahead)
        if reason:
            self.game.pad.release()
        return obs, info.reward, reason, info
