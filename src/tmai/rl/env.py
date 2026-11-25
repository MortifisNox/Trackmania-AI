"""The real-time training environment: one step = send an action, wait for the
next 20 Hz slot, read the game. Combines the game interface, track progress
and the episode monitor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect

import numpy as np

from tmai.game.clock import RealtimeClock
from tmai.game.episode import EpisodeMonitor
from tmai.game.interface import RestartAborted
from tmai.rl.features import frame_size, state_vector, to_controls
from tmai.track.progress import ProgressTracker, StepInfo

# Episode ends after which the future is worth nothing (no bootstrapping).
# Timeout, stuck and bad_pose are cut-offs, not outcomes, so learning bootstraps through them.
TERMINAL = {"finished", "wrong_way", "off_track"}
NO_INPUT = np.array([0.0, -1.0, -1.0], dtype=np.float32)
FULL_GAS = np.array([0.0, 1.0, -1.0], dtype=np.float32)
BAD_POSE_TIMEOUT_S = 1.0


class MapMismatchError(RuntimeError):
    """The map loaded in the game does not match the fitted LiDAR geometry."""


def _accepts_kwarg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


@dataclass
class Obs:
    frames: np.ndarray        # (stack, H, W) uint8, newest last
    state: np.ndarray         # (STATE_DIM,) float32
    frame: np.ndarray         # newest frame alone, for the replay buffer


class TMEnv:
    def __init__(self, cfg: dict, game, tracker: ProgressTracker, clock: RealtimeClock | None = None,
                 lidar=None, log=print):
        env = cfg["env"]
        self.lidar = lidar        # tmai.track.lidar.Lidar when env.lidar is on
        self.env_cfg = env
        self.game, self.tracker = game, tracker
        self.log = log
        # State mode keeps no camera; an empty frame stands in for it everywhere.
        self._no_frame = np.zeros((frame_size(env),) * 2, dtype=np.uint8)
        self.stack = env["frame_stack"]
        self.clock = clock or RealtimeClock(env["step_hz"])
        self.monitor = EpisodeMonitor(env["episode_timeout_s"], env["stuck_timeout_s"], env["stuck_speed_kmh"])
        self._frames: deque = deque(maxlen=self.stack)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self.last_tel = None
        self.lateness: list[float] = []
        self._warned_no_uid = False
        self._bad_pose_s = 0.0

    def _obs(self, frame, tel, lookahead) -> Obs:
        if frame is None:
            frame = self._no_frame
        self._frames.append(frame)
        feats = self.lidar.features(tel) if self.lidar is not None else None
        state = state_vector(tel, lookahead, self._prev_action, self.env_cfg, feats)
        return Obs(np.stack(self._frames), state, frame)

    def _check_map_uid(self, tel) -> None:
        if self.lidar is None:
            return
        expected_uid = getattr(self.lidar, "uid", "") or ""
        if not expected_uid:
            return
        live_uid = getattr(tel, "map_uid", "") or ""
        if live_uid:
            if live_uid != expected_uid:
                raise MapMismatchError(
                    f"wrong map loaded for LiDAR: game reports map UID {live_uid!r}, "
                    f"but geometry was fitted for {expected_uid!r}; load the matching map "
                    f"or re-run `tmai geometry-check`"
                )
        elif not self._warned_no_uid:
            self._warned_no_uid = True
            self.log(
                "WARNING: telemetry packet has no map UID (old TMAI Telemetry plugin?); "
                "cannot verify the loaded map matches the LiDAR geometry. "
                "Run `tmai install-plugin` and reload TMAI Telemetry."
            )

    def reset(self, should_abort=None) -> Obs:
        self.game.pad.release()
        if should_abort is not None and should_abort():
            raise RestartAborted("restart aborted")
        if should_abort is not None and _accepts_kwarg(self.game.restart, "should_abort"):
            self.game.restart(should_abort=should_abort)
        else:
            self.game.restart()
        try:
            if should_abort is not None and should_abort():
                raise RestartAborted("restart aborted")
            self.monitor.reset()
            self.clock.reset()
            if self.lidar is not None:
                self.lidar.reset()   # no scan carries over from the last attempt
            self._bad_pose_s = 0.0
            self._prev_action = (FULL_GAS if self.env_cfg.get("launch_hold_gas", True) else NO_INPUT).copy()
            frame, tel = self.game.observe()
            self._check_map_uid(tel)
            if frame is None:
                frame = self._no_frame
            self.tracker.reset(tel)
            self.last_tel = tel
            self._frames.clear()
            for _ in range(self.stack - 1):
                self._frames.append(frame)
            return self._obs(frame, tel, self.tracker.lookahead(tel))
        except BaseException:
            self.game.pad.release()
            raise

    def step(self, action) -> tuple[Obs, float, str | None, StepInfo]:
        """Returns (observation, reward, end reason or None, progress info)."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.game.act(to_controls(action, self.env_cfg.get("digital_pedals", True)))
        self.lateness.append(self.clock.tick())
        frame, tel = self.game.observe()
        if frame is None:
            frame = self._no_frame
        prev_tel = self.last_tel
        self.last_tel = tel
        info = self.tracker.step(tel)
        reason = info.done or self.monitor.check(tel)
        self._prev_action = action
        obs = self._obs(frame, tel, info.lookahead)
        if self.lidar is not None:
            if getattr(self.lidar, "last_valid", True):
                self._bad_pose_s = 0.0
            else:
                step_dt = 1.0 / self.env_cfg["step_hz"]
                race_dt = (
                    max(0.0, (tel.race_time_ms - prev_tel.race_time_ms) / 1000.0)
                    if prev_tel is not None else 0.0
                )
                self._bad_pose_s += max(step_dt, race_dt)
                if reason is None and self._bad_pose_s > BAD_POSE_TIMEOUT_S + 1e-9:
                    reason = "bad_pose"
        if reason:
            self.game.pad.release()
        return obs, info.reward, reason, info
