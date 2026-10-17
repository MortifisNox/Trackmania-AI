"""Loads recorded demo laps into a replay buffer, with rewards and observations
recomputed against the current reference line, so they match what the agent sees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tmai.budget import STATE_DIM
from tmai.game import protocol
from tmai.game.protocol import Telemetry
from tmai.rl.features import from_controls, state_vector
from tmai.rl.replay import ReplayBuffer
from tmai.track.progress import ProgressTracker

_DEFAULT_FLAGS = protocol.FLAG_PLAYER | protocol.FLAG_PLAYING | protocol.FLAG_GROUND


def demo_telemetry(d) -> list[Telemetry]:
    """Rebuild one Telemetry per recorded step. Demos recorded before flags were
    saved get 'playing, on ground' everywhere and 'finished' on the last step."""
    n = len(d["actions"])
    flags = d["flags"] if "flags" in d.files else None
    out = []
    for i in range(n):
        f = int(flags[i]) if flags is not None else _DEFAULT_FLAGS
        if flags is None and i == n - 1:
            f |= protocol.FLAG_FINISHED
        steer, gas, brake = (float(x) for x in d["actions"][i])
        out.append(Telemetry(
            flags=f, seq=i, race_time_ms=int(d["race_time_ms"][i]), cp_count=int(d["cp_count"][i]),
            last_cp_time_ms=int(d["last_cp_time_ms"][i]), pos=tuple(d["pos"][i]),
            speed=float(d["speed"][i]), gear=int(d["gear"][i]), rpm=float(d["rpm"][i]),
            steer=steer, gas=gas, brake=brake, vel=tuple(d["vel"][i]), dir=tuple(d["dir"][i]),
            up=tuple(d["up"][i]), front_speed=0.0, side_speed=0.0,
            wheel_material=tuple(int(x) for x in d["wheel_material"][i]),
            wheel_slip=tuple(d["wheel_slip"][i]),
        ))
    return out


def load_demo(path: Path, buffer: ReplayBuffer, tracker: ProgressTracker) -> tuple[int, float]:
    """Adds one demo lap; returns (transitions added, total reward). When the
    buffer's states include lidar beams, laps recorded without them are skipped."""
    n_beams = buffer.state.shape[1] - STATE_DIM
    with np.load(path) as d:
        if "frames" not in d.files:
            return 0, 0.0
        if n_beams and ("beams" not in d.files or d["beams"].shape[1:] != (n_beams,)):
            return 0, 0.0
        frames = d["frames"]
        beams = d["beams"] if n_beams else [None] * len(frames)
        tels = demo_telemetry(d)
    if len(frames) != len(tels) or len(tels) < 2:
        return 0, 0.0
    actions = [from_controls(t.steer, t.gas, t.brake) for t in tels]
    buffer.start_episode()
    tracker.reset(tels[0])
    state = state_vector(tels[0], tracker.lookahead(tels[0]), actions[0], beams[0])
    total, added = 0.0, 0
    for i in range(len(tels) - 1):
        info = tracker.step(tels[i + 1])
        # The inputs seen at step i+1 are the ones held between steps i and i+1.
        buffer.add(frames[i], state, actions[i + 1], info.reward, terminal=info.done == "finished")
        total += info.reward
        added += 1
        state = state_vector(tels[i + 1], info.lookahead, actions[i + 1], beams[i + 1])
        if info.done:
            break
    buffer.add(frames[i + 1], state)
    return added, total
