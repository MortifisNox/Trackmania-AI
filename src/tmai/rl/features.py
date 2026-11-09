"""Turns telemetry and track progress into the agent's observation and maps the
policy's actions to controller inputs. No torch here, so it is testable anywhere.

Two observation modes (env.observation):
- image: 64x64 camera frames for the CNN plus the 48-float state vector.
- state: no camera at all. The policy drives from game state alone (telemetry
  from the Openplanet plugin plus the reference line ahead), with the car's
  pitch and roll added and the lookahead allowed to reach much further."""

from __future__ import annotations

import numpy as np

from tmai.game.protocol import Telemetry

LOOKAHEAD_SCALE = 50.0   # metres; lookahead points are divided by this (or their distance, if further)
HEAD_DIM = 18            # telemetry and previous action, before the lookahead points
ATTITUDE_DIM = 3         # world up in the car's frame (pitch and roll), state mode only
WORLD_UP = (0.0, 1.0, 0.0)   # Trackmania's world is Y-up


def uses_camera(env: dict) -> bool:
    return env.get("observation", "image") == "image"


def frame_size(env: dict) -> int:
    """Side of the stored frames; 0 in state mode, where no frames are kept."""
    return env["frame_size"] if uses_camera(env) else 0


def state_dim(env: dict) -> int:
    return HEAD_DIM + 3 * env["lookahead_points"] + (0 if uses_camera(env) else ATTITUDE_DIM)


def lookahead_distances(env: dict) -> np.ndarray:
    """Distance ahead of each lookahead point. Spacing grows by
    env.lookahead_growth per point (1.0 = evenly spaced)."""
    growth = float(env.get("lookahead_growth", 1.0))
    gaps = env["lookahead_spacing_m"] * growth ** np.arange(env["lookahead_points"])
    return np.cumsum(gaps)


def car_frame(vec, forward, up) -> np.ndarray:
    """World vector -> (forward, left, up) components in the car's frame."""
    f = np.asarray(forward, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    f /= max(np.linalg.norm(f), 1e-9)
    u /= max(np.linalg.norm(u), 1e-9)
    left = np.cross(u, f)
    v = np.asarray(vec, dtype=np.float64)
    return np.array([v @ f, v @ left, v @ u])


def state_vector(tel: Telemetry, lookahead: np.ndarray, prev_action, env: dict | None = None) -> np.ndarray:
    """All values roughly in -1..1:
    speed, gear, rpm, velocity in the car frame (3), on ground, wheel surface
    materials (4), wheel slip (4), previous action (3), lookahead points
    (3 per point; 10 points = 48 floats in all). State mode adds world up in
    the car's frame (3), which says how the car is pitched and rolled."""
    env = env or {"observation": "image", "lookahead_points": 10, "lookahead_spacing_m": 10.0}
    vel = car_frame(tel.vel, tel.dir, tel.up) / 50.0
    head = [
        tel.speed / 100.0, tel.gear / 5.0, tel.rpm / 11000.0, *vel,
        1.0 if tel.on_ground else 0.0,
        *(m / 80.0 for m in tel.wheel_material),
        *(min(abs(s), 2.0) for s in tel.wheel_slip),
        *prev_action,
    ]
    points = np.asarray(lookahead, dtype=np.float64).reshape(-1, 3)
    if uses_camera(env):
        points = points / LOOKAHEAD_SCALE
    else:
        # Far points are scaled by their own distance, so they stay in range.
        scale = np.maximum(LOOKAHEAD_SCALE, lookahead_distances(env))[: len(points)]
        points = points / scale[:, None]
    parts = [np.asarray(head, dtype=np.float32), points.reshape(-1).astype(np.float32)]
    if not uses_camera(env):
        parts.append(car_frame(WORLD_UP, tel.dir, tel.up).astype(np.float32))
    out = np.concatenate(parts)
    want = state_dim(env)
    if out.shape[0] != want:
        raise ValueError(f"state vector has {out.shape[0]} values, expected {want}")
    return np.clip(out, -5.0, 5.0)


def to_controls(action) -> tuple[float, float, float]:
    """Policy output in [-1, 1]^3 -> (steer -1..1, gas 0..1, brake 0..1)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float(a[0]), float((a[1] + 1) / 2), float((a[2] + 1) / 2)


def from_controls(steer: float, gas: float, brake: float) -> np.ndarray:
    """Inverse of to_controls, for the inputs recorded in demo laps."""
    return np.array([steer, 2 * gas - 1, 2 * brake - 1], dtype=np.float32).clip(-1, 1)
