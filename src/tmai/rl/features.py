"""Turns telemetry and track progress into the agent's observation and maps the
policy's actions to controller inputs. No torch here, so it is testable anywhere."""

from __future__ import annotations

import numpy as np

from tmai.budget import STATE_DIM
from tmai.game.protocol import Telemetry

LOOKAHEAD_SCALE = 50.0   # metres; lookahead points are divided by this


def car_frame(vec, forward, up) -> np.ndarray:
    """World vector -> (forward, left, up) components in the car's frame."""
    f = np.asarray(forward, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    f /= max(np.linalg.norm(f), 1e-9)
    u /= max(np.linalg.norm(u), 1e-9)
    left = np.cross(u, f)
    v = np.asarray(vec, dtype=np.float64)
    return np.array([v @ f, v @ left, v @ u])


def state_vector(tel: Telemetry, lookahead: np.ndarray, prev_action, beams=None) -> np.ndarray:
    """48 floats, all roughly in -1..1:
    speed, gear, rpm, velocity in the car frame (3), on ground, wheel surface
    materials (4), wheel slip (4), previous action (3), lookahead points (30);
    then one 0..1 length per lidar beam when the lidar is on."""
    vel = car_frame(tel.vel, tel.dir, tel.up) / 50.0
    head = [
        tel.speed / 100.0, tel.gear / 5.0, tel.rpm / 11000.0, *vel,
        1.0 if tel.on_ground else 0.0,
        *(m / 80.0 for m in tel.wheel_material),
        *(min(abs(s), 2.0) for s in tel.wheel_slip),
        *prev_action,
    ]
    out = np.concatenate([np.asarray(head, dtype=np.float32),
                          np.asarray(lookahead, dtype=np.float32) / LOOKAHEAD_SCALE])
    if out.shape[0] != STATE_DIM:
        raise ValueError(f"state vector has {out.shape[0]} values, expected {STATE_DIM} "
                         "(env.lookahead_points must be 10)")
    if beams is not None:
        out = np.concatenate([out, np.asarray(beams, dtype=np.float32)])
    return np.clip(out, -5.0, 5.0)


def to_controls(action) -> tuple[float, float, float]:
    """Policy output in [-1, 1]^3 -> (steer -1..1, gas 0..1, brake 0..1)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float(a[0]), float((a[1] + 1) / 2), float((a[2] + 1) / 2)


def from_controls(steer: float, gas: float, brake: float) -> np.ndarray:
    """Inverse of to_controls, for the inputs recorded in demo laps."""
    return np.array([steer, 2 * gas - 1, 2 * brake - 1], dtype=np.float32).clip(-1, 1)
