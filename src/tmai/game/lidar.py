"""Screen "lidar": beams cast across the game image from a point near the car
until they reach the dark track border, as in TMRL. Each beam's length tells
the agent how much room there is in that direction. No torch here.

It only sees borders that are darker than the road (like tmrl-test's); check
the beams on your map with `tmai capture-test`, which draws them."""

from __future__ import annotations

import numpy as np


def wide_size(width: int, height: int, lidar_width: int) -> tuple[int, int]:
    """(w, h) of the frame the beams are measured on, keeping the aspect ratio."""
    return lidar_width, max(1, round(lidar_width * height / max(width, 1)))


class Lidar:
    def __init__(self, beams: int, origin=(0.5, 0.9), dark: int = 60, hits: int = 2):
        """`beams` spread evenly from straight left (-90 deg) through straight up
        the screen to straight right; `origin` is (x, y) as fractions of the
        image; a beam stops at the first `hits` pixels in a row darker than `dark`."""
        self.beams = beams
        self.origin = origin
        self.dark = dark
        self.hits = hits
        self.angles = np.radians(np.linspace(-90.0, 90.0, beams)) if beams > 1 else np.zeros(beams)
        self._shape: tuple[int, int] | None = None
        self._rays: list[tuple[np.ndarray, np.ndarray]] = []

    def _build(self, h: int, w: int) -> None:
        ox, oy = self.origin[0] * (w - 1), self.origin[1] * (h - 1)
        steps = np.arange(int(np.hypot(h, w)) + 1, dtype=np.float64)
        self._rays = []
        for a in self.angles:
            xs = np.rint(ox + steps * np.sin(a)).astype(int)
            ys = np.rint(oy - steps * np.cos(a)).astype(int)
            inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            n = int(np.argmin(inside)) if not inside.all() else len(inside)
            self._rays.append((ys[:n], xs[:n]))
        self._shape = (h, w)
        self.scale = float(h)   # beam lengths are given in image heights

    def measure(self, gray: np.ndarray) -> np.ndarray:
        """Beam lengths for one grayscale frame, as fractions of the image
        height, clipped to 0..1 (1 = clear road or the edge of the picture)."""
        if self._shape != gray.shape[:2]:
            self._build(*gray.shape[:2])
        out = np.empty(self.beams, dtype=np.float32)
        for i, (ys, xs) in enumerate(self._rays):
            if len(ys) < self.hits:
                out[i] = len(ys)
                continue
            dark = gray[ys, xs] < self.dark
            run = np.convolve(dark, np.ones(self.hits, dtype=int), mode="valid") >= self.hits
            stop = int(np.argmax(run)) if run.any() else len(ys)
            out[i] = stop
        return np.clip(out / self.scale, 0.0, 1.0)

    def endpoints(self, gray: np.ndarray, lengths: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Pixel (x, y) start and end of each beam, for drawing them."""
        if self._shape != gray.shape[:2]:
            self._build(*gray.shape[:2])
        h, w = gray.shape[:2]
        start = (round(self.origin[0] * (w - 1)), round(self.origin[1] * (h - 1)))
        out = []
        for (ys, xs), length in zip(self._rays, lengths):
            k = min(int(length * self.scale), len(ys) - 1)
            out.append((start, (int(xs[k]), int(ys[k])) if len(ys) else start))
        return out


def from_config(env: dict) -> Lidar | None:
    if not env.get("lidar_beams"):
        return None
    return Lidar(env["lidar_beams"], tuple(env["lidar_origin"]), env["lidar_dark"])
