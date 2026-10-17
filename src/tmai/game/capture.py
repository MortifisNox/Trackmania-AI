"""Grabs the Trackmania window's client area and shrinks it for the CNN."""

from __future__ import annotations

import numpy as np

GAME_EXE = "trackmania.exe"


def to_model_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Grayscale uint8 frame (H, W) or (H, W, 1|3|4) -> (size, size) uint8."""
    import cv2

    if frame.ndim == 3:
        if frame.shape[2] == 1:
            frame = frame[:, :, 0]
        else:
            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def _make_dpi_aware() -> None:
    # Without this, laptops at 125-150% display scaling report scaled window
    # coordinates and the capture box lands in the wrong place.
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def find_game_window() -> int:
    """HWND of the visible Trackmania window, found by process name (not title,
    since a PowerShell window in the project folder also says 'Trackmania')."""
    import psutil
    import win32gui
    import win32process

    pids = {p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == GAME_EXE}
    if not pids:
        raise RuntimeError("Trackmania is not running")
    found: list[int] = []

    def visit(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32process.GetWindowThreadProcessId(hwnd)[1] in pids:
            if win32gui.GetWindowText(hwnd):
                found.append(hwnd)

    win32gui.EnumWindows(visit, None)
    if not found:
        raise RuntimeError("Trackmania is running but its window was not found (minimised?)")
    return found[0]


def client_region(hwnd: int) -> tuple[int, int, int, int]:
    import win32gui

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    x0, y0 = win32gui.ClientToScreen(hwnd, (left, top))
    x1, y1 = win32gui.ClientToScreen(hwnd, (right, bottom))
    return x0, y0, x1, y1


class ScreenCapture:
    def __init__(self, size: int):
        import dxcam

        _make_dpi_aware()
        self.size = size
        self.hwnd = find_game_window()
        self.region = client_region(self.hwnd)
        self._cam = dxcam.create(output_color="GRAY")
        self._last: np.ndarray | None = None
        self.new_frames = 0

    def grab(self) -> np.ndarray:
        """Newest frame as (size, size) uint8. dxcam returns None when the
        screen has not changed, so the previous frame is reused then."""
        frame = self._cam.grab(region=self.region)
        if frame is not None:
            self._last = to_model_frame(frame, self.size)
            self.new_frames += 1
        if self._last is None:
            raise RuntimeError("no frame captured yet; is the game window on screen?")
        return self._last

    def close(self) -> None:
        self._cam.release()
