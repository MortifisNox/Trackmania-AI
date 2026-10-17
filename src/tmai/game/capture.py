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


def to_wide_frame(gray: np.ndarray, width: int) -> np.ndarray:
    """Grayscale (H, W) -> `width` pixels wide, same aspect ratio, for the lidar."""
    import cv2

    from tmai.game.lidar import wide_size

    return cv2.resize(gray, wide_size(gray.shape[1], gray.shape[0], width), interpolation=cv2.INTER_AREA)


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


def window_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """Visible window rectangle (what Windows Graphics Capture records)."""
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
    )
    return rect.left, rect.top, rect.right, rect.bottom


def crop_to_client(frame: np.ndarray, bounds: tuple, client: tuple) -> np.ndarray:
    """Cut the title bar and borders off a whole-window frame.

    `bounds` and `client` are screen rectangles (x0, y0, x1, y1); the frame may
    be scaled relative to them, so offsets are scaled to the frame's size."""
    h, w = frame.shape[:2]
    bw, bh = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if bw <= 0 or bh <= 0:
        return frame
    sx, sy = w / bw, h / bh
    x0 = max(0, round((client[0] - bounds[0]) * sx))
    y0 = max(0, round((client[1] - bounds[1]) * sy))
    x1 = min(w, round((client[2] - bounds[0]) * sx))
    y1 = min(h, round((client[3] - bounds[1]) * sy))
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1]


class WindowCapture:
    """Captures the game window itself through Windows Graphics Capture, so it
    keeps working when other windows cover the game or it loses focus (but not
    when it is minimised: Windows stops rendering minimised windows)."""

    def __init__(self, size: int, lidar_width: int = 0):
        import threading

        from windows_capture import WindowsCapture

        _make_dpi_aware()
        self.size = size
        self.lidar_width = lidar_width
        self.hwnd = find_game_window()
        self.region = client_region(self.hwnd)
        self._bounds = window_bounds(self.hwnd)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._last: np.ndarray | None = None
        self._wide: np.ndarray | None = None
        self.new_frames = 0
        self.closed = False

        cap = WindowsCapture(cursor_capture=False, window_hwnd=self.hwnd)
        cap.frame_handler = self._on_frame
        cap.closed_handler = self._on_closed
        self._control = cap.start_free_threaded()

    def _on_frame(self, frame, control) -> None:
        # The buffer is only valid inside this callback, so shrink it right away.
        bgra = crop_to_client(frame.frame_buffer, self._bounds, self.region)
        import cv2

        gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        wide = to_wide_frame(gray, self.lidar_width) if self.lidar_width else None
        with self._lock:
            self._last = small
            self._wide = wide
            self.new_frames += 1
        self._ready.set()

    def _on_closed(self) -> None:
        self.closed = True

    def grab(self) -> np.ndarray:
        if not self._ready.wait(5.0):
            raise RuntimeError("no frame from the game window; is it minimised?")
        with self._lock:
            return self._last

    def grab_wide(self) -> np.ndarray | None:
        """The newest frame at lidar resolution (only with lidar_width set)."""
        with self._lock:
            return self._wide

    def close(self) -> None:
        try:
            self._control.stop()
        except Exception:
            pass


def open_capture(size: int, backend: str, lidar_width: int = 0):
    """`window` keeps working behind other windows; `screen` (dxcam) needs the
    game visible on screen. Falls back to `screen` if window capture fails.
    With `lidar_width`, a wider grayscale frame is kept too, for the beams."""
    if backend == "window":
        try:
            return WindowCapture(size, lidar_width)
        except Exception as exc:  # missing package, old Windows build, etc.
            print(f"window capture unavailable ({type(exc).__name__}: {exc}); using screen capture")
    return ScreenCapture(size, lidar_width)


class ScreenCapture:
    def __init__(self, size: int, lidar_width: int = 0):
        import dxcam

        _make_dpi_aware()
        self.size = size
        self.lidar_width = lidar_width
        self._wide: np.ndarray | None = None
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
            if self.lidar_width:
                self._wide = to_wide_frame(frame[:, :, 0] if frame.ndim == 3 else frame, self.lidar_width)
            self.new_frames += 1
        if self._last is None:
            raise RuntimeError("no frame captured yet; is the game window on screen?")
        return self._last

    def grab_wide(self) -> np.ndarray | None:
        return self._wide

    def close(self) -> None:
        self._cam.release()
