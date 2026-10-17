"""Keeps the virtual controller away from Windows' own UI.

The virtual pad is a real Xbox controller as far as Windows can tell, so any
part of Windows that accepts controller navigation (the Alt+Tab switcher, Task
View, Start, search, Game Bar, Settings and other store apps) reads the
agent's steering as menu moves and its restart button as Back. The guard says
when that would happen, so the pad can go neutral until it is safe again.

Modes (env.input_guard):
  shell  block while Alt or the Windows key is held, or a controller-navigable
         Windows surface is in front; the game may stay in the background
  game   only drive while the Trackmania window is in front
  off    never block
"""

from __future__ import annotations

import sys

MODES = ("shell", "game", "off")

GAME_EXE = "trackmania.exe"

# Alt and Win open the switcher, Task View and Start; block from the key press
# so the stick is already neutral when the overlay appears.
_KEYS = {0x12: "Alt is held", 0x5B: "the Windows key is held", 0x5C: "the Windows key is held"}

# Window classes of Windows surfaces that take controller navigation.
SHELL_CLASSES = {
    "MultitaskingViewFrame": "Alt+Tab / Task View is open",      # Windows 10
    "XamlExplorerHostIslandWindow": "Alt+Tab / Task View is open",  # Windows 11
    "TaskSwitcherWnd": "Alt+Tab is open",                        # classic switcher
    "ForegroundStaging": "Windows is switching windows",
    "Windows.UI.Core.CoreWindow": "a Windows menu or store app is in front",
    "ApplicationFrameWindow": "a Windows store app is in front",
}
SHELL_PROCESSES = {
    "startmenuexperiencehost.exe": "Start is open",
    "searchapp.exe": "Windows search is open",
    "searchhost.exe": "Windows search is open",
    "shellexperiencehost.exe": "a Windows panel is open",
    "gamebar.exe": "Xbox Game Bar is open",
    "applicationframehost.exe": "a Windows store app is in front",
}


class Foreground:
    """What is in front right now: (window class, process exe, pressed keys)."""

    def __init__(self, window_class: str, exe: str, keys: set[int]):
        self.window_class, self.exe, self.keys = window_class, exe.lower(), keys


def blocked_reason(mode: str, fg: Foreground | None) -> str | None:
    """Why the pad must stay neutral, or None when the agent may drive."""
    if mode == "off":
        return None
    if fg is None:
        return "no window is in front"
    if mode == "game":
        return None if fg.exe == GAME_EXE else "the game window is not in front"
    for vk in sorted(fg.keys):
        if vk in _KEYS:
            return _KEYS[vk]
    if fg.exe == GAME_EXE:
        return None
    return SHELL_CLASSES.get(fg.window_class) or SHELL_PROCESSES.get(fg.exe)


class _WindowsProbe:
    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self._buf = ctypes.create_unicode_buffer(256)
        self._pid = wintypes.DWORD()
        self._ctypes = ctypes
        self._names: dict[int, str] = {}

    def _exe(self, pid: int) -> str:
        name = self._names.get(pid)
        if name is None:
            import psutil

            try:
                name = psutil.Process(pid).name()
            except (psutil.Error, OSError):
                return ""   # not cached: the process may still be starting
            if len(self._names) > 256:
                self._names.clear()
            self._names[pid] = name
        return name

    def __call__(self) -> Foreground | None:
        u = self.user32
        keys = {vk for vk in _KEYS if u.GetAsyncKeyState(vk) & 0x8000}
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return None
        u.GetClassNameW(hwnd, self._buf, len(self._buf))
        u.GetWindowThreadProcessId(hwnd, self._ctypes.byref(self._pid))
        return Foreground(self._buf.value, self._exe(self._pid.value), keys)


class InputGuard:
    def __init__(self, mode: str = "shell", probe=None):
        if mode not in MODES:
            raise ValueError(f"unknown input guard {mode!r}; pick one of {MODES}")
        if probe is None and mode != "off" and sys.platform != "win32":
            mode = "off"   # nothing to guard against off Windows
        self.mode = mode
        self.probe = probe or (_WindowsProbe() if mode != "off" else None)

    @property
    def active(self) -> bool:
        return self.mode != "off"

    def blocked(self) -> str | None:
        if not self.active:
            return None
        return blocked_reason(self.mode, self.probe())
