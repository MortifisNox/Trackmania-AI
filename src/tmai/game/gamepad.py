"""Virtual Xbox 360 controller (ViGEmBus via vgamepad)."""

from __future__ import annotations

import threading
import time

# Names accepted in config for env.restart_button.
BUTTONS = {
    "A": "XUSB_GAMEPAD_A", "B": "XUSB_GAMEPAD_B", "X": "XUSB_GAMEPAD_X", "Y": "XUSB_GAMEPAD_Y",
    "BACK": "XUSB_GAMEPAD_BACK", "START": "XUSB_GAMEPAD_START",
    "DPAD_UP": "XUSB_GAMEPAD_DPAD_UP", "DPAD_DOWN": "XUSB_GAMEPAD_DPAD_DOWN",
    "DPAD_LEFT": "XUSB_GAMEPAD_DPAD_LEFT", "DPAD_RIGHT": "XUSB_GAMEPAD_DPAD_RIGHT",
    "LB": "XUSB_GAMEPAD_LEFT_SHOULDER", "RB": "XUSB_GAMEPAD_RIGHT_SHOULDER",
}


def clip_action(action) -> tuple[float, float, float]:
    """(steer -1..1, gas 0..1, brake 0..1), clipped."""
    steer, gas, brake = (float(a) for a in action)
    return (max(-1.0, min(1.0, steer)), max(0.0, min(1.0, gas)), max(0.0, min(1.0, brake)))


class Gamepad:
    """The agent's controller. With a guard, a watcher thread neutralises the
    pad the moment Windows' own UI could read it (see tmai.game.input_guard),
    and act() sends nothing until the guard clears."""

    def __init__(self, restart_button: str = "B", pad=None, guard=None, poll_s: float = 0.01):
        if restart_button not in BUTTONS:
            raise ValueError(f"unknown button {restart_button!r}; pick one of {sorted(BUTTONS)}")
        if pad is None:
            import vgamepad

            pad = vgamepad.VX360Gamepad()
            self._restart = getattr(vgamepad.XUSB_BUTTON, BUTTONS[restart_button])
        else:
            self._restart = restart_button
        self.pad = pad
        self.guard = guard
        self.blocked: str | None = None     # why the pad is held neutral right now
        self._interrupted: str | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        if guard is not None and guard.active:
            threading.Thread(target=self._watch, args=(poll_s,), name="input-guard", daemon=True).start()

    def _refresh(self) -> str | None:
        """Re-check the guard; call with the lock held."""
        reason = self.guard.blocked() if self.guard is not None else None
        if reason and not self.blocked:
            self.pad.reset()
            self.pad.update()
        if reason:
            self._interrupted = reason
        self.blocked = reason
        return reason

    def _watch(self, poll_s: float) -> None:
        while not self._closed.wait(poll_s):
            with self._lock:
                self._refresh()

    def take_interruption(self) -> str | None:
        """Why the pad was held neutral since the last call, if it was."""
        with self._lock:
            self._refresh()
            reason, self._interrupted = self._interrupted, None
            return reason

    def act(self, action) -> tuple[float, float, float]:
        steer, gas, brake = clip_action(action)
        with self._lock:
            if self._refresh():
                return 0.0, 0.0, 0.0
            self.pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
            self.pad.right_trigger_float(value_float=gas)
            self.pad.left_trigger_float(value_float=brake)
            self.pad.update()
        return steer, gas, brake

    def release(self) -> None:
        with self._lock:
            self.pad.reset()
            self.pad.update()

    def wait_until_clear(self, timeout_s: float, poll_s: float = 0.05) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                reason = self._refresh()
            if not reason:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"not pressing restart while {reason} (env.input_guard)")
            time.sleep(poll_s)

    def press_restart(self, hold_s: float = 0.1, wait_s: float = 10.0) -> None:
        # The restart button is Back in Windows menus, so never press it there.
        self.wait_until_clear(wait_s)
        self.release()
        with self._lock:
            if self._refresh():
                raise TimeoutError(f"not pressing restart while {self.blocked} (env.input_guard)")
            self.pad.press_button(button=self._restart)
            self.pad.update()
        time.sleep(hold_s)
        with self._lock:
            self.pad.release_button(button=self._restart)
            self.pad.update()

    def close(self) -> None:
        self._closed.set()
        self.release()
