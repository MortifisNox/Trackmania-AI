"""Virtual Xbox 360 controller (ViGEmBus via vgamepad)."""

from __future__ import annotations

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
    def __init__(self, restart_button: str = "B", pad=None):
        if restart_button not in BUTTONS:
            raise ValueError(f"unknown button {restart_button!r}; pick one of {sorted(BUTTONS)}")
        if pad is None:
            import vgamepad

            pad = vgamepad.VX360Gamepad()
            self._restart = getattr(vgamepad.XUSB_BUTTON, BUTTONS[restart_button])
        else:
            self._restart = restart_button
        self.pad = pad

    def act(self, action) -> tuple[float, float, float]:
        steer, gas, brake = clip_action(action)
        self.pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.pad.right_trigger_float(value_float=gas)
        self.pad.left_trigger_float(value_float=brake)
        self.pad.update()
        return steer, gas, brake

    def release(self) -> None:
        self.pad.reset()
        self.pad.update()

    def press_restart(self, hold_s: float = 0.1) -> None:
        self.release()
        self.pad.press_button(button=self._restart)
        self.pad.update()
        time.sleep(hold_s)
        self.pad.release_button(button=self._restart)
        self.pad.update()
