"""One object for everything the agent touches in the game."""

from __future__ import annotations

import time

from tmai.game.gamepad import Gamepad
from tmai.game.protocol import Telemetry
from tmai.game.telemetry import TelemetryClient


class GameInterface:
    def __init__(self, cfg: dict, capture: bool = True):
        env = cfg["env"]
        self.telemetry = TelemetryClient(port=env["telemetry_port"]).start()
        self.restart_button = env["restart_button"]
        self.pad = Gamepad(self.restart_button)
        self.capture = None
        if capture:
            from tmai.game.capture import ScreenCapture

            self.capture = ScreenCapture(env["frame_size"])

    def observe(self, timeout: float = 2.0):
        tel = self.telemetry.wait(timeout)
        frame = self.capture.grab() if self.capture else None
        return frame, tel

    def act(self, action) -> tuple[float, float, float]:
        return self.pad.act(action)

    def restart(self, timeout: float = 15.0) -> Telemetry:
        """Restart the run and return the first telemetry once the car may move."""
        before = self.telemetry.wait(5.0)
        self.pad.press_restart()
        deadline = time.monotonic() + timeout
        restarted = False
        tel = before
        while time.monotonic() < deadline:
            tel = self.telemetry.wait(2.0, after_seq=tel.seq)
            if not restarted:
                # A new run shows as the race clock jumping back near zero.
                restarted = tel.playing and tel.race_time_ms < before.race_time_ms - 100
                continue
            if tel.race_time_ms >= 0:
                return tel
        raise TimeoutError(
            f"restart did not happen after pressing {self.restart_button} (env.restart_button); "
            "make sure the game window has focus and that button respawns or gives up in game"
        )

    def close(self) -> None:
        self.pad.release()
        self.telemetry.close()
        if self.capture:
            self.capture.close()
