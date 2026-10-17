"""One object for everything the agent touches in the game."""

from __future__ import annotations

import time

from tmai.game.gamepad import Gamepad
from tmai.game.input_guard import InputGuard
from tmai.game.protocol import Telemetry
from tmai.game.telemetry import TelemetryClient


class GameInterface:
    def __init__(self, cfg: dict, capture: bool = True):
        env = cfg["env"]
        self.telemetry = TelemetryClient(port=env["telemetry_port"]).start()
        self.restart_button = env["restart_button"]
        self.pad = Gamepad(self.restart_button, guard=InputGuard(env["input_guard"]))
        self.capture = None
        if capture:
            from tmai.game.capture import open_capture

            self.capture = open_capture(env["frame_size"], env["capture_backend"])

    def observe(self, timeout: float = 2.0):
        tel = self.telemetry.wait(timeout)
        frame = self.capture.grab() if self.capture else None
        return frame, tel

    def act(self, action) -> tuple[float, float, float]:
        return self.pad.act(action)

    def take_input_interruption(self) -> str | None:
        """Why the pad was held neutral since the last call (Windows UI in
        front, Alt held, ...), or None if every action reached the game."""
        return self.pad.take_interruption()

    def restart(self, timeout: float = 15.0) -> Telemetry:
        """Restart the run and return the first telemetry once the car may move."""
        before = self.telemetry.wait(5.0)
        self.pad.press_restart()
        pressed = time.monotonic()
        deadline = pressed + timeout
        restarted = False
        prev = tel = before
        while time.monotonic() < deadline:
            tel = self.telemetry.wait(2.0, after_seq=tel.seq)
            if not restarted:
                restarted = restart_detected(before, prev, tel, time.monotonic() - pressed)
                prev = tel
                continue
            if tel.race_time_ms >= 0:
                return tel
        raise TimeoutError(
            f"restart did not happen after pressing {self.restart_button} (env.restart_button); "
            "make sure the game window has focus and that button respawns or gives up in game"
        )

    def close(self) -> None:
        self.pad.close()
        self.telemetry.close()
        if self.capture:
            self.capture.close()


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def restart_detected(before: Telemetry, prev: Telemetry, now: Telemetry, waited_s: float) -> bool:
    """True once the game shows a fresh run after the restart button was pressed."""
    if not now.playing:
        return False
    if now.race_time_ms < before.race_time_ms - 100:
        return True  # race clock jumped back
    if _dist(prev.pos, now.pos) > 3.0 and now.speed_kmh < 5:
        return True  # teleported back to the start (or a checkpoint)
    # Already parked at the start: nothing visibly changes, but the car is ready.
    return waited_s > 2.0 and now.speed_kmh < 1 and _dist(before.pos, now.pos) < 1.0
