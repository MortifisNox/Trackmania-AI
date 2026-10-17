"""One object for everything the agent touches in the game."""

from __future__ import annotations

import time

from tmai.game.gamepad import Gamepad
from tmai.game.protocol import Telemetry
from tmai.game.telemetry import TelemetryClient


class CheckpointRespawn(RuntimeError):
    """The restart button respawned at a checkpoint instead of restarting the run."""


class GameInterface:
    def __init__(self, cfg: dict, capture: bool = True):
        env = cfg["env"]
        self.telemetry = TelemetryClient(port=env["telemetry_port"]).start()
        self.restart_button = env["restart_button"]
        self.menu_button = env.get("menu_button", "A")
        self.pad = Gamepad(self.restart_button)
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

    def restart(self, timeout: float = 15.0, menu_after_s: float = 2.5) -> Telemetry:
        """Restart the run and return the first telemetry once the car may move.

        After a personal best the game shows its end-of-race menu (Improve, Save
        replay, ...), which ignores the restart button. If the run has not
        restarted after `menu_after_s` and the game is not in play, tap
        env.menu_button (A) to pick the menu's default, Improve, which starts the
        run again; restart and menu presses alternate until one works."""
        before = self.telemetry.wait(5.0)
        self.pad.press_restart()
        pressed = last_press = time.monotonic()
        deadline = pressed + timeout
        restarted = False
        presses = 1
        prev = tel = before
        while time.monotonic() < deadline:
            try:
                tel = self.telemetry.wait(2.0, after_seq=tel.seq)
            except TimeoutError:
                if restarted:
                    raise
                continue   # some menus pause the plugin's packets
            if not restarted:
                restarted = restart_detected(before, prev, tel, time.monotonic() - pressed)
                prev = tel
                if not restarted and not tel.playing and time.monotonic() - last_press > menu_after_s:
                    if presses % 2:
                        self.pad.press(self.menu_button)
                    else:
                        self.pad.press_restart()
                    presses += 1
                    last_press = time.monotonic()
                continue
            if tel.race_time_ms > 3000:
                # The clock kept running: the button respawned at a checkpoint
                # instead of restarting the run. Episodes must start from the line.
                raise CheckpointRespawn(
                    f"{self.restart_button} respawned at a checkpoint (race clock still at "
                    f"{tel.race_time_ms / 1000:.1f}s). In game, bind Give up to a free gamepad button "
                    "(Settings > Controls) and pass it, e.g. --set env.restart_button=BACK"
                )
            if tel.race_time_ms >= 0:
                return tel
        raise TimeoutError(
            f"restart did not happen after pressing {self.restart_button} (env.restart_button); "
            f"make sure the game window has focus and that button respawns or gives up in game "
            f"(also tried {self.menu_button}, env.menu_button, for the end-of-race menu)"
        )

    def close(self) -> None:
        self.pad.release()
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
