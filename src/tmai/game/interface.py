"""One object for everything the agent touches in the game."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable

from tmai.game.gamepad import Gamepad
from tmai.game.protocol import INT32_MAX, UNKNOWN_RACE_MS, Telemetry
from tmai.game.telemetry import TelemetryClient


FULL_GAS_CONTROLS = (0.0, 1.0, 0.0)   # neutral steer, full gas, no brake
ABORT_POLL_S = 0.1
STALL_TIMEOUT_S = 2.0


class CheckpointRespawn(RuntimeError):
    """The restart button respawned at a checkpoint instead of restarting the run."""


class RestartAborted(RuntimeError):
    """restart() was aborted because the caller requested a stop or pause."""


class GameInterface:
    def __init__(self, cfg: dict, capture: bool = True):
        env = cfg["env"]
        self.telemetry = TelemetryClient(port=env["telemetry_port"]).start()
        self.restart_button = env["restart_button"]
        self.menu_button = env.get("menu_button", "A")
        self.max_age = env.get("max_telemetry_age_s", 0.1)
        self.launch_hold_gas = bool(env.get("launch_hold_gas", True))
        self._last_observed_seq: int | None = None
        self.pad = Gamepad(self.restart_button)
        self.capture = None
        if capture and platform.system() != "Windows":
            self.close()
            raise RuntimeError("the camera (env.observation image) needs Windows window capture; "
                               "on Linux set observation to state in tmai_settings.json, "
                               "or pass --set env.observation=state")
        if capture:
            from tmai.game.capture import open_capture

            self.capture = open_capture(env["frame_size"], env["capture_backend"])

    def observe(self, timeout: float = 2.0):
        """The current game state. Only a packet received within
        env.max_telemetry_age_s counts; if none comes, the controls are
        released before the TimeoutError, so the car never drives blind."""
        try:
            tel = self.telemetry.wait(timeout, after_seq=self._last_observed_seq, max_age=self.max_age)
        except TimeoutError:
            self.pad.release()
            raise
        frame = self.capture.grab() if self.capture else None
        self._last_observed_seq = tel.seq
        return frame, tel

    def act(self, action) -> tuple[float, float, float]:
        return self.pad.act(action)

    def _release_pad(self) -> None:
        if hasattr(self.pad, "release"):
            self.pad.release()

    def _get_before_packet(self, deadline: float, should_abort: Callable[[], bool] | None) -> Telemetry | None:
        latest_fn = getattr(self.telemetry, "latest", None)
        if callable(latest_fn):
            before = latest_fn()
            if before is not None:
                return before
        remaining = max(0.0, deadline - time.monotonic())
        wait_s = min(0.2, remaining)
        if should_abort is not None:
            wait_s = min(ABORT_POLL_S, wait_s)
        try:
            return self.telemetry.wait(wait_s, max_age=self.max_age)
        except TimeoutError:
            if should_abort is not None and should_abort():
                raise RestartAborted("restart aborted")
            try:
                return self.telemetry.wait(0.0, max_age=None)
            except TimeoutError:
                return None

    def restart(self, timeout: float = 15.0, menu_after_s: float = 2.5, settle_s: float = 1.5,
                should_abort: Callable[[], bool] | None = None) -> Telemetry:
        """Restart the run and return the first telemetry once the car may move.

        After a personal best the game shows its end-of-race menu (Improve, Save
        replay, ...), which ignores the restart button and may pause telemetry
        packets. We use the last known packet (even a stale one) as `before`,
        press restart immediately, and alternate restart and menu presses until
        fresh restart telemetry arrives. Gas is only held once fresh countdown
        telemetry is received, and the pad is released on every unsuccessful
        exit (including KeyboardInterrupt, RuntimeError, TimeoutError,
        CheckpointRespawn, and RestartAborted).

        If `should_abort` is given, it is polled during countdown and menu waits;
        returning True releases the pad and raises RestartAborted.

        The race clock can lag the teleport by a few frames (seen under
        Proton), so a clock that still reads the old run only counts as a
        checkpoint respawn once it has kept running for `settle_s`."""
        try:
            if should_abort is not None and should_abort():
                raise RestartAborted("restart aborted")
            deadline = time.monotonic() + timeout
            before = self._get_before_packet(deadline, should_abort)
            if should_abort is not None and should_abort():
                raise RestartAborted("restart aborted")
            self.pad.press_restart()
            pressed = last_press = time.monotonic()
            last_fresh_at = pressed
            restarted = False
            launch_hold_gas = bool(getattr(self, "launch_hold_gas", False))
            gas_held = False
            teleported_at: float | None = None
            teleport_race_ms: int | None = None
            presses = 1
            prev = tel = before
            while (remaining := deadline - time.monotonic()) > 0:
                if should_abort is not None and should_abort():
                    raise RestartAborted("restart aborted")
                poll_s = ABORT_POLL_S if should_abort is not None else STALL_TIMEOUT_S
                wait_s = min(poll_s, remaining)
                after_seq = tel.seq if tel is not None else None
                try:
                    sample = self.telemetry.wait(wait_s, after_seq=after_seq, max_age=self.max_age)
                except TimeoutError:
                    if should_abort is not None and should_abort():
                        raise RestartAborted("restart aborted")
                    if restarted:
                        if (should_abort is None
                                or time.monotonic() - last_fresh_at >= STALL_TIMEOUT_S
                                or deadline - time.monotonic() <= 0):
                            raise
                        continue
                    sample = None   # some menus pause the plugin's packets
                now = time.monotonic()
                if should_abort is not None and should_abort():
                    raise RestartAborted("restart aborted")
                if sample is not None:
                    last_fresh_at = now
                    prev_tel = prev
                    tel = sample
                    if not tel.playing or tel.finished:
                        # A fresh non-playing / menu packet clears any restart latch,
                        # releases held gas, and lets the menu-button logic resume.
                        if restarted:
                            restarted = False
                            if gas_held:
                                self._release_pad()
                                gas_held = False
                        teleported_at = None
                        teleport_race_ms = None
                        if before is None:
                            before = tel
                    else:
                        if before is None:
                            if tel.has_player and tel.cp_count == 0 and tel.race_time_ms < 1000:
                                restarted = True
                            else:
                                before = tel
                        else:
                            restarted |= restart_detected(before, tel)
                        if restarted:
                            teleported_at = None
                            teleport_race_ms = None
                        if (restarted and launch_hold_gas and not gas_held
                                and tel.has_player and tel.race_time_ms < 0):
                            # Hold full gas through the countdown so the car launches on frame 0 of GO.
                            self.pad.act(FULL_GAS_CONTROLS)
                            gas_held = True
                        if (restarted and tel.has_player
                                and tel.cp_count == 0 and 0 <= tel.race_time_ms < 1000):
                            if should_abort is not None and should_abort():
                                raise RestartAborted("restart aborted")
                            if launch_hold_gas:
                                if not gas_held:
                                    self.pad.act(FULL_GAS_CONTROLS)
                            else:
                                self._release_pad()
                            return tel
                        if not restarted and prev_tel is not None and prev_tel.playing:
                            if teleported_at is None and _teleported(prev_tel, tel):
                                teleported_at = now
                                teleport_race_ms = tel.race_time_ms
                            if teleported_at is not None:
                                if teleport_race_ms is not None and tel.race_time_ms < teleport_race_ms:
                                    teleported_at = None
                                    teleport_race_ms = None
                                else:
                                    clock_running = (
                                        (teleport_race_ms is not None and tel.race_time_ms > teleport_race_ms)
                                        or (before is not None and tel.race_time_ms > before.race_time_ms)
                                        or tel.cp_count > 0
                                    )
                                    if (now - teleported_at >= settle_s
                                            and (tel.cp_count > 0 or tel.race_time_ms >= 1000)
                                            and clock_running):
                                        # The clock kept running after the teleport, so this was a checkpoint respawn.
                                        raise CheckpointRespawn(
                                            f"{self.restart_button} respawned at a checkpoint (race clock still at "
                                            f"{tel.race_time_ms / 1000:.1f}s). In game, bind Give up to a free gamepad button "
                                            "(Settings > Controls) and pass it, e.g. --set env.restart_button=BACK"
                                        )
                    prev = tel
                if (not restarted
                        and (sample is None or tel is None or not tel.playing or tel.finished)
                        and now - last_press > menu_after_s):
                    if should_abort is not None and should_abort():
                        raise RestartAborted("restart aborted")
                    if presses % 2:
                        self.pad.press(self.menu_button)
                    else:
                        self.pad.press_restart()
                    presses += 1
                    last_press = time.monotonic()
            raise TimeoutError(
                f"restart did not happen after pressing {self.restart_button} (env.restart_button); "
                f"make sure the game window has focus and that button respawns or gives up in game "
                f"(also tried {self.menu_button}, env.menu_button, for the end-of-race menu)"
            )
        except BaseException:
            self._release_pad()
            raise

    def close(self) -> None:
        self.pad.release()
        self.telemetry.close()
        if self.capture:
            self.capture.close()


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _teleported(prev: Telemetry, now: Telemetry) -> bool:
    """True when `now.pos` jumped farther from `prev.pos` than normal driving
    could explain (without assuming the car is stopped after a respawn)."""
    dist = _dist(prev.pos, now.pos)
    if dist <= 3.0:
        return False
    dt_s = max(0.0, (now.race_time_ms - prev.race_time_ms) / 1000.0)
    avg_speed = 0.5 * (abs(prev.speed) + abs(now.speed))
    if dist > avg_speed * dt_s * 1.5 + 3.0:
        return True
    if any(prev.vel):
        disp = tuple(b - a for a, b in zip(prev.pos, now.pos))
        if sum(d * v for d, v in zip(disp, prev.vel)) < 0:
            return True
    return False


def restart_detected(before: Telemetry, now: Telemetry) -> bool:
    """True once the game shows a fresh run after the restart button was pressed."""
    if not now.has_player or now.finished or now.cp_count or now.race_time_ms >= 1000:
        return False
    old_start = before.clocks[1] if before.clocks else 0
    new_start = now.clocks[1] if now.clocks else 0
    return ((0 < new_start < INT32_MAX and new_start != old_start)
            or ((now.playing or now.race_time_ms < UNKNOWN_RACE_MS)
                and now.race_time_ms < before.race_time_ms - 100))
