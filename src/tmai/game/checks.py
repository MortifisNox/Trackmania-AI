"""Hands-on checks for phase 1, run on the gaming laptop with a map loaded."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from tmai.game.clock import RealtimeClock
from tmai.game.episode import EpisodeMonitor


def _line(t) -> str:
    return (
        f"seq {t.seq:>7}  {'PLAY' if t.playing else 'FIN ' if t.finished else '----'}  "
        f"time {t.race_time_ms / 1000:7.2f}s  cps {t.cp_count:>2}  "
        f"{t.speed_kmh:6.1f} km/h  gear {t.gear}  rpm {t.rpm:6.0f}  "
        f"steer {t.steer:+.2f} gas {t.gas:.2f} brake {t.brake:.2f}  "
        f"ground {'y' if t.on_ground else 'n'}  mats {t.wheel_material}"
    )


def watch_telemetry(cfg: dict, seconds: float) -> int:
    from tmai.game.telemetry import TelemetryClient

    client = TelemetryClient(port=cfg["env"]["telemetry_port"]).start()
    try:
        first = client.wait(10.0)
        start, last_print = time.monotonic(), 0.0
        while time.monotonic() - start < seconds:
            t = client.wait(2.0)
            now = time.monotonic()
            if now - last_print >= 0.2:
                print(_line(t))
                last_print = now
            time.sleep(0.005)
        t = client.latest()
        rate = (t.seq - first.seq) / max(1e-6, time.monotonic() - start)
        print(f"\nplugin sends {rate:.0f} packets/s (one per game frame)")
        return 0
    except TimeoutError as exc:
        print(f"FAIL: {exc}")
        print("Is a map loaded, and is the TMAI Telemetry plugin running (F3 > Openplanet > Log)?")
        return 1
    finally:
        client.close()


def _countdown(seconds: int = 5) -> None:
    print("Click on the Trackmania window now so it has focus.")
    for i in range(seconds, 0, -1):
        print(f"  starting in {i}...", flush=True)
        time.sleep(1)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    return ok


def gamepad_test(cfg: dict) -> int:
    from tmai.game.interface import GameInterface

    game = GameInterface(cfg, capture=False)
    results = []
    try:
        _countdown()
        game.restart()
        results.append(_check("restart", True))
        for steer in (-1.0, 1.0):
            game.act((steer, 1.0, 0.0))
            time.sleep(0.6)
            t = game.telemetry.wait(2.0)
            results.append(_check(f"steer {steer:+.0f}", abs(t.steer - steer) < 0.2, f"game saw {t.steer:+.2f}"))
            results.append(_check("gas", t.gas > 0.8, f"game saw {t.gas:.2f}"))
        game.act((0.0, 0.0, 1.0))
        time.sleep(0.6)
        t = game.telemetry.wait(2.0)
        results.append(_check("brake", t.brake > 0.5 or t.speed_kmh < 5, f"game saw {t.brake:.2f}"))
        game.pad.release()
        game.restart()
        results.append(_check("second restart", True))
    except TimeoutError as exc:
        results.append(_check("restart", False, str(exc)))
    finally:
        game.close()
    return 0 if all(results) else 1


def capture_test(cfg: dict, out_dir: Path, frames: int = 120) -> int:
    import cv2

    from tmai.game.capture import ScreenCapture

    cap = ScreenCapture(cfg["env"]["frame_size"])
    try:
        x0, y0, x1, y1 = cap.region
        print(f"game window client area: {x1 - x0}x{y1 - y0} at ({x0}, {y0})")
        start = time.perf_counter()
        for _ in range(frames):
            frame = cap.grab()
        fps = frames / (time.perf_counter() - start)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "capture_test.png"
        big = cv2.resize(frame, (frame.shape[1] * 6, frame.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), big)
        _check("capture rate", fps >= 40, f"{fps:.0f} frames/s (need 20, want 40+ for headroom)")
        print(f"saved what the agent sees to {path.resolve()}")
        return 0 if fps >= 40 else 1
    finally:
        cap.close()


def drive_test(cfg: dict, episodes: int) -> int:
    """Scripted driver: full throttle, straight ahead, at the real control rate."""
    from tmai.game.interface import GameInterface

    env = cfg["env"]
    game = GameInterface(cfg, capture=True)
    clock = RealtimeClock(env["step_hz"])
    monitor = EpisodeMonitor(env["episode_timeout_s"], env["stuck_timeout_s"], env["stuck_speed_kmh"])
    lateness, ages, ok = [], [], True
    try:
        _countdown()
        for ep in range(1, episodes + 1):
            game.restart()
            monitor.reset()
            clock.reset()
            steps, reason = 0, None
            while reason is None:
                lateness.append(clock.tick())
                frame, t = game.observe()
                ages.append(game.telemetry.age())
                game.act((0.0, 1.0, 0.0))
                reason = monitor.check(t)
                steps += 1
            game.pad.release()
            print(f"episode {ep}: {reason} after {steps} steps, {t.race_time_ms / 1000:.2f}s, "
                  f"{t.cp_count} checkpoints, top speed so far {t.speed_kmh:.0f} km/h")
    except TimeoutError as exc:
        ok = _check("drive loop", False, str(exc))
    finally:
        game.close()
    if lateness:
        late_ms = sorted(x * 1000 for x in lateness)
        p99 = late_ms[int(0.99 * (len(late_ms) - 1))]
        print(f"\nstep lateness: median {statistics.median(late_ms):.1f} ms, p99 {p99:.1f} ms, "
              f"overruns {clock.overruns}/{clock.steps}")
        print(f"telemetry age: median {statistics.median(ages) * 1000:.1f} ms")
        ok &= _check("timing", p99 < 25, "p99 lateness must stay under half a step (25 ms)")
    return 0 if ok else 1
