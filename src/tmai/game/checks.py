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
        f"ground {'y' if t.on_ground else 'n'}  mats {t.wheel_material}  clocks {t.clocks}"
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

    from tmai.game.capture import open_capture
    from tmai.game.lidar import from_config

    lidar = from_config(cfg["env"])
    cap = open_capture(cfg["env"]["frame_size"], cfg["env"]["capture_backend"],
                       cfg["env"]["lidar_width"] if lidar else 0)
    print(f"capture backend: {type(cap).__name__}")
    try:
        x0, y0, x1, y1 = cap.region
        print(f"game window client area: {x1 - x0}x{y1 - y0} at ({x0}, {y0})")
        start = time.perf_counter()
        new_before = cap.new_frames
        while time.perf_counter() - start < 3.0:
            frame = cap.grab()
            time.sleep(0.002)
        fps = (cap.new_frames - new_before) / (time.perf_counter() - start)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "capture_test.png"
        big = cv2.resize(frame, (frame.shape[1] * 6, frame.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), big)
        _check("capture rate", fps >= 20, f"{fps:.0f} new frames/s (need 20; drive or watch a replay meanwhile, unchanged frames are not counted)")
        print(f"saved what the agent sees to {path.resolve()}")
        if lidar is not None:
            wide = cap.grab_wide()
            lengths = lidar.measure(wide)
            scale = max(1, 1024 // wide.shape[1])
            img = cv2.cvtColor(cv2.resize(wide, (wide.shape[1] * scale, wide.shape[0] * scale),
                                          interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            for (x0, y0), (x1, y1) in lidar.endpoints(wide, lengths):
                cv2.line(img, (x0 * scale, y0 * scale), (x1 * scale, y1 * scale), (0, 0, 255), 2)
                cv2.circle(img, (x1 * scale, y1 * scale), 4, (0, 255, 255), -1)
            beams_path = out_dir / "lidar_test.png"
            cv2.imwrite(str(beams_path), img)
            print("lidar beam lengths (left to right, 1 = clear):", " ".join(f"{v:.2f}" for v in lengths))
            if statistics.median(lengths.tolist()) < 0.05:
                print("most beams stop right away: env.lidar_origin is probably on the car or on a dark area")
            print(f"saved the lidar beams to {beams_path.resolve()}: each red beam should stop at a track border")
        return 0 if fps >= 20 else 1
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
            steps, reason, top = 0, None, 0.0
            while reason is None:
                lateness.append(clock.tick())
                frame, t = game.observe()
                ages.append(game.telemetry.age())
                game.act((0.0, 1.0, 0.0))
                reason = monitor.check(t)
                top = max(top, t.speed_kmh)
                steps += 1
            game.pad.release()
            print(f"episode {ep}: {reason} after {steps} steps, {t.race_time_ms / 1000:.2f}s, "
                  f"{t.cp_count} checkpoints, top speed {top:.0f} km/h")
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


def focus_test(cfg: dict, out_dir: Path) -> int:
    """Drives while another window has focus, to see whether the game still
    takes controller input and the capture still sees the game."""
    import cv2

    from tmai.game.interface import GameInterface

    game = GameInterface(cfg, capture=True)
    results = []
    try:
        print(f"capture backend: {type(game.capture).__name__}")
        print("Click on THIS PowerShell window (or any other window) now, so the game is NOT focused.")
        print("You can also drag another window over part of the game.")
        for i in range(8, 0, -1):
            print(f"  starting in {i}...", flush=True)
            time.sleep(1)
        game.restart()
        results.append(_check("restart without focus", True))
        frames_before = game.capture.new_frames
        start = time.monotonic()
        top = 0.0
        while time.monotonic() - start < 4.0:
            game.act((0.0, 1.0, 0.0))
            frame, t = game.observe()
            top = max(top, t.speed_kmh)
            time.sleep(0.05)
        game.pad.release()
        fps = (game.capture.new_frames - frames_before) / 4.0
        results.append(_check("game took input without focus", top > 30, f"top speed {top:.0f} km/h"))
        results.append(_check("capture without focus", fps >= 20, f"{fps:.0f} new frames/s"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "focus_test.png"
        cv2.imwrite(str(path), cv2.resize(frame, (frame.shape[1] * 6, frame.shape[0] * 6),
                                          interpolation=cv2.INTER_NEAREST))
        print(f"saved what the agent saw to {path.resolve()} (it should show the game, not your other window)")
    except TimeoutError as exc:
        results.append(_check("restart without focus", False, str(exc)))
    finally:
        game.close()
    return 0 if all(results) else 1
