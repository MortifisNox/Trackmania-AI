"""Phase 2 commands: record demo laps, build the reference line, watch progress live."""

from __future__ import annotations

import time

import numpy as np

from tmai.track.reference import ReferenceLine
from tmai.track.splits import SplitTracker
from tmai.track.store import TrackStore


def _store(cfg: dict) -> TrackStore:
    return TrackStore(cfg["run"]["output_dir"], cfg["run"]["track"])


def build_reference(cfg: dict) -> int:
    store = _store(cfg)
    demo = store.best_demo()
    with np.load(demo) as d:
        ref = ReferenceLine.from_positions(d["pos"], cfg["track"]["reference_spacing_m"])
        cp_times = [int(x) for x in d["cp_times"]]
        lap_ms = int(d["lap_time_ms"])
    ref.save(store.reference)
    SplitTracker(cp_times).save(store.splits)
    print(f"reference line for {store.name!r}: {ref.length:.0f} m from the {lap_ms / 1000:.3f}s lap "
          f"({demo.name}), {len(cp_times)} checkpoint splits")
    return 0


def record_demo(cfg: dict, laps: int) -> int:
    from tmai.game.interface import GameInterface
    from tmai.track.demo import record_laps

    store = _store(cfg)
    game = GameInterface(cfg, capture=True)
    try:
        record_laps(game, store.demos, laps, cfg["env"]["step_hz"])
    except KeyboardInterrupt:
        print("stopped")
    finally:
        game.close()
    if store.demo_files():
        return build_reference(cfg)
    return 1


def progress_test(cfg: dict, seconds: float) -> int:
    from tmai.game.clock import RealtimeClock
    from tmai.game.telemetry import TelemetryClient
    from tmai.track.progress import ProgressTracker

    store = _store(cfg)
    if not store.reference.exists():
        print(f"no reference line for {store.name!r}; run tmai record-demo first")
        return 1
    ref = ReferenceLine.load(store.reference)
    tracker = ProgressTracker(ref, cfg, SplitTracker.load(store.splits))
    client = TelemetryClient(port=cfg["env"]["telemetry_port"]).start()
    clock = RealtimeClock(cfg["env"]["step_hz"])
    try:
        t = client.wait(10.0)
        tracker.reset(t)
        total, last_print, prev_time = 0.0, 0.0, t.race_time_ms
        start = time.monotonic()
        print("Drive the map yourself; restart whenever you like.")
        while time.monotonic() - start < seconds:
            clock.tick()
            t = client.wait(2.0)
            if t.race_time_ms < prev_time - 500:
                tracker.reset(t)
                total = 0.0
                print("-- restart --")
            prev_time = t.race_time_ms
            ended = tracker.ended
            info = tracker.step(t)
            total += info.reward
            now = time.monotonic()
            if info.split_delta_ms is not None:
                print(f"   checkpoint {t.cp_count}: {info.split_delta_ms / 1000:+.3f}s vs best")
            if info.done and not ended:
                print(f"   episode end: {info.done} (total reward {total:.1f})")
            if now - last_print >= 0.5 and not ended:
                ahead = info.lookahead[:3]
                print(f"time {t.race_time_ms / 1000:6.2f}s  progress {info.progress * 100:5.1f}%  "
                      f"({info.progress_m:6.0f}/{ref.length:.0f} m)  off-line {info.lateral_m:4.1f} m  "
                      f"reward/step {info.reward:+.3f}  total {total:7.1f}  "
                      f"next point fwd {ahead[0]:5.1f} m left {ahead[1]:+5.1f} m")
                last_print = now
        return 0
    except TimeoutError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        client.close()
