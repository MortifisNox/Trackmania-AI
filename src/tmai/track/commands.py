"""Phase 2 commands: record demo laps, build the reference line, watch progress live."""

from __future__ import annotations

import struct
import time
from pathlib import Path

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
        ref = ReferenceLine.from_positions(d["pos"], cfg["track"]["reference_spacing_m"],
                                           d["race_time_ms"] / 1000.0)
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

    from tmai.rl.features import uses_camera

    store = _store(cfg)
    game = GameInterface(cfg, capture=uses_camera(cfg["env"]))
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


def _geometry_storage():
    from tmai.game.paths import openplanet_dir
    from tmai.game.plugin import GEOMETRY_STORAGE

    return openplanet_dir() / "PluginStorage" / GEOMETRY_STORAGE


def geometry_check(cfg: dict, export_file: str | None = None) -> int:
    """Finds the map's geometry export (or uses `export_file`), fits its placement
    to the demo laps and saves the result for the lidar, with a picture from above."""
    import hashlib
    from tmai.track.geometry import _parse_export, build_world, latest_export, read_header_name
    from tmai.track.lidar import fit_placement, lap_samples, save_fit, top_down_png

    store = _store(cfg)
    storage = _geometry_storage()
    published = False
    try:
        if export_file:
            path = Path(export_file)
            if not path.exists() and (storage / export_file).exists():
                path = storage / export_file
            if not path.exists():
                print(f"no export {export_file!r} (looked in the current folder and {storage})")
                return 1
        else:
            path = latest_export(storage, store.name)
        if path is None:
            found = sorted(storage.glob("*.tmgeo"), key=lambda f: f.stat().st_mtime, reverse=True)
            if not found:
                print(f"no map geometry in {storage}.")
                print("Run tmai install-plugin, reload TMAI Map Geometry (F3 > Developer > (Re)load plugin),")
                print("then load the map from local play; the plugin exports it when the race starts.")
                return 1
            print(f"no export of a map named {store.name!r} in {storage}; found:")
            for f in found:
                try:
                    name = repr(read_header_name(f))
                except (OSError, ValueError, struct.error):
                    name = "(unreadable)"
                print(f"  {f.name}  map {name}")
            print(f"Load {store.name!r} from local play so the plugin exports it, then run this again.")
            print("If the map's in-game name differs from the track name, pick its export with --export FILE.")
            return 1
        raw = path.read_bytes()
        content_sha256 = hashlib.sha256(raw).hexdigest()
        export = _parse_export(raw, path)
        insts = export.instances
        print(f"{path.name}: map {export.name!r}, {len(export.shapes)} collision shapes, {len(insts)} placed "
              f"({sum(i.kind == 0 for i in insts)} grid blocks, {sum(i.kind == 1 for i in insts)} free blocks, "
              f"{sum(i.kind == 2 for i in insts)} items)")
        demos = store.demo_files()
        if not demos:
            print(f"no demo laps for {store.name!r}; record some with tmai record-demo first")
            return 1
        pos, up = lap_samples(demos)
        print(f"lining up with {len(demos)} demo laps ({len(pos)} on-ground samples)...")
        fit, _ = fit_placement(export, pos, up)
        laps = []
        for f in demos:
            with np.load(f) as d:
                laps.append(d["pos"])
        png = top_down_png(build_world(export, fit.placement), store.geometry_png, laps)
        print(f"\n{fit.on_surface:.1%} of lap samples have the road under them; the car sits "
              f"{fit.ride_median_m:.2f} m above it (90% within {fit.ride_p90_m:.2f} m of that); "
              f"{fit.triangles} triangles")
        if fit.ok():
            save_fit(store.geometry, path, export, fit, content_sha256=content_sha256)
            published = True
            print(f"saved {store.geometry}" + (f" and {png}" if png else ""))
            print("OK: the geometry lines up with your laps. Next: tmai lidar-test")
            return 0
        if png:
            print(f"saved {png} (did not publish {store.geometry})")
        print("NOT OK: the geometry does not line up yet. Send the lines above, "
              "the TMAI geometry lines from F3 > Openplanet > Log and the picture.")
        return 1
    finally:
        if not published:
            try:
                store.geometry.unlink()
            except FileNotFoundError:
                pass


def lidar_test(cfg: dict, seconds: float) -> int:
    """You drive; prints the virtual lidar's distances live."""
    from tmai.game.clock import RealtimeClock
    from tmai.game.telemetry import TelemetryClient
    from tmai.track.lidar import CLEAR, WALL, load_lidar

    store = _store(cfg)
    if not store.geometry.exists():
        print(f"no lined-up geometry for {store.name!r}; run tmai geometry-check first")
        return 1
    lidar = load_lidar(store.geometry)
    angles = lidar.layout.angles_deg
    client = TelemetryClient(port=cfg["env"]["telemetry_port"]).start()
    clock = RealtimeClock(10.0)
    mark = {CLEAR: " ", WALL: "|", 2: "~"}
    print("Drive the map. Columns go from right (-90) to left (+90) of the car;")
    print("each shows metres to the wall (|) or road edge (~), blank when clear to "
          f"{lidar.layout.range_m:.0f} m.")
    print("  ".join(f"{a:+4.0f} " for a in angles) + "   scan")
    costs = []
    log_path = store.root / "lidar_test.csv"
    log = open(log_path, "w", encoding="utf-8")
    log.write("time_s,x,y,z,dx,dy,dz,ux,uy,uz," + ",".join(f"d{a:+.0f},k{a:+.0f}" for a in angles) + "\n")
    try:
        client.wait(10.0)
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            clock.tick()
            t = client.wait(2.0)
            scan = lidar.scan(t.pos, t.dir, t.up)
            costs.append(scan.seconds)
            log.write(",".join(f"{v:.3f}" for v in (time.monotonic() - start, *t.pos, *t.dir, *t.up)) + ","
                      + ",".join(f"{d:.2f},{int(k)}" for d, k in zip(scan.dist, scan.kind)) + "\n")
            cols = [f"{d:4.0f}{mark[int(k)]}" if k != CLEAR else "   - " for d, k in zip(scan.dist, scan.kind)]
            print("  ".join(cols) + f"   {scan.seconds * 1000:4.1f} ms")
        if costs:
            c = np.array(costs) * 1000
            print(f"\nscan time: median {np.median(c):.1f} ms, 95% {np.percentile(c, 95):.1f} ms, "
                  f"max {c.max():.1f} ms")
        print(f"saved every scan to {log_path}")
        return 0
    except TimeoutError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        log.close()
        client.close()
