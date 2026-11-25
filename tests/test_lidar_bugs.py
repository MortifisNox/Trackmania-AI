"""Regression tests for virtual-LiDAR and map-geometry bugs."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from test_geometry import _geometry_cfg, quad, straight_road_map, wall, write_export
from tmai.track import geometry as g
from tmai.track.lidar import (
    CLEAR,
    DROP,
    WALL,
    Lidar,
    LidarLayout,
    fit_placement,
    grade,
    load_lidar,
    save_fit,
)


def _write_demo(path, pos, up):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = np.full(len(pos), 8, dtype=np.int64)
    np.savez(path, pos=np.asarray(pos, dtype=np.float32), up=np.asarray(up, dtype=np.float32), flags=flags)


# --- Bug 1: failed geometry-check still publishes geometry.json, and plane above car passes ---

def test_fit_rejects_plane_above_car_and_far_below():
    # Road plane at y = 10.0; car sits at y = 9.0 (plane is 1.0 m ABOVE the car).
    verts, tris = quad(0, 32, 10.0, 0, 32)
    world_above = g.WorldMesh(
        np.asarray([[verts[a], verts[b], verts[c]] for a, b, c in tris], dtype=np.float64),
        np.full(len(tris), 16, dtype=np.uint16),
        np.zeros(len(tris), dtype=np.uint16),
        np.zeros(len(tris), dtype=np.int32),
        [],
    )
    pos = np.column_stack([np.full(40, 16.0), np.full(40, 9.0), np.linspace(2.0, 30.0, 40)])
    up = np.tile([0.0, 1.0, 0.0], (40, 1))
    fit = grade(world_above, pos, up, g.Placement())
    assert not fit.ok(), f"plane 1 m above car should fail fit.ok(), got {fit}"
    # Even if grade is invoked with a large `above`, negative ride height must not pass ok().
    fit_wide = grade(world_above, pos, up, g.Placement(), above=2.0)
    assert not fit_wide.ok(), f"negative ride height should fail fit.ok(), got {fit_wide}"

    # Road plane 2.0 m BELOW the car (y = 7.0 vs car y = 9.0) must also fail.
    verts_low, tris_low = quad(0, 32, 7.0, 0, 32)
    world_low = g.WorldMesh(
        np.asarray([[verts_low[a], verts_low[b], verts_low[c]] for a, b, c in tris_low], dtype=np.float64),
        np.full(len(tris_low), 16, dtype=np.uint16),
        np.zeros(len(tris_low), dtype=np.uint16),
        np.zeros(len(tris_low), dtype=np.int32),
        [],
    )
    assert not grade(world_low, pos, up, g.Placement()).ok()


def test_geometry_check_does_not_publish_on_failure_and_removes_stale_fit(tmp_path, monkeypatch):
    commands, storage, cfg = _geometry_cfg(tmp_path, monkeypatch)
    # Place a single road block far away from the demo laps so fit fails.
    write_export(
        storage / "bad.tmgeo",
        [(*quad(0, 32, 0.0, 0, 32), 16)],
        blocks=[("road", (0, 8, 0), 0, (1, 1, 1), [(0, [])])],
        name="T",
    )
    store = commands._store(cfg)
    z = np.linspace(325.0, 380.0, 40)
    pos = np.column_stack([np.full_like(z, 336.0), np.full_like(z, 8.4), z])
    up = np.tile([0.0, 1.0, 0.0], (len(z), 1))
    _write_demo(store.demos / "lap1.npz", pos, up)

    # Plant a stale geometry.json from an earlier run.
    store.geometry.parent.mkdir(parents=True, exist_ok=True)
    store.geometry.write_text(json.dumps({"export": "stale.tmgeo", "placement": {"yoff": -64.0}}))

    rc = commands.geometry_check(cfg)
    assert rc == 1
    assert not store.geometry.exists(), "failed geometry-check must not leave a training-usable geometry.json"


def test_geometry_check_publishes_atomically_on_success(tmp_path, monkeypatch):
    commands, storage, cfg = _geometry_cfg(tmp_path, monkeypatch)
    export_path = straight_road_map(storage, rows=4)
    # Rewrite with map name "T" so latest_export finds it.
    ex = g.read_export(export_path)
    shapes = [(s.verts.tolist(), s.tris.tolist(), int(s.ids[0, 0])) for s in ex.shapes]
    blocks = [(i.name, i.coord, i.dir, i.size, [(p.shape, []) for p in i.parts]) for i in ex.instances]
    write_export(storage / "good.tmgeo", shapes, blocks, name="T")
    export_path.unlink()

    store = commands._store(cfg)
    z = np.linspace(325.0, 440.0, 40)
    pos = np.column_stack([np.full_like(z, 336.0), np.full_like(z, 8.4), z])
    up = np.tile([0.0, 1.0, 0.0], (len(z), 1))
    _write_demo(store.demos / "lap1.npz", pos, up)

    rc = commands.geometry_check(cfg)
    assert rc == 0
    assert store.geometry.exists()
    data = json.loads(store.geometry.read_text())
    assert data["map"] == "T" and data["on_surface"] >= 0.95
    assert isinstance(data.get("export_sha256"), str) and len(data["export_sha256"]) == 64
    assert not list(store.root.glob("*.tmp"))
    # Validated fit loads cleanly through load_lidar().
    loaded = load_lidar(store.geometry)
    assert len(loaded.world.tris) > 0


# --- Bug 2: wall behind road-like ramp, and deck edge over disconnected lower floor ---

def test_wall_found_behind_road_like_ramp(tmp_path):
    # Flat road at y = 0 (world y = 8 with yoff=-64), gentle ramp in front of a vertical wall at 10 m.
    # Car at (336, 8.4, 330), wall ray at y = 9.4 (local y = 1.4), z starts at 330 (local z = 10).
    # Ramp rises from local y = 0 at z = 12 to y = 3 at z = 24 (slope 0.25, normal.y ~ 0.97 > 0.6).
    # Ray at local y = 1.4 crosses ramp at z = 17.6 (7.6 m ahead), then hits wall at z = 20.0 (10.0 m ahead).
    road = quad(0, 32, 0.0, 0, 32)
    ramp_v = [(0, 0.0, 12.0), (32, 0.0, 12.0), (32, 3.0, 24.0), (0, 3.0, 24.0)]
    ramp_t = [(0, 1, 2), (0, 2, 3)]
    wall_v = [(0, 0.0, 20.0), (32, 0.0, 20.0), (32, 6.0, 20.0), (0, 6.0, 20.0)]
    wall_t = [(0, 1, 2), (0, 2, 3)]
    shapes = [(*road, 16), (ramp_v, ramp_t, 16), (wall_v, wall_t, 16)]
    blocks = [("b0", (10, 9, 10), 0, (1, 1, 1), [(0, []), (1, []), (2, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "rw.tmgeo", shapes, blocks)))
    lidar = Lidar(world, LidarLayout(angles_deg=(0,), range_m=60.0))
    scan = lidar.scan(pos=(336.0, 8.4, 330.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert scan.kind[0] == WALL
    assert scan.dist[0] == pytest.approx(10.0, abs=0.05)


def test_deck_edge_reports_drop_despite_disconnected_floor_below(tmp_path):
    # Deck at local y = 0 (world y = 8) from local z = 0..15 (world z = 320..335).
    # Car at z = 330: deck ends 5 m ahead (between 3 m and 6 m probes).
    # Disconnected lower floor at local y = -2.0 (world y = 6) from z = 15..96.
    deck = quad(0, 32, 0.0, 0, 15)
    lower = quad(0, 32, -2.0, 15, 96)
    shapes = [(*deck, 16), (*lower, 16)]
    blocks = [("b0", (10, 9, 10), 0, (1, 1, 1), [(0, []), (1, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "deck.tmgeo", shapes, blocks)))
    lidar = Lidar(world, LidarLayout(angles_deg=(0,), range_m=60.0))
    scan = lidar.scan(pos=(336.0, 8.4, 330.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert scan.kind[0] == DROP
    assert scan.dist[0] == pytest.approx(6.0, abs=0.01)
    assert np.all(np.isnan(scan.height[0]))


# --- Bug 3: large ray batch (> 32,768 rays) loses ray hits ---

def test_ray_caster_handles_over_32768_rays():
    tris = np.array([[[5.0, -5.0, -5.0], [5.0, 5.0, -5.0], [5.0, 0.0, 5.0]]], dtype=np.float64)
    rc = g.RayCaster(tris, cell=4.0)
    n = 33_000
    origins = np.zeros((n, 3), dtype=np.float64)
    dirs = np.tile([1.0, 0.0, 0.0], (n, 1))
    dist, hit = rc.cast(origins, dirs, 10.0)
    assert np.all(hit == 0), f"missed {(hit != 0).sum()} of {n} rays; last ray dist={dist[-1]}"
    assert np.allclose(dist, 5.0)
    # Also verify _candidates directly does not produce negative or missing ray IDs.
    r, t = rc._candidates(origins, dirs, np.full(n, 10.0))
    assert r.min() == 0 and r.max() == n - 1 and len(np.unique(r)) == n


# --- Bug 4: bad-pose features leak previous lap scan ---

def test_bad_pose_returns_no_info_instead_of_stale_scan(tmp_path):
    # Wall 5 m ahead of the car.
    road = quad(0, 32, 0.0, 0, 32)
    wall_v = [(0, 0.0, 15.0), (32, 0.0, 15.0), (32, 5.0, 15.0), (0, 5.0, 15.0)]
    shapes = [(*road, 16), (wall_v, [(0, 1, 2), (0, 2, 3)], 16)]
    blocks = [("b0", (10, 9, 10), 0, (1, 1, 1), [(0, []), (1, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "leak.tmgeo", shapes, blocks)))
    layout = LidarLayout(angles_deg=(0,), range_m=60.0)
    lidar = Lidar(world, layout)

    good = SimpleNamespace(pos=np.array([336.0, 8.4, 330.0]), dir=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    f_good = lidar.features(good)
    assert f_good[0] == pytest.approx(5.0 / 60.0, abs=0.01) and f_good[1] == 1.0

    # Now simulate start of a new lap with zero orientation from the plugin.
    bad_zero = SimpleNamespace(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 0.0]))
    f_bad = lidar.features(bad_zero)
    expected_empty = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(f_bad, expected_empty), f"leaked previous scan: {f_bad}"

    # Zero `up` with non-zero `dir` must also return no-info rather than degenerate basis.
    bad_up = SimpleNamespace(pos=np.array([336.0, 8.4, 330.0]), dir=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 0.0, 0.0]))
    assert np.allclose(lidar.features(bad_up), expected_empty)

    # reset() must exist and be callable at episode start.
    lidar.reset()
    assert np.allclose(lidar.features(bad_zero), expected_empty)


# --- Bug 5: deduplication erases distinct variants/mobils/ground and empty blocks shadow non-empty ones ---

def test_unique_instances_preserves_distinct_variants_and_skips_empty():
    part_a = [g.Part(0, False, [])]
    part_b = [g.Part(1, False, [])]

    # Empty block in SRC_BLOCKS must NOT shadow a non-empty block in SRC_BAKED at the same pose.
    empty_blk = g.Instance(g.SRC_BLOCKS, g.KIND_GRID_BLOCK, "Road", [], coord=(1, 2, 3), dir=0, ground=False)
    baked_blk = g.Instance(g.SRC_BAKED, g.KIND_GRID_BLOCK, "Road", part_a, coord=(1, 2, 3), dir=0, ground=False)
    out = g.unique_instances([empty_blk, baked_blk])
    assert len(out) == 1 and out[0] is baked_blk

    # Distinct ground state at same name/coord/dir must both survive.
    air_blk = g.Instance(g.SRC_BLOCKS, g.KIND_GRID_BLOCK, "Road", part_a, coord=(1, 2, 3), dir=0, ground=False)
    gnd_blk = g.Instance(g.SRC_BAKED, g.KIND_GRID_BLOCK, "Road", part_b, coord=(1, 2, 3), dir=0, ground=True)
    assert len(g.unique_instances([air_blk, gnd_blk])) == 2

    # Distinct variant_index or mobil_index at same name/coord/dir/ground must both survive.
    v0 = g.Instance(g.SRC_BLOCKS, g.KIND_GRID_BLOCK, "Road", part_a, coord=(1, 2, 3), dir=0, ground=False,
                    variant_index=0, mobil_variant_index=0, mobil_index=0)
    v1 = g.Instance(g.SRC_BAKED, g.KIND_GRID_BLOCK, "Road", part_b, coord=(1, 2, 3), dir=0, ground=False,
                    variant_index=1, mobil_variant_index=0, mobil_index=0)
    m1 = g.Instance(g.SRC_BAKED, g.KIND_GRID_BLOCK, "Road", part_b, coord=(1, 2, 3), dir=0, ground=False,
                    variant_index=0, mobil_variant_index=0, mobil_index=1)
    assert len(g.unique_instances([v0, v1, m1])) == 3

    # Exact duplicate across Blocks and BakedBlocks still collapses to one.
    dup = g.Instance(g.SRC_BAKED, g.KIND_GRID_BLOCK, "Road", part_a, coord=(1, 2, 3), dir=0, ground=False,
                     variant_index=0, mobil_variant_index=0, mobil_index=0)
    assert len(g.unique_instances([v0, dup])) == 1

    # Free blocks follow the same rules: distinct variant/mobil/ground survive and empty does not shadow.
    f_empty = g.Instance(g.SRC_BLOCKS, g.KIND_FREE_BLOCK, "Free", [], pos=np.array([1.0, 2.0, 3.0]),
                         ypr=np.array([0.1, 0.0, 0.0]), ground=False, variant_index=2, mobil_index=1)
    f_real = g.Instance(g.SRC_BAKED, g.KIND_FREE_BLOCK, "Free", part_a, pos=np.array([1.0, 2.0, 3.0]),
                        ypr=np.array([0.1, 0.0, 0.0]), ground=False, variant_index=2, mobil_index=1)
    f_other = g.Instance(g.SRC_BAKED, g.KIND_FREE_BLOCK, "Free", part_b, pos=np.array([1.0, 2.0, 3.0]),
                         ypr=np.array([0.1, 0.0, 0.0]), ground=False, variant_index=2, mobil_index=2)
    assert len(g.unique_instances([f_empty, f_real, f_other])) == 2


def test_read_export_supports_v1_and_v2_variant_mobil_indices(tmp_path):
    from test_geometry import _parts, _str

    verts, tris = quad(0, 32, 0.0, 0, 32)
    shape_bytes = (
        struct.pack("<I", 1)
        + _str("shape") + struct.pack("<I", 0)
        + struct.pack("<I", len(verts)) + struct.pack(f"<{3 * len(verts)}f", *np.ravel(verts))
        + struct.pack("<I", len(tris)) + struct.pack(f"<{3 * len(tris)}I", *np.ravel(tris))
        + struct.pack(f"<{2 * len(tris)}H", *([16, 0] * len(tris)))
    )
    header_tail = _str("uid1") + _str("Test") + _str("Deco") + struct.pack("<IIII", 48, 40, 48, 8) + shape_bytes

    # v1 file (no variant/mobil fields) still loads cleanly with indices defaulting to 0.
    v1_body = (
        struct.pack("<I", 1)
        + struct.pack("<BB", g.SRC_BLOCKS, g.KIND_GRID_BLOCK) + _str("Road")
        + struct.pack("<iiiBB", 1, 2, 3, 0, 1) + struct.pack("<III", 1, 1, 1)
        + struct.pack("<6f", *([0.0] * 6)) + _parts([(0, [])])
    )
    p1 = tmp_path / "v1.tmgeo"
    p1.write_bytes(struct.pack("<IH", g.MAGIC, 1) + header_tail + v1_body)
    ex1 = g.read_export(p1)
    assert len(ex1.instances) == 1
    assert ex1.instances[0].ground is True
    assert (ex1.instances[0].variant_index, ex1.instances[0].mobil_variant_index, ex1.instances[0].mobil_index) == (0, 0, 0)

    # v2 file preserves variant_index, mobil_variant_index, and mobil_index.
    v2_body = (
        struct.pack("<I", 1)
        + struct.pack("<BB", g.SRC_BLOCKS, g.KIND_GRID_BLOCK) + _str("Road")
        + struct.pack("<iiiBB", 1, 2, 3, 0, 1) + struct.pack("<III", 1, 1, 1)
        + struct.pack("<6f", *([0.0] * 6)) + struct.pack("<III", 3, 2, 5) + _parts([(0, [])])
    )
    p2 = tmp_path / "v2.tmgeo"
    p2.write_bytes(struct.pack("<IH", g.MAGIC, 2) + header_tail + v2_body)
    ex2 = g.read_export(p2)
    assert (ex2.instances[0].variant_index, ex2.instances[0].mobil_variant_index, ex2.instances[0].mobil_index) == (3, 2, 5)


def test_deck_edge_at_14m_with_lower_floor_and_negative_pos_nan(tmp_path):
    # Deck ends 14 m ahead (between 12 m and 16 m probes, dd = 4 m), floor 2 m below.
    deck = quad(0, 32, 0.0, 0, 24)
    lower = quad(0, 32, -2.0, 24, 96)
    shapes = [(*deck, 16), (*lower, 16)]
    blocks = [("b0", (10, 9, 10), 0, (1, 1, 1), [(0, []), (1, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "deck14.tmgeo", shapes, blocks)))
    lidar = Lidar(world, LidarLayout(angles_deg=(0,), range_m=60.0))
    scan = lidar.scan(pos=(336.0, 8.4, 330.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert scan.kind[0] == DROP and scan.dist[0] == pytest.approx(16.0, abs=0.01)

    # Non-finite pos in features() returns no-info scan instead of crashing or leaking.
    bad_nan = SimpleNamespace(pos=np.array([np.nan, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]))
    assert np.allclose(lidar.features(bad_nan), np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))


# --- Review finding 1: fit content hash + stale geometry.json cleanup on all failing paths ---

def test_load_lidar_rejects_modified_export_and_legacy_fit_without_hash(tmp_path):
    road_verts = np.array([[-4, 0, -4], [4, 0, -4], [4, 0, 65], [-4, 0, 65]], dtype=np.float32)
    tri_idx = [(0, 2, 1), (0, 3, 2)]
    blocks = [("road", (0, 8, 0), 0, (1, 1, 1), [(0, [])])]
    export_path = tmp_path / "uid1.tmgeo"
    fit_path = tmp_path / "geometry.json"

    write_export(export_path, [(road_verts.tolist(), tri_idx, 16)], blocks, name="T")
    ex = g.read_export(export_path)
    pos = np.column_stack([np.zeros(20), np.full(20, 0.04), np.linspace(0.0, 60.0, 20)])
    up = np.tile([0.0, 1.0, 0.0], (20, 1))
    placement = g.Placement()
    fit = grade(g.build_world(ex, placement), pos, up, placement)
    assert fit.ok()
    save_fit(fit_path, export_path, ex, fit)

    # Matching export loads cleanly.
    loaded = load_lidar(fit_path)
    assert np.allclose(np.unique(loaded.world.tris[..., 1]), [0.0])

    # Overwriting the same path + UID with geometry 8 m lower must be refused.
    write_export(export_path, [((road_verts + [0, -8, 0]).tolist(), tri_idx, 16)], blocks, name="T")
    with pytest.raises(ValueError, match="re-run tmai geometry-check"):
        load_lidar(fit_path)

    # Old fit without export_sha256 (or with an empty hash) must also be refused.
    legacy = json.loads(fit_path.read_text())
    legacy.pop("export_sha256")
    fit_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="re-run tmai geometry-check"):
        load_lidar(fit_path)

    legacy["export_sha256"] = ""
    fit_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="re-run tmai geometry-check"):
        load_lidar(fit_path)


def test_geometry_check_removes_stale_fit_on_all_early_exits(tmp_path, monkeypatch):
    commands, storage, cfg = _geometry_cfg(tmp_path, monkeypatch)
    store = commands._store(cfg)

    def plant_stale():
        store.geometry.parent.mkdir(parents=True, exist_ok=True)
        store.geometry.write_text(json.dumps({"export": "stale.tmgeo", "placement": {}}))
        assert store.geometry.exists()

    # 1. Explicit --export missing.tmgeo
    plant_stale()
    assert commands.geometry_check(cfg, "missing.tmgeo") == 1
    assert not store.geometry.exists()

    # 2. Empty storage directory (no .tmgeo files)
    plant_stale()
    assert commands.geometry_check(cfg) == 1
    assert not store.geometry.exists()

    # 3. Only an export for a different map
    write_export(storage / "other.tmgeo", [(*quad(0, 32, 0.0, 0, 32), 16)], name="OtherMap")
    plant_stale()
    assert commands.geometry_check(cfg) == 1
    assert not store.geometry.exists()

    # 4. Matching map export exists, but no demo laps recorded
    write_export(storage / "match.tmgeo", [(*quad(0, 32, 0.0, 0, 32), 16)], name="T")
    plant_stale()
    assert commands.geometry_check(cfg) == 1
    assert not store.geometry.exists()

    # 5. Corrupt export raises while reading
    corrupt = storage / "corrupt.tmgeo"
    corrupt.write_bytes(b"NOT_A_TMGEO_FILE")
    plant_stale()
    with pytest.raises(ValueError):
        commands.geometry_check(cfg, str(corrupt))
    assert not store.geometry.exists()


# --- Review finding 2: wall filter must not discard sloped or floating obstacles ---

def _mesh(*parts):
    tris = np.concatenate(parts)
    ids = np.zeros(len(tris), dtype=np.int32)
    return g.WorldMesh(tris, ids.astype(np.uint16), ids.astype(np.uint16), ids, [])


def _quad_pts(a, b, c, d):
    return np.array([[a, b, c], [a, c, d]], dtype=np.float64)


def _slab(y, z0, z1):
    return _quad_pts((-4, y, z0), (4, y, z0), (4, y, z1), (-4, y, z1))


def test_wall_detects_sloped_and_floating_road_like_obstacles():
    road = _slab(0.0, -4.0, 65.0)
    # 50-deg panel (|normal.y| ~ 0.64 > 0.6) floating above the road at z = 6.2..7.4.
    barrier = _quad_pts((-4, 1.05, 6.2), (4, 1.05, 6.2), (4, 2.49, 7.4), (-4, 2.49, 7.4))
    world = _mesh(road, barrier)
    scan = Lidar(world, LidarLayout(angles_deg=(0.0,))).scan(
        pos=(0.0, 0.2, 0.0), forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)
    )
    assert scan.kind[0] == WALL
    assert scan.dist[0] == pytest.approx(6.325, abs=0.02)

    # Also verify a far sloped barrier at ~34 m (between 30 m and 38 m probes) is caught as WALL.
    far_barrier = _quad_pts((-4, 1.05, 34.0), (4, 1.05, 34.0), (4, 2.49, 35.2), (-4, 2.49, 35.2))
    scan_far = Lidar(_mesh(road, far_barrier), LidarLayout(angles_deg=(0.0,))).scan(
        pos=(0.0, 0.2, 0.0), forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)
    )
    assert scan_far.kind[0] == WALL
    assert scan_far.dist[0] == pytest.approx(34.125, abs=0.02)


# --- Review finding 3: overhead surface must not erase the road below ---

def test_overhead_panel_does_not_erase_road_below():
    road = _slab(0.0, -4.0, 65.0)
    overhead = _slab(2.0, 5.0, 8.0)
    scan = Lidar(_mesh(road, overhead), LidarLayout(angles_deg=(0.0,))).scan(
        pos=(0.0, 0.2, 0.0), forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)
    )
    assert scan.kind[0] == CLEAR
    assert scan.dist[0] == pytest.approx(60.0)
    assert np.allclose(scan.height[0], [-0.2, -0.2, -0.2], atol=1e-3)

    # Overhead roof extending directly over the car as well must not hijack the anchor probe.
    roof = _slab(2.0, -4.0, 12.0)
    scan_under = Lidar(_mesh(road, roof), LidarLayout(angles_deg=(0.0,))).scan(
        pos=(0.0, 0.2, 0.0), forward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)
    )
    assert scan_under.kind[0] == CLEAR
    assert scan_under.dist[0] == pytest.approx(60.0)
    assert np.allclose(scan_under.height[0], [-0.2, -0.2, -0.2], atol=1e-3)

    # Whole-frame 90-degree roll invariance (car-up frame preserved).
    rot = g._rz(np.deg2rad(90.0))
    world_rot = _mesh(_mesh(road, overhead).tris @ rot.T)
    scan_rot = Lidar(world_rot, LidarLayout(angles_deg=(0.0,))).scan(
        pos=rot @ [0.0, 0.2, 0.0], forward=rot @ [0.0, 0.0, 1.0], up=rot @ [0.0, 1.0, 0.0]
    )
    assert scan_rot.kind[0] == CLEAR
    assert np.allclose(scan_rot.height[0], [-0.2, -0.2, -0.2], atol=1e-3)


# --- Review finding 4: placement search must accept passing candidate over failing equal-coverage one ---

def test_fit_placement_accepts_passing_candidate_at_equal_coverage():
    # Two road segments: segment A is at y=0 with no mobil offset; segment B has a
    # STEP_MOBIL_GEOM offset of -0.8 m. With use_mobil_geom=True, ALL 20 samples still
    # fall inside [-0.05, 1.0] (16 at ride=0.04, 4 at ride=0.84), so on_surface == 1.0,
    # median == 0.04, BUT ride_p90_m == 0.80 > 0.5 -> best.ok() is False!
    # Switching to use_mobil_geom=False keeps on_surface == 1.0 while bringing ride_p90_m
    # to 0.0 -> cand.ok() is True and MUST replace the failing default despite 0.0 coverage gain.
    seg_a_verts = np.array([[-4, 0, -4], [4, 0, -4], [4, 0, 50], [-4, 0, 50]], dtype=np.float32)
    seg_b_verts = np.array([[-4, 0, 50], [4, 0, 50], [4, 0, 70], [-4, 0, 70]], dtype=np.float32)
    tris = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
    ids = np.array([[16, 0], [16, 0]], dtype=np.uint16)
    shapes = [g.Shape("A", 0, seg_a_verts, tris, ids), g.Shape("B", 0, seg_b_verts, tris, ids)]
    offset = g.Step(g.STEP_MOBIL_GEOM, np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0]))
    instances = [
        g.Instance(g.SRC_BLOCKS, g.KIND_GRID_BLOCK, "RoadA", [g.Part(0, False, [])], coord=(0, 8, 0)),
        g.Instance(g.SRC_BLOCKS, g.KIND_GRID_BLOCK, "RoadB", [g.Part(1, False, [offset])], coord=(0, 8, 0)),
    ]
    export = g.MapExport("uid", "T", "Deco", (48, 40, 48), 8, shapes, instances)
    z = np.concatenate([np.linspace(0.0, 45.0, 16), np.linspace(52.0, 65.0, 4)])
    pos = np.column_stack([np.zeros(20), np.full(20, 0.04), z])
    up = np.tile([0.0, 1.0, 0.0], (20, 1))

    default_fit = grade(g.build_world(export, g.Placement()), pos, up, g.Placement())
    assert default_fit.on_surface == 1.0 and not default_fit.ok() and default_fit.ride_p90_m > 0.5

    best, _ = fit_placement(export, pos, up, log=lambda *_: None)
    assert best.ok()
    assert best.placement.use_mobil_geom is False
    assert best.on_surface == 1.0 and best.ride_p90_m == pytest.approx(0.0, abs=1e-6)


# --- Review finding 5: per-hit ride window and hit selection past blockers in grade() ---

def test_grade_enforces_per_hit_ride_window_and_selects_past_blockers():
    road = _slab(0.0, -4.0, 65.0)
    up = np.tile([0.0, 1.0, 0.0], (20, 1))

    # Downward-wound two-sided road triangles still count as valid support.
    pos = np.column_stack([np.zeros(20), np.full(20, 0.04), np.linspace(0.0, 60.0, 20)])
    fit_ok = grade(_mesh(road), pos, up, g.Placement())
    assert fit_ok.on_surface == 1.0 and fit_ok.ok()

    # One sample 1.3 m above the road exceeds RIDE_MAX_M (1.0 m) and must NOT score as supported.
    pos_high = pos.copy()
    pos_high[-1, 1] = 1.3
    fit_one = grade(_mesh(road), pos_high, up, g.Placement())
    assert fit_one.on_surface == pytest.approx(19 / 20)
    pos_high[-2, 1] = 1.3
    fit_two = grade(_mesh(road), pos_high, up, g.Placement())
    assert fit_two.on_surface == pytest.approx(18 / 20) and not fit_two.ok()

    # Steep blocker right in front of the road (both above car and inside the ride window)
    # must be skipped during hit selection so the flat road underneath is chosen.
    pos_flat = np.column_stack([np.zeros(20), np.full(20, 0.2), np.linspace(0.0, 60.0, 20)])
    steep_inside = _quad_pts((-0.01, 0.06, -4.0), (0.01, 0.14, -4.0), (0.01, 0.14, 65.0), (-0.01, 0.06, 65.0))
    fit_steep = grade(_mesh(road, steep_inside), pos_flat, up, g.Placement())
    assert fit_steep.ok() and fit_steep.on_surface == 1.0 and fit_steep.ride_median_m == pytest.approx(0.2, abs=1e-4)

    # Overhead road-like slab above ride_min (-0.05 m) must also be skipped in favour of the road below.
    overhead_slab = _slab(0.4, -4.0, 65.0)
    fit_over = grade(_mesh(road, overhead_slab), pos_flat, up, g.Placement())
    assert fit_over.ok() and fit_over.on_surface == 1.0 and fit_over.ride_median_m == pytest.approx(0.2, abs=1e-4)
