"""Map geometry export format, placement, ray casting and the virtual lidar,
on small hand-built maps written in the plugin's format."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tmai.track import geometry as g
from tmai.track.lidar import CLEAR, DROP, WALL, Lidar, LidarLayout, fit_placement, lap_samples


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<H", len(b)) + b


def _parts(parts) -> bytes:
    out = struct.pack("<I", len(parts))
    for shape, steps in parts:
        out += struct.pack("<IBB", shape, 0, len(steps))
        for kind, vals in steps:
            out += struct.pack("<BB", kind, len(vals)) + struct.pack(f"<{len(vals)}f", *vals)
    return out


def quad(x0, x1, y, z0, z1):
    """Upward road slab as (verts, tris)."""
    v = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    return v, [(0, 1, 2), (0, 2, 3)]


def wall(x, y0, y1, z0, z1):
    v = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    return v, [(0, 1, 2), (0, 2, 3)]


def write_export(path, shapes, blocks=(), items=(), name="Test"):
    """shapes: list of (verts, tris, physic id); blocks: (name, coord, dir, size, parts);
    items: (name, pos, yaw, parts)."""
    out = struct.pack("<IH", g.MAGIC, g.VERSION) + _str("uid1") + _str(name) + _str("Deco")
    out += struct.pack("<IIII", 48, 40, 48, 8)
    out += struct.pack("<I", len(shapes))
    for verts, tris, phys in shapes:
        out += _str("shape") + struct.pack("<I", 0)
        out += struct.pack("<I", len(verts)) + struct.pack(f"<{3 * len(verts)}f", *np.ravel(verts))
        out += struct.pack("<I", len(tris)) + struct.pack(f"<{3 * len(tris)}I", *np.ravel(tris))
        out += struct.pack(f"<{2 * len(tris)}H", *([phys, 0] * len(tris)))
    out += struct.pack("<I", len(blocks) + len(items))
    for bname, coord, d, size, parts in blocks:
        out += struct.pack("<BB", g.SRC_BLOCKS, g.KIND_GRID_BLOCK) + _str(bname)
        out += struct.pack("<iiiBB", *coord, d, 0) + struct.pack("<III", *size)
        out += struct.pack("<6f", *([0.0] * 6)) + struct.pack("<III", 0, 0, 0) + _parts(parts)
    for iname, pos, yaw, parts in items:
        out += struct.pack("<BB", g.SRC_ITEMS, g.KIND_ITEM) + _str(iname)
        out += struct.pack("<3f", *pos) + struct.pack("<4f", yaw, 0.0, 0.0, 1.0)
        out += struct.pack("<B", 0) + struct.pack("<24f", *([0.0] * 24))
        out += struct.pack("<3f", 0, 0, 0) + struct.pack("<B", 0) + _parts(parts)
    path.write_bytes(out)
    return path


def straight_road_map(tmp_path, rows: int = 6, wall_left: bool = True):
    """A road one block wide along +z at grid row cy=9 (y = 8 with yoff -64),
    with a wall along its x=0 side (or none)."""
    shapes = [(*quad(0, 32, 0.0, 0, 32), 16)]
    if wall_left:
        shapes.append((*wall(0.0, 0.0, 4.0, 0, 32), 16))
    parts = [(i, [(g.STEP_MOBIL_GEOM, [0, 0, 0, 0, 0, 0])]) for i in range(len(shapes))]
    blocks = [(f"RoadTechStraight{k}", (10, 9, 10 + k), 0, (1, 1, 1), parts) for k in range(rows)]
    return write_export(tmp_path / "m.tmgeo", shapes, blocks)


def test_export_round_trip(tmp_path):
    path = straight_road_map(tmp_path)
    ex = g.read_export(path)
    assert ex.uid == "uid1" and ex.name == "Test" and ex.size == (48, 40, 48)
    assert len(ex.shapes) == 2 and len(ex.instances) == 6
    assert ex.instances[0].coord == (10, 9, 10)
    assert g.read_header_name(path) == "Test"
    assert g.latest_export(tmp_path, "Test") == path
    assert g.latest_export(tmp_path, "Other") is None


def test_rejects_wrong_version(tmp_path):
    path = straight_road_map(tmp_path)
    data = bytearray(path.read_bytes())
    data[4:6] = struct.pack("<H", 99)
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="version"):
        g.read_export(path)


@pytest.mark.parametrize("d, local, world", [
    (0, (1, 0, 2), (1, 0, 2)),
    (1, (1, 0, 2), (32 - 2, 0, 1)),      # (SZ - z, x)
    (2, (1, 0, 2), (32 - 1, 0, 32 - 2)), # (SX - x, SZ - z)
    (3, (1, 0, 2), (2, 0, 32 - 1)),      # (z, SX - x)
])
def test_grid_direction_keeps_block_on_its_cell(d, local, world):
    inst = g.Instance(g.SRC_BLOCKS, g.KIND_GRID_BLOCK, "b", [], coord=(0, 8, 0), dir=d)
    m = g.instance_matrix(inst, g.Placement(yoff=-64))
    out = m[:3, :3] @ np.array(local, float) + m[:3, 3]
    assert np.allclose(out, world)


def test_quaternion_and_euler():
    assert np.allclose(g.quat_matrix(0, 0, 0, 1), np.eye(3))
    half = np.sqrt(0.5)
    r = g.quat_matrix(0, half, 0, half)            # 90 degrees about +Y
    assert np.allclose(r @ [1, 0, 0], [0, 0, -1])
    assert np.allclose(g.euler(0, np.pi / 2, 0), r)


def test_ray_caster_nearest_hit_and_miss():
    tris = np.array([
        [[0, 0, -5], [0, 5, 5], [0, -5, 5]],       # plane x=0
        [[3, 0, -5], [3, 5, 5], [3, -5, 5]],       # plane x=3
    ], float)
    rc = g.RayCaster(tris, cell=4.0)
    d, t = rc.cast([[-2, 0.5, 0.5], [-2, 0.5, 0.5], [-2, 50, 0]], [[1, 0, 0], [-1, 0, 0], [1, 0, 0]], 10.0)
    assert d[0] == pytest.approx(2.0) and t[0] == 0
    assert np.isinf(d[1]) and t[1] == -1
    assert np.isinf(d[2])
    d, _ = rc.cast([[-2, 0.5, 0.5]], [[1, 0, 0]], 1.0)    # beyond max distance
    assert np.isinf(d[0])


def test_ray_caster_long_diagonal_ray_crosses_buckets():
    tris = np.array([[[100, 0, 90], [100, 0, 110], [100, 10, 100]]], float)
    rc = g.RayCaster(tris, cell=8.0)
    o = np.array([0.0, 2.0, 0.0])
    target = np.array([100.0, 2.0, 100.0])
    dvec = (target - o) / np.linalg.norm(target - o)
    d, _ = rc.cast(o[None], dvec[None], 200.0)
    assert d[0] == pytest.approx(np.linalg.norm(target - o), rel=1e-6)


def test_not_collidable_triangles_are_dropped(tmp_path):
    shapes = [(*quad(0, 32, 0.0, 0, 32), g.NOT_COLLIDABLE)]
    path = write_export(tmp_path / "n.tmgeo", shapes, [("b", (0, 8, 0), 0, (1, 1, 1), [(0, [])])])
    assert len(g.build_world(g.read_export(path)).tris) == 0


def test_lidar_sees_wall_left_and_edge_right(tmp_path):
    world = g.build_world(g.read_export(straight_road_map(tmp_path)), g.Placement(yoff=-64))
    lidar = Lidar(world, LidarLayout(angles_deg=(-90, 0, 90), range_m=60.0))
    # Car in the middle of the road (x = 336, block 10), driving +z at y = 8.4.
    # Forward +z and up +y make "left" = up x forward = +x, so +90 deg looks at x > 336.
    scan = lidar.scan(pos=(336.0, 8.4, 340.0), forward=(0, 0, 1), up=(0, 1, 0))
    right, ahead, left = scan.dist
    assert scan.kind[0] == WALL and right == pytest.approx(16.0, abs=0.01)   # wall at x = 320
    assert scan.kind[2] == DROP and 15.0 <= left <= 20.0                    # road ends at x = 352
    assert scan.kind[1] == CLEAR and ahead == 60.0                          # road ends 172 m ahead
    near_end = lidar.scan(pos=(336.0, 8.4, 480.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert near_end.kind[1] == DROP and 32.0 <= near_end.dist[1] <= 38.0    # road ends at z = 512
    feats = scan.features(lidar.layout)
    assert feats.shape == (15,) and feats[1] == 1.0 and feats[11] == 0.0   # 5 per ray
    assert feats[2] == pytest.approx(-0.04, abs=0.01)                       # road 0.4 m below at 9 m


def test_lidar_ramp_ahead_is_road_not_wall(tmp_path):
    ramp_v = [(0, 0, 0), (32, 0, 0), (32, 8, 32), (0, 8, 32)]
    shapes = [(*quad(0, 32, 0.0, 0, 32), 16), (ramp_v, [(0, 1, 2), (0, 2, 3)], 16)]
    blocks = [("flat", (10, 9, 10), 0, (1, 1, 1), [(0, [])]), ("ramp", (10, 9, 11), 0, (1, 1, 1), [(1, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "r.tmgeo", shapes, blocks)))
    lidar = Lidar(world, LidarLayout(angles_deg=(0,), range_m=40.0, probes_m=(4, 8, 12), height_at_m=(8,)))
    scan = lidar.scan(pos=(336.0, 8.4, 330.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert scan.kind[0] != WALL


def test_fit_finds_grid_height_from_laps(tmp_path):
    ex = g.read_export(straight_road_map(tmp_path))
    # Laps driven on the road placed at yoff = -56 (one row higher than the default).
    z = np.linspace(325, 500, 60)
    pos = np.stack([np.full_like(z, 336.0), np.full_like(z, 16.4), z], axis=1)
    up = np.tile([0.0, 1.0, 0.0], (len(z), 1))
    fit, tried = fit_placement(ex, pos, up, log=lambda *_: None)
    assert fit.placement.yoff == -56.0
    assert fit.on_surface == 1.0 and fit.ride_median_m == pytest.approx(0.4, abs=1e-3)
    assert fit.ok() and len(tried) > 20


def test_lap_samples_keep_on_ground_frames(tmp_path):
    f = tmp_path / "lap.npz"
    flags = np.array([8, 0, 8, 8, 9], dtype=np.int64)
    np.savez(f, flags=flags, pos=np.arange(15.0).reshape(5, 3), up=np.tile([0, 1, 0], (5, 1)))
    pos, up = lap_samples([f], every=1)
    assert len(pos) == 4 and np.allclose(pos[1], [6, 7, 8])


def test_scan_is_fast_on_a_dense_map(tmp_path):
    rng = np.random.default_rng(0)
    # 400 blocks of road plus 40k scattered triangles, like pillars and decoration.
    shapes = [(*quad(0, 32, 0.0, 0, 32), 16), (*wall(0.0, 0.0, 4.0, 0, 32), 16)]
    clutter = rng.uniform(-3, 3, size=(200, 3))
    verts = np.concatenate([clutter, clutter + [0.5, 0, 0], clutter + [0, 0.5, 0.5]])
    tris = [(i, i + 200, i + 400) for i in range(200)]
    shapes.append((verts, tris, 16))
    parts = [(0, []), (1, []), (2, [])]
    blocks = [(f"b{i}", (i % 20, 9, i // 20), i % 4, (1, 1, 1), parts) for i in range(400)]
    world = g.build_world(g.read_export(write_export(tmp_path / "d.tmgeo", shapes, blocks)))
    lidar = Lidar(world)
    lidar.scan((320.0, 8.4, 320.0), (0, 0, 1), (0, 1, 0))
    times = [lidar.scan((320.0 + k, 8.4, 320.0), (0.3, 0, 1), (0, 1, 0)).seconds for k in range(10)]
    assert np.median(times) < 0.03


def test_lap_samples_without_flags_use_wheel_material(tmp_path):
    f = tmp_path / "lap.npz"
    wm = np.array([[16, 16, 16, 16], [80, 80, 80, 80], [16, 80, 80, 80], [5, 80, 16, 80], [80, 80, 80, 80]])
    np.savez(f, wheel_material=wm, pos=np.arange(15.0).reshape(5, 3), up=np.tile([0, 1, 0], (5, 1)))
    pos, _ = lap_samples([f], every=1)
    assert np.allclose(pos, [[0, 1, 2], [9, 10, 11]])
    g2 = tmp_path / "bare.npz"
    np.savez(g2, pos=np.arange(9.0).reshape(3, 3), up=np.tile([0, 1, 0], (3, 1)))
    assert len(lap_samples([g2], every=1)[0]) == 3


def _geometry_cfg(tmp_path, monkeypatch):
    from tmai.config import load_config
    from tmai.track import commands

    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(commands, "_geometry_storage", lambda: storage)
    return commands, storage, load_config(None, [f"run.output_dir={tmp_path / 'runs'}", "run.track=T"])


def test_geometry_check_refuses_export_of_another_map(tmp_path, monkeypatch, capsys):
    commands, storage, cfg = _geometry_cfg(tmp_path, monkeypatch)
    write_export(storage / "other.tmgeo", [(*quad(0, 32, 0.0, 0, 32), 16)], name="Fall 2026 - 01")
    assert commands.geometry_check(cfg) == 1
    out = capsys.readouterr().out
    assert "other.tmgeo" in out and "'Fall 2026 - 01'" in out and "Load 'T'" in out
    assert "--export" in out


def test_geometry_check_export_flag_forces_a_file(tmp_path, monkeypatch, capsys):
    commands, storage, cfg = _geometry_cfg(tmp_path, monkeypatch)
    write_export(storage / "other.tmgeo", [(*quad(0, 32, 0.0, 0, 32), 16)], name="Fall 2026 - 01")
    assert commands.geometry_check(cfg, "other.tmgeo") == 1
    out = capsys.readouterr().out
    assert "map 'Fall 2026 - 01'" in out and "no demo laps for 'T'" in out
    assert commands.geometry_check(cfg, "missing.tmgeo") == 1


def test_prefab_quaternions_are_read_w_first():
    half = np.sqrt(0.5)
    # Stored as the plugin writes them; real exports carry identity as (1, 0, 0, 0).
    ident = g.Step(g.STEP_TRANS_QUAT, np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(g.step_matrix(ident, g.Placement()), g.affine(trans=(1, 2, 3)))
    yaw90 = g.Step(g.STEP_TRANS_QUAT, np.array([0, 0, 0, half, 0.0, half, 0.0]))
    assert np.allclose(g.step_matrix(yaw90, g.Placement())[:3, :3] @ [1, 0, 0], [0, 0, -1])
    flipped = g.step_matrix(ident, g.Placement(quat_wxyz=False))[:3, :3]
    assert np.allclose(flipped @ [0, 1, 0], [0, -1, 0])          # x-first reading: 180 deg about X


def test_lidar_follows_a_steep_ramp_and_reports_its_height(tmp_path):
    # Flat block, then a ramp rising 16 m over the next 32 m (about 27 degrees).
    ramp_v = [(0, 0, 0), (32, 0, 0), (32, 16, 32), (0, 16, 32)]
    shapes = [(*quad(0, 32, 0.0, 0, 32), 16), (ramp_v, [(0, 1, 2), (0, 2, 3)], 16)]
    blocks = [("flat", (10, 9, 10), 0, (1, 1, 1), [(0, [])]), ("ramp", (10, 9, 11), 0, (1, 1, 1), [(1, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "s.tmgeo", shapes, blocks)))
    lidar = Lidar(world, LidarLayout(angles_deg=(0,), range_m=40.0, probes_m=(3, 9, 20, 30, 38),
                                     height_at_m=(9, 20, 38)))
    # Car at z = 325, so the ramp starts 27 m ahead and the probe at 38 m is 11 m up it.
    scan = lidar.scan(pos=(336.0, 8.0, 325.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert scan.kind[0] != DROP
    assert scan.height[0, 0] == pytest.approx(0.0, abs=0.01)
    assert scan.height[0, 2] == pytest.approx(11 * 0.5, abs=0.05)


def test_overhead_gate_does_not_hide_the_road_under_it(tmp_path):
    # Flat road, with a gate's top bar 8 m above it across the next block.
    shapes = [(*quad(0, 32, 0.0, 0, 32), 16), (*quad(0, 32, 8.0, 0, 32), 4)]
    blocks = [("flat", (10, 9, 10), 0, (1, 1, 1), [(0, [])]),
              ("flat", (10, 9, 11), 0, (1, 1, 1), [(0, []), (1, [])])]
    world = g.build_world(g.read_export(write_export(tmp_path / "gate.tmgeo", shapes, blocks)))
    lidar = Lidar(world, LidarLayout(angles_deg=(0,), range_m=40.0, probes_m=(3, 9, 20, 30, 38),
                                     height_at_m=(9, 20, 38)))
    scan = lidar.scan(pos=(336.0, 8.0, 325.0), forward=(0, 0, 1), up=(0, 1, 0))
    assert scan.height[0] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
