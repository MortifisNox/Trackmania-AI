"""The map's collision geometry, as exported by the TMAI Map Geometry plugin,
placed in the world, plus a ray caster over it.

The plugin writes raw placements (grid coordinate and direction, or position
and angles) and, for every collision surface, the chain of local transforms
it found under the block or item. All placement maths happens here, so it can
be checked against recorded laps (`tmai.track.lidar.check_against_laps`)
without reloading anything in the game. Format: openplanet/TMAI_MapGeometry/main.as.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x4F454754          # "TGEO"
VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
CELL = np.array([32.0, 8.0, 32.0])   # one block grid cell, metres
NOT_COLLIDABLE = 28          # EPlugSurfaceMaterialId::NotCollidable
ID_UNKNOWN = 0xFFFF          # primitives written as boxes carry no surface id

SRC_BLOCKS, SRC_BAKED, SRC_ITEMS = 0, 1, 2
KIND_GRID_BLOCK, KIND_FREE_BLOCK, KIND_ITEM = 0, 1, 2
STEP_TRANS_QUAT, STEP_MOBIL_GEOM = 1, 2   # trans + quat values: tx, ty, tz, then the plugin's q.x, q.y, q.z, q.w


@dataclass
class Step:
    kind: int
    values: np.ndarray


@dataclass
class Part:
    shape: int
    moving: bool
    steps: list[Step]


@dataclass
class Instance:
    source: int
    kind: int
    name: str
    parts: list[Part]
    coord: tuple[int, int, int] = (0, 0, 0)
    dir: int = 0
    ground: bool = False
    size: tuple[int, int, int] = (1, 1, 1)   # variant size in grid cells
    variant_index: int = 0
    mobil_variant_index: int = 0
    mobil_index: int = 0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ypr: np.ndarray = field(default_factory=lambda: np.zeros(3))   # free block: yaw, pitch, roll
    angles: np.ndarray = field(default_factory=lambda: np.zeros(3))   # item: yaw, pitch, roll
    scale: float = 1.0
    location_set: bool = False
    block_location: np.ndarray = field(default_factory=lambda: np.zeros(12))
    location_in_block: np.ndarray = field(default_factory=lambda: np.zeros(12))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pivot_ok: bool = False


@dataclass
class Shape:
    name: str
    approx: int              # primitives written as their bounding box
    verts: np.ndarray        # (V, 3) float32, surface-local
    tris: np.ndarray         # (T, 3) uint32
    ids: np.ndarray          # (T, 2) uint16: physic id, gameplay id


@dataclass
class MapExport:
    uid: str
    name: str
    decoration: str
    size: tuple[int, int, int]
    deco_base_height: int
    shapes: list[Shape]
    instances: list[Instance]


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, fmt: str):
        out = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return out if len(out) > 1 else out[0]

    def array(self, dtype, count: int) -> np.ndarray:
        dt = np.dtype(dtype).newbyteorder("<")
        out = np.frombuffer(self.data, dtype=dt, count=count, offset=self.pos)
        self.pos += dt.itemsize * count
        return out.copy()

    def string(self) -> str:
        n = self.take("H")
        out = self.data[self.pos:self.pos + n].decode("utf-8", errors="replace")
        self.pos += n
        return out


def _read_parts(r: _Reader) -> list[Part]:
    parts = []
    for _ in range(r.take("I")):
        shape, flags, n_steps = r.take("IBB")
        steps = []
        for _ in range(n_steps):
            kind, n = r.take("BB")
            steps.append(Step(kind, r.array(np.float32, n).astype(np.float64)))
        parts.append(Part(shape, bool(flags & 1), steps))
    return parts


def _parse_export(data: bytes, path: Path) -> MapExport:
    r = _Reader(data)
    magic, version = r.take("IH")
    if magic != MAGIC:
        raise ValueError(f"{path} is not a TMAI map geometry file")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"{path} has format version {version}, this tmai reads {SUPPORTED_VERSIONS}; reinstall the plugins")
    uid, name, deco = r.string(), r.string(), r.string()
    size = r.take("III")
    deco_base = r.take("I")
    shapes = []
    for _ in range(r.take("I")):
        sname = r.string()
        approx = r.take("I")
        nv = r.take("I")
        verts = r.array(np.float32, nv * 3).reshape(nv, 3)
        nt = r.take("I")
        tris = r.array(np.uint32, nt * 3).reshape(nt, 3)
        ids = r.array(np.uint16, nt * 2).reshape(nt, 2)
        shapes.append(Shape(sname, approx, verts, tris, ids))
    instances = []
    for _ in range(r.take("I")):
        source, kind = r.take("BB")
        iname = r.string()
        if kind == KIND_ITEM:
            pos = np.array(r.take("3f"))
            yaw, pitch, roll, scale = r.take("4f")
            loc_set = bool(r.take("B"))
            block_loc = r.array(np.float32, 12).astype(np.float64)
            loc_in = r.array(np.float32, 12).astype(np.float64)
            pivot = np.array(r.take("3f"))
            pivot_ok = bool(r.take("B"))
            inst = Instance(source, kind, iname, _read_parts(r), pos=pos,
                            angles=np.array([yaw, pitch, roll]), scale=scale, location_set=loc_set,
                            block_location=block_loc, location_in_block=loc_in, pivot=pivot, pivot_ok=pivot_ok)
        else:
            cx, cy, cz, d, ground = r.take("iiiBB")
            sx, sy, sz = r.take("III")
            pos = np.array(r.take("3f"))
            ypr = np.array(r.take("3f"))
            if version >= 2:
                v_idx, mv_idx, m_idx = r.take("III")
            else:
                v_idx = mv_idx = m_idx = 0
            inst = Instance(source, kind, iname, _read_parts(r), coord=(cx, cy, cz), dir=d,
                            ground=bool(ground), size=(sx, sy, sz), variant_index=v_idx,
                            mobil_variant_index=mv_idx, mobil_index=m_idx, pos=pos, ypr=ypr)
        instances.append(inst)
    if r.pos != len(r.data):
        raise ValueError(f"{path}: {len(r.data) - r.pos} unread bytes; plugin and tmai disagree on the format")
    return MapExport(uid, name, deco, size, deco_base, shapes, instances)


def read_export(path: Path) -> MapExport:
    p = Path(path)
    return _parse_export(p.read_bytes(), p)


# ---------------------------------------------------------------- transforms

def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def affine(rot=None, trans=None, scale: float = 1.0) -> np.ndarray:
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    if scale != 1.0:
        m[:3, :3] *= scale
    if trans is not None:
        m[:3, 3] = trans
    return m


def euler(pitch: float, yaw: float, roll: float, sign: float = 1.0) -> np.ndarray:
    """Trackmania's yaw-roll-pitch rotation (as composed by Openplanet plugins
    that draw placed items): yaw about +Y, then roll about Z, then pitch about X."""
    return _ry(sign * yaw) @ _rz(sign * roll) @ _rx(sign * pitch)


def quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# Grid block direction -> rotation of block-local (x, z) and the shift that puts
# the turned block back on its own cells (SX, SZ = footprint in metres).
# Clockwise seen from above: dir 1 maps local (x, z) to (SZ - z, x).
def _dir_matrix(d: int, sx: float, sz: float, mirror: bool) -> np.ndarray:
    if mirror and d in (1, 3):
        d = 4 - d
    if d == 1:
        return affine(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), (sz, 0, 0))
    if d == 2:
        return affine(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]), (sx, 0, sz))
    if d == 3:
        return affine(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), (0, 0, sx))
    return np.eye(4)


@dataclass(frozen=True)
class Placement:
    """The conventions that turn raw placements into world positions. `yoff`
    (the world height of grid row 0) differs per map and is fitted; the rest
    are switches the lap check can try when a map does not line up."""
    yoff: float = -64.0
    mirror_dirs: bool = False
    euler_sign: float = 1.0
    use_mobil_geom: bool = True
    use_item_pivot: bool = True
    # Prefab entry quaternions arrive as (w, x, y, z): identity shows up as
    # (1, 0, 0, 0) in real exports, and only this order joins item pieces up.
    quat_wxyz: bool = True


def instance_matrix(inst: Instance, p: Placement) -> np.ndarray:
    if inst.kind == KIND_GRID_BLOCK:
        cx, cy, cz = inst.coord
        sx, sz = inst.size[0] * CELL[0], inst.size[2] * CELL[2]
        base = affine(trans=(cx * CELL[0], cy * CELL[1] + p.yoff, cz * CELL[2]))
        return base @ _dir_matrix(inst.dir, sx, sz, p.mirror_dirs)
    if inst.kind == KIND_FREE_BLOCK:
        yaw, pitch, roll = inst.ypr
        return affine(euler(pitch, yaw, roll, p.euler_sign), inst.pos)
    yaw, pitch, roll = inst.angles
    m = affine(euler(pitch, yaw, roll, p.euler_sign), inst.pos, inst.scale if inst.scale > 1e-4 else 1.0)
    if p.use_item_pivot and inst.pivot_ok:
        m = m @ affine(trans=inst.pivot)
    return m


def step_matrix(step: Step, p: Placement) -> np.ndarray:
    v = step.values
    if step.kind == STEP_TRANS_QUAT:
        a, b, c, d = v[3:7]
        q = (b, c, d, a) if p.quat_wxyz else (a, b, c, d)   # quat_matrix takes x, y, z, w
        return affine(quat_matrix(*q), v[:3])
    if step.kind == STEP_MOBIL_GEOM:
        if not p.use_mobil_geom:
            return np.eye(4)
        rx, ry, rz = np.radians(v[3:6])
        return affine(euler(rx, ry, rz, p.euler_sign), v[:3])
    return np.eye(4)


def unique_instances(instances: list[Instance]) -> list[Instance]:
    """Blocks and BakedBlocks overlap; keep one copy of each placed block
    (BakedBlocks adds the generated ones: pillars, clips, terrain). Only blocks
    that produced collision parts mark a key as seen, and keys distinguish
    ground state, variant and mobil indices."""
    seen, out = set(), []
    for inst in sorted(instances, key=lambda i: i.source):
        if not inst.parts:
            continue
        if inst.kind == KIND_GRID_BLOCK:
            key = ("g", inst.name, inst.coord, inst.dir, inst.ground,
                   inst.variant_index, inst.mobil_variant_index, inst.mobil_index)
        elif inst.kind == KIND_FREE_BLOCK:
            key = ("f", inst.name, tuple(np.round(inst.pos, 2)), tuple(np.round(inst.ypr, 3)),
                   inst.ground, inst.variant_index, inst.mobil_variant_index, inst.mobil_index)
        else:
            key = ("i", id(inst))
        if key in seen:
            continue
        seen.add(key)
        out.append(inst)
    return out


@dataclass
class WorldMesh:
    tris: np.ndarray       # (N, 3, 3) float64 world corners
    physic: np.ndarray     # (N,) uint16
    gameplay: np.ndarray   # (N,) uint16
    owner: np.ndarray      # (N,) index into `instances`
    instances: list[Instance]


def build_world(export: MapExport, placement: Placement = Placement(),
                include_moving: bool = True) -> WorldMesh:
    instances = unique_instances(export.instances)
    chunks, phys, gameplay, owners = [], [], [], []
    for k, inst in enumerate(instances):
        base = instance_matrix(inst, placement)
        for part in inst.parts:
            if part.moving and not include_moving:
                continue
            shape = export.shapes[part.shape]
            m = base
            for st in part.steps:
                m = m @ step_matrix(st, placement)
            v = shape.verts.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
            chunks.append(v[shape.tris])
            phys.append(shape.ids[:, 0])
            gameplay.append(shape.ids[:, 1])
            owners.append(np.full(len(shape.tris), k, dtype=np.int32))
    if not chunks:
        empty = np.zeros((0, 3, 3))
        return WorldMesh(empty, np.zeros(0, np.uint16), np.zeros(0, np.uint16), np.zeros(0, np.int32), instances)
    tris = np.concatenate(chunks)
    physic = np.concatenate(phys)
    keep = physic != NOT_COLLIDABLE
    return WorldMesh(tris[keep], physic[keep], np.concatenate(gameplay)[keep],
                     np.concatenate(owners)[keep], instances)


# ---------------------------------------------------------------- ray casting

class RayCaster:
    """Two-sided ray casting over a static triangle soup. Trackmania
    `GmSurfMesh` collision surfaces collide from both sides without guaranteed
    upward winding, so Möller–Trumbore tests `abs(det) > 1e-12` (no back-face
    culling). Triangles are bucketed in a 3D grid of `cell`-metre cubes (maps
    stack pillars, decoration and ground under the road, so height matters);
    each batch of rays gathers the cubes its segments pass through and tests
    every (ray, triangle) pair at once."""

    BITS = 16                       # per axis: +-32768 cubes, far beyond any map
    OFFSET = 1 << (BITS - 1)
    KEY_MASK = (1 << (3 * BITS)) - 1   # rays go in the bits above the cube key
    # Signed int64 holds 15 non-negative bits above the 48-bit cube key (max 32768).
    # Chunking well below that prevents overflow for arbitrarily large ray batches.
    MAX_BATCH = 16384
    _MID_CORNERS = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(1, 7)], dtype=bool)

    def __init__(self, tris: np.ndarray, cell: float = 4.0):
        tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
        self.cell = float(cell)
        self.v0 = tris[:, 0]
        self.e1 = tris[:, 1] - tris[:, 0]
        self.e2 = tris[:, 2] - tris[:, 0]
        n = np.cross(self.e1, self.e2)
        self.normal = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.count = len(tris)
        if self.count == 0:
            self._keys = np.zeros(0, np.int64)
            self._starts = np.zeros(1, np.int64)
            self._ids = np.zeros(0, np.int64)
            return
        lo = np.floor(tris.min(axis=1) / self.cell).astype(np.int64)
        hi = np.floor(tris.max(axis=1) / self.cell).astype(np.int64)
        size = hi - lo + 1
        per = size.prod(axis=1)
        tri_ids = np.repeat(np.arange(self.count), per)
        offs = np.arange(per.sum()) - np.repeat(np.cumsum(per) - per, per)
        sx, sy = np.repeat(size[:, 0], per), np.repeat(size[:, 1], per)
        ix = lo[tri_ids, 0] + offs % sx
        iy = lo[tri_ids, 1] + (offs // sx) % sy
        iz = lo[tri_ids, 2] + offs // (sx * sy)
        keys = self._key(ix, iy, iz)
        order = np.argsort(keys, kind="stable")
        keys, tri_ids = keys[order], tri_ids[order]
        self._keys, first = np.unique(keys, return_index=True)
        self._starts = np.append(first, len(keys))
        self._ids = tri_ids

    @classmethod
    def _key(cls, ix, iy, iz):
        b, o = cls.BITS, cls.OFFSET
        return (((np.asarray(ix, np.int64) + o) << (2 * b))
                | ((np.asarray(iy, np.int64) + o) << b)
                | (np.asarray(iz, np.int64) + o))

    def _candidates_chunk(self, origins: np.ndarray, dirs: np.ndarray, max_dist: np.ndarray):
        """(ray index, triangle index) pairs for a chunk with len(origins) <= MAX_BATCH."""
        step = self.cell * 0.25
        n = np.ceil(max_dist / step).astype(np.int64) + 1
        ray = np.repeat(np.arange(len(origins), dtype=np.int64), n)
        k = np.arange(n.sum(), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
        t = np.minimum(k * step, max_dist[ray])
        cells = np.floor((origins[ray] + dirs[ray] * t[:, None]) / self.cell).astype(np.int64)
        # Samples are a quarter cube apart, so neighbours differ by at most one
        # cube per axis. Both endpoints are already in `cells`; intermediate
        # corners of the spanned box are only needed when >= 2 axes change.
        keys = self._key(cells[:, 0], cells[:, 1], cells[:, 2])
        shift = 3 * self.BITS
        packed = (ray << shift) | keys
        if len(ray) > 1:
            same = ray[1:] == ray[:-1]
            diff = cells[1:] != cells[:-1]
            diag = same & (diff.sum(axis=1) >= 2)
            if diag.any():
                c_cur = cells[1:][diag]
                c_prv = cells[:-1][diag]
                box = np.where(self._MID_CORNERS[None], c_cur[:, None, :], c_prv[:, None, :])  # (D, 6, 3)
                d_keys = self._key(box[..., 0], box[..., 1], box[..., 2]).reshape(-1)
                d_ray = np.repeat(ray[1:][diag], 6)
                packed = np.concatenate([packed, (d_ray << shift) | d_keys])
        pair = np.unique(packed)
        ray_of, keys = pair >> shift, pair & self.KEY_MASK
        slot = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        found = self._keys[slot] == keys
        ray_of, slot = ray_of[found], slot[found]
        if len(slot) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        lens = self._starts[slot + 1] - self._starts[slot]
        pair_ray = np.repeat(ray_of, lens)
        idx = np.arange(lens.sum(), dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
        pair_tri = self._ids[np.repeat(self._starts[slot], lens) + idx]
        return pair_ray, pair_tri

    def _candidates(self, origins: np.ndarray, dirs: np.ndarray, max_dist: np.ndarray):
        """(ray index, triangle index) pairs worth testing."""
        n_rays = len(origins)
        if n_rays == 0 or self.count == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        if n_rays <= self.MAX_BATCH:
            return self._candidates_chunk(origins, dirs, max_dist)
        rays_out, tris_out = [], []
        for start in range(0, n_rays, self.MAX_BATCH):
            end = min(start + self.MAX_BATCH, n_rays)
            pr, pt = self._candidates_chunk(origins[start:end], dirs[start:end], max_dist[start:end])
            if len(pr):
                rays_out.append(pr + start)
                tris_out.append(pt)
        if not rays_out:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(rays_out), np.concatenate(tris_out)

    def _intersect_chunk(self, origins: np.ndarray, dirs: np.ndarray, max_dist: np.ndarray, tri_filter=None):
        """All (ray_idx, tri_idx, dist) intersections in a chunk, sorted by
        (ray_idx, dist) with duplicate (ray_idx, tri_idx) pairs removed."""
        empty_i = np.zeros(0, np.int64)
        empty_f = np.zeros(0, np.float64)
        r, t = self._candidates_chunk(origins, dirs, max_dist)
        if len(r) == 0:
            return empty_i, empty_i, empty_f
        if tri_filter is not None:
            keep = tri_filter(t)
            r, t = r[keep], t[keep]
            if len(r) == 0:
                return empty_i, empty_i, empty_f
        d = dirs[r]
        e1, e2 = self.e1[t], self.e2[t]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        e2x, e2y, e2z = e2[:, 0], e2[:, 1], e2[:, 2]
        px = dy * e2z - dz * e2y
        py = dz * e2x - dx * e2z
        pz = dx * e2y - dy * e2x
        det = e1[:, 0] * px + e1[:, 1] * py + e1[:, 2] * pz
        ok = np.abs(det) > 1e-12
        if not ok.any():
            return empty_i, empty_i, empty_f
        r, t = r[ok], t[ok]
        inv = 1.0 / det[ok]
        s = origins[r] - self.v0[t]
        u = (s[:, 0] * px[ok] + s[:, 1] * py[ok] + s[:, 2] * pz[ok]) * inv
        ok2 = (u >= 0.0) & (u <= 1.0)
        if not ok2.any():
            return empty_i, empty_i, empty_f
        r, t, inv, s, u = r[ok2], t[ok2], inv[ok2], s[ok2], u[ok2]
        e1 = self.e1[t]
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        e1x, e1y, e1z = e1[:, 0], e1[:, 1], e1[:, 2]
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        d = dirs[r]
        v = (d[:, 0] * qx + d[:, 1] * qy + d[:, 2] * qz) * inv
        ok3 = (v >= 0.0) & (u + v <= 1.0)
        if not ok3.any():
            return empty_i, empty_i, empty_f
        r, t, inv = r[ok3], t[ok3], inv[ok3]
        e2 = self.e2[t]
        tt = (e2[:, 0] * qx[ok3] + e2[:, 1] * qy[ok3] + e2[:, 2] * qz[ok3]) * inv
        good = (tt >= 0.0) & (tt <= max_dist[r])
        r, t, tt = r[good], t[good], tt[good]
        if len(r) == 0:
            return empty_i, empty_i, empty_f
        order = np.lexsort((t, tt, r))
        r, t, tt = r[order], t[order], tt[order]
        uniq = np.concatenate([[True], (r[1:] != r[:-1]) | (t[1:] != t[:-1])])
        return r[uniq], t[uniq], tt[uniq]

    def _cast_chunk(self, origins: np.ndarray, dirs: np.ndarray, max_dist: np.ndarray, tri_filter=None):
        dist = np.full(len(origins), np.inf)
        hit = np.full(len(origins), -1, dtype=np.int64)
        r, t, tt = self._intersect_chunk(origins, dirs, max_dist, tri_filter=tri_filter)
        if len(r) == 0:
            return dist, hit
        first = np.concatenate([[True], r[1:] != r[:-1]])
        dist[r[first]] = tt[first]
        hit[r[first]] = t[first]
        return dist, hit

    def cast_all(self, origins, dirs, max_dist, tri_filter=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All hits for each ray within max_dist, returned as parallel arrays
        `(ray_idx, tri_idx, dist)` sorted by `(ray_idx, dist)` with unique
        `(ray_idx, tri_idx)` pairs. `dirs` must be unit length."""
        origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
        max_dist = np.broadcast_to(np.asarray(max_dist, dtype=np.float64), (len(origins),)).copy()
        n_rays = len(origins)
        empty_i = np.zeros(0, np.int64)
        empty_f = np.zeros(0, np.float64)
        if self.count == 0 or n_rays == 0:
            return empty_i, empty_i, empty_f
        if n_rays <= self.MAX_BATCH:
            return self._intersect_chunk(origins, dirs, max_dist, tri_filter=tri_filter)
        rs, ts, ds = [], [], []
        for start in range(0, n_rays, self.MAX_BATCH):
            end = min(start + self.MAX_BATCH, n_rays)
            r, t, d = self._intersect_chunk(
                origins[start:end], dirs[start:end], max_dist[start:end], tri_filter=tri_filter
            )
            if len(r):
                rs.append(r + start)
                ts.append(t)
                ds.append(d)
        if not rs:
            return empty_i, empty_i, empty_f
        return np.concatenate(rs), np.concatenate(ts), np.concatenate(ds)

    def cast(self, origins, dirs, max_dist, tri_filter=None) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the nearest hit for each ray (inf if none within
        max_dist) and the triangle hit (-1 if none). `dirs` must be unit length.
        Optional `tri_filter(tri_ids) -> bool mask` skips triangles before
        intersection so rays search past rejected surfaces."""
        origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
        max_dist = np.broadcast_to(np.asarray(max_dist, dtype=np.float64), (len(origins),)).copy()
        n_rays = len(origins)
        if self.count == 0 or n_rays == 0:
            return np.full(n_rays, np.inf), np.full(n_rays, -1, dtype=np.int64)
        if n_rays <= self.MAX_BATCH:
            return self._cast_chunk(origins, dirs, max_dist, tri_filter=tri_filter)
        dist = np.full(n_rays, np.inf)
        hit = np.full(n_rays, -1, dtype=np.int64)
        for start in range(0, n_rays, self.MAX_BATCH):
            end = min(start + self.MAX_BATCH, n_rays)
            dist[start:end], hit[start:end] = self._cast_chunk(
                origins[start:end], dirs[start:end], max_dist[start:end], tri_filter=tri_filter
            )
        return dist, hit


def export_path(storage_dir: Path, uid: str) -> Path:
    return Path(storage_dir) / f"{uid}.tmgeo"


def latest_export(storage_dir: Path, map_name: str | None = None) -> Path | None:
    """The newest export in the plugin's storage folder, optionally the newest
    whose map name matches (formatting codes are already stripped)."""
    files = sorted(Path(storage_dir).glob("*.tmgeo"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files:
        if map_name is None:
            return f
        try:
            if read_header_name(f) == map_name:
                return f
        except (OSError, ValueError, struct.error):
            continue
    return None


def read_header_name(path: Path) -> str:
    r = _Reader(Path(path).read_bytes()[:4096])
    magic, _ = r.take("IH")
    if magic != MAGIC:
        raise ValueError("not a geometry file")
    r.string()
    return r.string()
