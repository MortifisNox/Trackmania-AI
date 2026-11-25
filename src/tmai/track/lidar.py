"""The virtual lidar: how far the car can go in each direction before a wall or
the end of the road, cast against the map's collision geometry
(tmai.track.geometry) from the car pose the telemetry plugin already streams.

Two kinds of rays per direction, all in the car's own frame:
- one ray just above the car, parallel to the road, finds walls. A hit on a
  surface that faces up (a ramp ahead) is road, not a wall;
- support probes cast down onto the road at fixed distances along the same
  direction. The first probe that finds no road marks the edge (a drop).

Because the geometry is static and the pose is in every telemetry packet, a
scan can be recomputed for any recorded frame, demo laps included.

Also here: the check that the exported geometry lines up with recorded laps,
which fits the map's grid height and picks the placement conventions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from tmai.track.geometry import MapExport, Placement, RayCaster, WorldMesh, _parse_export, build_world

CLEAR, WALL, DROP = 0, 1, 2
FLAG_GROUND = 8   # telemetry flag: a wheel touches the ground
NO_MATERIAL = 80  # EPlugSurfaceMaterialId::XXX_Null: the wheel touches nothing
RIDE_MIN_M = -0.05  # min car height above a supporting road surface in grade()/Fit.ok()
RIDE_MAX_M = 1.0    # max car height above a supporting road surface in grade()/Fit.ok()
ANCHOR_ABOVE_M = 0.5  # max height above car center for the under-car support anchor (ignores overhead roofs)


@dataclass(frozen=True)
class LidarLayout:
    angles_deg: tuple[float, ...] = (-90, -60, -40, -25, -15, -8, -3, 0, 3, 8, 15, 25, 40, 60, 90)
    range_m: float = 60.0
    # Wall rays start this far above the car position, along car up: above the
    # 10 cm rubber curbs of curves (those are road edges, found by the probes).
    height_m: float = 1.0
    probes_m: tuple[float, ...] = (3, 6, 9, 12, 16, 20, 25, 30, 38, 46, 54)
    probe_above_m: float = 3.0      # support probes start at least this far above the car's road plane...
    probe_below_m: float = 4.0      # ...and look at least this far below it;
    probe_slope: float = 0.5        # farther out the window grows with distance, to follow slopes up to ~27 deg
    probe_step_m: float = 0.4       # max height step between probes beyond what the surface slope explains
    height_at_m: tuple[float, ...] = (9, 20, 38)   # probes whose road height (car frame) is a feature
    height_scale_m: float = 10.0
    road_normal_min: float = 0.6    # a hit whose normal . up is above this is road, not wall

    @property
    def rays(self) -> int:
        return len(self.angles_deg)

    @property
    def per_ray(self) -> int:
        """Features per direction: distance, is-wall, then one road height per height_at_m."""
        return 2 + len(self.height_at_m)


@dataclass
class Scan:
    dist: np.ndarray        # (rays,) metres to the wall or edge, range_m if clear
    kind: np.ndarray        # (rays,) CLEAR, WALL or DROP
    seconds: float          # time the scan took
    height: np.ndarray | None = None   # (rays, len(height_at_m)) road height above the car's road plane, nan if none

    def features(self, layout: LidarLayout) -> np.ndarray:
        """Per ray: distance / range (0..1), 1.0 for a wall (0.0 for an edge or
        clear), then the road height at each height_at_m distance divided by
        height_scale_m (-1..1; 0 where there is no road, which the distance
        already tells)."""
        d = np.clip(self.dist / layout.range_m, 0.0, 1.0)
        h = np.zeros((len(d), len(layout.height_at_m))) if self.height is None else self.height
        h = np.clip(np.nan_to_num(h / layout.height_scale_m, nan=0.0), -1.0, 1.0)
        out = np.concatenate([d[:, None], (self.kind == WALL)[:, None], h], axis=1)
        return out.reshape(-1).astype(np.float32)


def car_basis(forward, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (forward, left, up); `left` as in tmai.rl.features.car_frame."""
    u = np.asarray(up, dtype=np.float64)
    nu = np.linalg.norm(u)
    if not np.isfinite(nu) or nu < 1e-6:
        raise ValueError("car up vector is degenerate")
    u = u / nu
    f = np.asarray(forward, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError("car forward vector is not finite")
    f = f - (f @ u) * u
    nf = np.linalg.norm(f)
    if not np.isfinite(nf) or nf < 1e-6:
        raise ValueError("car forward and up are parallel")
    f /= nf
    return f, np.cross(u, f), u


class Lidar:
    def __init__(self, world: WorldMesh, layout: LidarLayout = LidarLayout(), uid: str = ""):
        self.world, self.layout = world, layout
        self.uid = str(uid)
        self.caster = RayCaster(world.tris)
        a = np.radians(np.asarray(layout.angles_deg, dtype=np.float64))
        self._cos, self._sin = np.cos(a), np.sin(a)
        self._probes = np.asarray(layout.probes_m, dtype=np.float64)
        self._dd = np.diff(np.concatenate([[0.0], self._probes]))
        self._above = np.maximum(layout.probe_above_m, layout.probe_slope * self._probes)
        self._below = np.maximum(layout.probe_below_m, layout.probe_slope * self._probes)
        missing = set(layout.height_at_m) - set(layout.probes_m)
        if missing:
            raise ValueError(f"height_at_m {sorted(missing)} must be among probes_m")
        self._height_idx = np.array([list(layout.probes_m).index(h) for h in layout.height_at_m], dtype=int)
        empty_ray = np.zeros(layout.per_ray, dtype=np.float32)
        empty_ray[0] = 1.0
        self._empty = np.tile(empty_ray, layout.rays)
        self._last: np.ndarray | None = None
        self.last_valid: bool = True

    def reset(self) -> None:
        """Reset any per-episode state before a new lap or episode."""
        self._last = None
        self.last_valid = True

    def scan(self, pos, forward, up) -> Scan:
        t0 = time.perf_counter()
        L = self.layout
        pos = np.asarray(pos, dtype=np.float64)
        if not np.all(np.isfinite(pos)):
            raise ValueError("car position is not finite")
        f, left, u = car_basis(forward, up)
        dirs = self._cos[:, None] * f + self._sin[:, None] * left          # (A, 3)
        n_a, n_p = len(dirs), len(self._probes)

        # Trackmania collision meshes are two-sided (triangle winding is not
        # guaranteed to point +car_up), so road-like orientation checks use
        # abs(normal . car_up). Directional slope -nd/nu is invariant to sign.
        road_filter = lambda t: np.abs(self.caster.normal[t] @ u) > L.road_normal_min

        # 1. Support probes: anchor directly under the car plus probes along
        # each direction. Cast all hits in the initial window so overhead
        # panels that fail continuity are skipped in favour of the road below.
        anchor_above = min(ANCHOR_ABOVE_M, L.probe_above_m)
        base = pos + dirs[:, None, :] * self._probes[None, :, None]        # (A, P, 3)
        probe_origins = np.vstack([
            (pos + u * anchor_above)[None, :],
            (base + u * L.probe_above_m).reshape(-1, 3),
        ])
        probe_ranges = np.full(len(probe_origins), L.probe_above_m + L.probe_below_m, dtype=np.float64)
        probe_ranges[0] = anchor_above + L.probe_below_m
        down = np.broadcast_to(-u, (len(probe_origins), 3))
        r_all, t_all, d_all = self.caster.cast_all(probe_origins, down, probe_ranges, tri_filter=road_filter)

        # Anchor under the car: highest road-like hit at or below anchor_above.
        has_anchor = False
        h_car = 0.0
        s_car = np.zeros(n_a, dtype=np.float64)
        anchor_end = int(np.searchsorted(r_all, 1))
        if anchor_end > 0:
            has_anchor = True
            h_car = anchor_above - float(d_all[0])
            n_car = self.caster.normal[t_all[0]]
            s_car = np.clip(-(dirs @ n_car) / (n_car @ u), -L.probe_slope, L.probe_slope)

        p_ray = r_all[anchor_end:] - 1
        p_tri = t_all[anchor_end:]
        p_h = L.probe_above_m - d_all[anchor_end:]
        if len(p_tri):
            n_hit = self.caster.normal[p_tri]
            p_s = np.clip(-(n_hit * dirs[p_ray // n_p]).sum(axis=1) / (n_hit @ u), -L.probe_slope, L.probe_slope)
        else:
            p_s = np.zeros(0, dtype=np.float64)
        starts = np.searchsorted(p_ray, np.arange(n_a * n_p + 1, dtype=np.int64))

        h = np.full((n_a, n_p), np.nan, dtype=np.float64)
        slope = np.zeros((n_a, n_p), dtype=np.float64)
        valid = np.zeros((n_a, n_p), dtype=bool)
        alive = np.ones(n_a, dtype=bool)
        prev_h = np.full(n_a, h_car, dtype=np.float64) if has_anchor else np.zeros(n_a, dtype=np.float64)
        prev_s = s_car.copy() if has_anchor else np.zeros(n_a, dtype=np.float64)

        def _pick_continuous(hc: np.ndarray, sc: np.ndarray, ph: float, ps: float, dd: float,
                             unanchored_first: bool) -> tuple[float, float] | None:
            if len(hc) == 0:
                return None
            if unanchored_first:
                lo = -L.probe_slope * dd - L.probe_step_m
                hi = L.probe_slope * dd + L.probe_step_m
            else:
                lo = np.minimum(ps, sc) * dd - L.probe_step_m
                hi = np.maximum(ps, sc) * dd + L.probe_step_m
            dh = hc - ph
            ok = (dh >= lo) & (dh <= hi)
            if not ok.any():
                return None
            idx = int(np.argmax(ok))
            return float(hc[idx]), float(sc[idx])

        for p in range(n_p):
            alive_idx = np.flatnonzero(alive)
            if len(alive_idx) == 0:
                break
            dd = float(self._dd[p])
            unanchored_first = (p == 0 and not has_anchor)
            max_dh = L.probe_slope * dd + L.probe_step_m
            retry_rays: list[int] = []
            retry_lo: list[float] = []
            retry_hi: list[float] = []

            for a in alive_idx:
                slot = int(a) * n_p + p
                s0, s1 = int(starts[slot]), int(starts[slot + 1])
                picked = _pick_continuous(p_h[s0:s1], p_s[s0:s1], float(prev_h[a]), float(prev_s[a]), dd, unanchored_first)
                if picked is not None:
                    h[a, p], slope[a, p] = picked
                    valid[a, p] = True
                    prev_h[a], prev_s[a] = picked
                else:
                    win_lo = float(prev_h[a]) - max_dh
                    win_hi = float(prev_h[a]) + max_dh
                    if win_lo < -L.probe_below_m or win_hi > L.probe_above_m:
                        retry_rays.append(int(a))
                        retry_lo.append(win_lo)
                        retry_hi.append(win_hi)
                    else:
                        alive[a] = False

            if retry_rays:
                rr = np.asarray(retry_rays, dtype=np.int64)
                w_lo = np.asarray(retry_lo, dtype=np.float64)
                w_hi = np.asarray(retry_hi, dtype=np.float64)
                r_origins = base[rr, p] + u * w_hi[:, None]
                r_ranges = w_hi - w_lo
                rr_hit, rt_hit, rd_hit = self.caster.cast_all(
                    r_origins, np.broadcast_to(-u, (len(rr), 3)), r_ranges, tri_filter=road_filter
                )
                r_starts = np.searchsorted(rr_hit, np.arange(len(rr) + 1, dtype=np.int64))
                if len(rt_hit):
                    rn = self.caster.normal[rt_hit]
                    rs = np.clip(-(rn * dirs[rr[rr_hit]]).sum(axis=1) / (rn @ u), -L.probe_slope, L.probe_slope)
                    rh = w_hi[rr_hit] - rd_hit
                else:
                    rs = np.zeros(0, dtype=np.float64)
                    rh = np.zeros(0, dtype=np.float64)
                for j, a in enumerate(retry_rays):
                    s0, s1 = int(r_starts[j]), int(r_starts[j + 1])
                    picked = _pick_continuous(rh[s0:s1], rs[s0:s1], float(prev_h[a]), float(prev_s[a]), dd, unanchored_first)
                    if picked is not None:
                        h[a, p], slope[a, p] = picked
                        valid[a, p] = True
                        prev_h[a], prev_s[a] = picked
                    else:
                        alive[a] = False

        missing = ~valid
        height = h[:, self._height_idx].copy()
        height[missing[:, self._height_idx]] = np.nan
        n_valid = valid.sum(axis=1)
        edge = np.where(n_valid < n_p, self._probes[np.minimum(n_valid, n_p - 1)], np.inf)

        # 2. Walls: one ray per direction at height_m above car. Skip a hit
        # only when it is road-like AND continues the drivable support surface
        # at that distance (within the same continuity tolerance as the probes).
        # Any hit strictly past `edge` cannot change dist/kind, so clamp ranges.
        origin = pos + u * L.height_m
        wall_ranges = np.minimum(edge, L.range_m)
        wr_all, wt_all, wd_all = self.caster.cast_all(np.repeat(origin[None], n_a, 0), dirs, wall_ranges)
        wall = np.full(n_a, np.inf, dtype=np.float64)
        if len(wr_all):
            w_starts = np.searchsorted(wr_all, np.arange(n_a + 1, dtype=np.int64))
            wn = self.caster.normal[wt_all]
            wnu = wn @ u
            w_road_like = np.abs(wnu) > L.road_normal_min
            h_hit = L.height_m
            step = L.probe_step_m

            for a in range(n_a):
                s0, s1 = int(w_starts[a]), int(w_starts[a + 1])
                if s0 == s1:
                    continue
                nv = int(n_valid[a])
                if has_anchor:
                    ds = np.concatenate([[0.0], self._probes[:nv]])
                    hs = np.concatenate([[h_car], h[a, :nv]])
                    ss = np.concatenate([[s_car[a]], slope[a, :nv]])
                else:
                    ds = self._probes[:nv]
                    hs = h[a, :nv]
                    ss = slope[a, :nv]
                n_sup = len(ds)
                dir_a = dirs[a]

                for k in range(s0, s1):
                    t_hit = float(wd_all[k])
                    if not w_road_like[k]:
                        wall[a] = t_hit
                        break
                    if n_sup == 0 or (nv < n_p and t_hit >= edge[a]):
                        wall[a] = t_hit
                        break
                    s_hit = float(np.clip(-(wn[k] @ dir_a) / wnu[k], -L.probe_slope, L.probe_slope))
                    pos_i = int(np.searchsorted(ds, t_hit))
                    continues = True
                    if pos_i > 0:
                        dd_l = t_hit - float(ds[pos_i - 1])
                        s_l = float(ss[pos_i - 1])
                        dh_l = h_hit - float(hs[pos_i - 1])
                        lo_l = min(s_l, s_hit) * dd_l - step
                        hi_l = max(s_l, s_hit) * dd_l + step
                        if dh_l < lo_l or dh_l > hi_l:
                            continues = False
                    else:
                        # Unanchored ray hitting before probes[0]: check from (0, 0).
                        if h_hit < -L.probe_slope * t_hit - step or h_hit > L.probe_slope * t_hit + step:
                            continues = False
                    if continues and pos_i < n_sup:
                        dd_r = float(ds[pos_i]) - t_hit
                        s_r = float(ss[pos_i])
                        dh_r = float(hs[pos_i]) - h_hit
                        lo_r = min(s_hit, s_r) * dd_r - step
                        hi_r = max(s_hit, s_r) * dd_r + step
                        if dh_r < lo_r or dh_r > hi_r:
                            continues = False
                    if not continues:
                        wall[a] = t_hit
                        break

        dist = np.minimum(np.minimum(wall, edge), L.range_m)
        kind = np.where(dist >= L.range_m, CLEAR, np.where(wall <= edge, WALL, DROP))
        return Scan(dist, kind, time.perf_counter() - t0, height)

    def features(self, tel) -> np.ndarray:
        """Observation features for one telemetry packet (see Scan.features).
        A pose the scan cannot use (e.g. zero orientation before live plugin
        data arrives) returns a fixed no-information scan rather than leaking a
        previous scan from another lap."""
        try:
            feats = self.scan(tel.pos, tel.dir, tel.up).features(self.layout)
        except ValueError:
            self.last_valid = False
            return self._empty.copy()
        self.last_valid = True
        self._last = feats
        return feats


# ---------------------------------------------------------------- lining up with laps

@dataclass
class Fit:
    placement: Placement
    on_surface: float          # share of on-ground lap samples with a surface just under the car
    ride_median_m: float       # median height of the car position above that surface
    ride_p90_m: float          # 90th percentile of |ride - median|
    samples: int
    triangles: int

    def ok(self) -> bool:
        return (
            self.samples > 0
            and self.on_surface >= 0.95
            and np.isfinite(self.ride_median_m)
            and np.isfinite(self.ride_p90_m)
            and RIDE_MIN_M <= self.ride_median_m <= RIDE_MAX_M
            and self.ride_p90_m <= 0.5
        )


def lap_samples(demo_files: list[Path], every: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Positions and up vectors of on-ground samples from recorded laps. Laps
    recorded before `flags` was saved use the wheel materials instead (at least
    two wheels on something); with neither, every sample counts."""
    pos, up = [], []
    for f in demo_files:
        with np.load(f) as d:
            if "flags" in d.files:
                ground = (d["flags"].astype(np.int64) & FLAG_GROUND) != 0
            elif "wheel_material" in d.files:
                ground = (np.asarray(d["wheel_material"]) != NO_MATERIAL).sum(axis=1) >= 2
            else:
                ground = np.ones(len(d["pos"]), dtype=bool)
            pos.append(d["pos"][ground][::every])
            up.append(d["up"][ground][::every])
    if not pos:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pos).astype(np.float64), np.concatenate(up).astype(np.float64)


def grade(world: WorldMesh, pos: np.ndarray, up: np.ndarray, placement: Placement,
          above: float = 0.25, below: float = 1.5,
          ride_min: float = RIDE_MIN_M, ride_max: float = RIDE_MAX_M) -> Fit:
    """Score how well `world` supports the recorded on-ground lap samples.

    A sample counts as supported only if a road-like triangle intersects its
    downward ray within the per-hit ride-height window `[max(-above, ride_min),
    min(below, ride_max)]`. Hits outside that window or on steep/vertical
    blockers are skipped during selection so the actual supporting road
    underneath is picked.

    Two-sided collision surfaces: Trackmania `GmSurfMesh` triangles collide
    from both sides and do not guarantee upward winding, so the road-normal
    test intentionally uses `abs(normal . car_up) > 0.5`.
    """
    n_samples = len(pos)
    if n_samples == 0:
        return Fit(placement, 0.0, float("nan"), float("nan"), 0, len(world.tris))
    caster = RayCaster(world.tris)
    u = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    lo_ride = max(-above, ride_min)
    hi_ride = min(below, ride_max)
    if hi_ride < lo_ride:
        return Fit(placement, 0.0, float("nan"), float("nan"), n_samples, len(world.tris))

    # Cast over the exact valid ride-height window [lo_ride, hi_ride], keeping
    # all hits so steep blockers in front of the road are skipped during selection.
    origins = pos - u * lo_ride
    r_hit, t_hit, d_hit = caster.cast_all(origins, -u, hi_ride - lo_ride)
    ride_all = np.full(n_samples, np.nan, dtype=np.float64)
    if len(r_hit):
        road_ok = np.abs((caster.normal[t_hit] * u[r_hit]).sum(axis=1)) > 0.5
        r_ok = r_hit[road_ok]
        d_ok = d_hit[road_ok]
        if len(r_ok):
            first = np.concatenate([[True], r_ok[1:] != r_ok[:-1]])
            ride_all[r_ok[first]] = lo_ride + d_ok[first]

    found = np.isfinite(ride_all)
    ride = ride_all[found]
    med = float(np.median(ride)) if found.any() else float("nan")
    p90 = float(np.percentile(np.abs(ride - med), 90)) if found.any() else float("nan")
    return Fit(placement, float(found.mean()), med, p90, n_samples, len(world.tris))


def fit_placement(export: MapExport, pos: np.ndarray, up: np.ndarray, log=print) -> tuple[Fit, list[Fit]]:
    """Try grid heights and placement conventions; keep what puts the recorded
    laps on the road. Grid height is tried in whole 8 m rows around -64 m."""
    tried: list[Fit] = []

    def score(p: Placement) -> Fit:
        fit = grade(build_world(export, p), pos, up, p)
        tried.append(fit)
        return fit

    def _rank(f: Fit) -> tuple[bool, float, float, float]:
        p90 = f.ride_p90_m if np.isfinite(f.ride_p90_m) else float("inf")
        med = abs(f.ride_median_m) if np.isfinite(f.ride_median_m) else float("inf")
        return (f.ok(), f.on_surface, -p90, -med)

    def better(a: Fit, b: Fit) -> bool:
        return _rank(a) > _rank(b)

    best = score(Placement())
    for k in range(-12, 13):
        if k == 0:
            continue
        cand = score(Placement(yoff=-64.0 + 8.0 * k))
        if better(cand, best):
            best = cand
    log(f"  grid height {best.placement.yoff:+.0f} m: {best.on_surface:.0%} of lap samples on a surface")
    for change in ({"mirror_dirs": True}, {"euler_sign": -1.0}, {"use_mobil_geom": False},
                   {"use_item_pivot": False}, {"quat_wxyz": False}):
        cand = score(replace(best.placement, **change))
        if better(cand, best) and ((cand.ok() and not best.ok()) or cand.on_surface > best.on_surface + 0.005):
            log(f"  {change}: {cand.on_surface:.0%} (was {best.on_surface:.0%}), keeping it")
            best = cand
    return best, tried


def export_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_fit(path: Path, export_file: Path, export: MapExport, fit: Fit,
             content_sha256: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = content_sha256 if content_sha256 is not None else export_sha256(export_file)
    data = {
        "export": str(export_file),
        "export_sha256": digest,
        "uid": export.uid,
        "map": export.name,
        "placement": asdict(fit.placement),
        "on_surface": fit.on_surface,
        "ride_median_m": fit.ride_median_m,
        "ride_p90_m": fit.ride_p90_m,
        "samples": fit.samples,
        "triangles": fit.triangles,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = json.dumps(data, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_fit(path: Path) -> tuple[Path, Placement]:
    data = json.loads(Path(path).read_text())
    return Path(data["export"]), Placement(**data["placement"])


def load_lidar(fit_path: Path, layout: LidarLayout = LidarLayout()) -> Lidar:
    fit_path = Path(fit_path)
    data = json.loads(fit_path.read_text())
    export_file = Path(data["export"])
    placement = Placement(**data["placement"])
    expected_hash = data.get("export_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            f"{fit_path} has no export content hash (export_sha256); re-run tmai geometry-check"
        )
    if not export_file.exists():
        raise FileNotFoundError(
            f"geometry export {export_file} is missing; re-run tmai geometry-check"
        )
    raw = export_file.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"geometry export {export_file} does not match the validated hash in {fit_path}; "
            "re-run tmai geometry-check"
        )
    export = _parse_export(raw, export_file)
    world = build_world(export, placement)
    if len(world.tris) == 0:
        raise ValueError(
            f"geometry export {export_file} has zero collision triangles; "
            "re-export the map and re-run tmai geometry-check"
        )
    uid = str(data.get("uid") or export.uid or "")
    return Lidar(world, layout, uid=uid)


def top_down_png(world: WorldMesh, path: Path, laps: list[np.ndarray] = (), size: int = 1600) -> Path | None:
    """A picture from above: road surfaces light, walls dark, laps in red."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if len(world.tris) == 0:
        return None
    xz = world.tris[:, :, [0, 2]]
    pts_all = [xz.reshape(-1, 2)] + [np.asarray(lap)[:, [0, 2]] for lap in laps if len(lap)]
    allp = np.concatenate(pts_all)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = (size - 20) / max(float((hi - lo).max()), 1.0)

    def px(p):
        q = (np.asarray(p) - lo) * scale + 10
        return [(float(x), float(size - y)) for x, y in q]

    img = Image.new("RGB", (size, size), (20, 22, 26))
    draw = ImageDraw.Draw(img)
    e1 = world.tris[:, 1] - world.tris[:, 0]
    e2 = world.tris[:, 2] - world.tris[:, 0]
    n = np.cross(e1, e2)
    ny = np.abs(n[:, 1]) / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    order = np.argsort(world.tris[:, :, 1].mean(axis=1))      # low surfaces first
    for i in order:
        colour = (120, 130, 140) if ny[i] > 0.6 else (220, 200, 90)
        draw.polygon(px(xz[i]), fill=colour if ny[i] > 0.6 else None, outline=colour)
    for lap in laps:
        if len(lap) > 1:
            draw.line(px(np.asarray(lap)[:, [0, 2]]), fill=(230, 60, 60), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
