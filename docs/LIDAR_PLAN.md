# Virtual lidar: plan and status

## Decision (2026-11-25)

The car senses its surroundings with a **virtual lidar computed in Python**:

1. The **TMAI Map Geometry** plugin (`openplanet/TMAI_MapGeometry`) reads the
   map's collision mesh: the triangles the car actually hits. It writes them to
   `OpenplanetNext/PluginStorage/TMAI_MapGeometry/<map uid>.tmgeo`, once per
   map, when a race on the map starts.
2. Python (`src/tmai/track/geometry.py`) places those triangles in the world
   and casts rays against them from the car pose that TMAI Telemetry already
   streams. The lidar layout is in `src/tmai/track/lidar.py`.

This replaces the earlier plan to call a raycast inside the game, which is
recorded below. That plan stalled: Openplanet exposes no raycast, and calling
an engine function would break SAFE_USE rule 5. The collision mesh, however,
is exposed by name:

- `CPlugSurface.m_GmSurf` leads to `GmSurfMesh.m_Verts` and `m_Tris`. Every
  triangle carries its physics material (`EPlugSurfaceMaterialId`, with
  `NotCollidable` = 28).
- Placed blocks reach that surface through block info → variant → mobil →
  `PrefabFid` (a `CPlugPrefab`) → `Ents[].Model` (`CPlugStaticObjectModel.Shape`).
- Placed items reach it through `ItemModel.EntityModel`.

The open-source Trigger Visualizer plugin (Unlicense) already walks this path
while a map is being driven. Its placement maths, and the measured grid
placement in vjeux/trackmania-tas `tools/mapgeom`, guided ours.

What this design buys:

- **No game-side cost per frame, and no freshness problem.** The geometry is
  static, so each scan uses the same pose packet as the rest of the observation.
- **Demo laps get lidar too.** Scans are recomputed from each recorded pose, so
  the demos you already drove can be used.
- **Checkable without the game.** The plugin writes raw placements (grid
  coordinate and direction, or position and angles) and the chain of local
  transforms under each block. All placement maths is in Python.
  `tmai geometry-check` fits it to the demo laps: on-ground samples must have
  a road surface just under the car.

Project rules still hold. The plugin is read-only and offline-only, and it
writes only to its own storage folder. It makes two `Dev::GetOffsetVec3` reads,
both sanity-checked: free-block position and rotation at `CGameCtnBlock.Dir`
+0x8 and +0x14, and the item pivot at `AbsolutePositionInMap` +0x30. These are
allowed under PLUGIN_RESEARCH.md's rules.

## The lidar layout (version 1)

- **Directions:** 15, in the car's own frame: -90, -60, -40, -25, -15, -8, -3,
  0, 3, 8, 15, 25, 40, 60 and 90 degrees from straight ahead.
- **Range:** 60 m.
- **Walls:** one ray per direction, 1.0 m above the car position, above the
  10 cm rubber curbs of curves (at 0.5 m, 74 of 2,574 wall hits on a
  Fall 2026 - 01 lap were curbs the car drives over; at 1.0 m, 4). A hit on a
  surface facing up (normal·up > 0.6), such as a ramp ahead, counts as road,
  not as a wall.
- **Road edges and slopes:** support probes at 3, 6, 9, 12, 16, 20, 25, 30,
  38, 46 and 54 m along each direction, cast down along the car's own up.
  Each probe first looks from 3 m above the car's road plane to 4 m below it,
  which finds the road under overhead gates (their bars sit about 8 m up).
  Only a probe that finds nothing there looks again, from max(3 m,
  0.5 × distance) above to max(4 m, 0.5 × distance) below, so slopes up to
  about 27° stay "road". The first probe that finds no surface marks the edge.
- **Features:** per direction:
  - the distance divided by the range;
  - 1.0 for a wall;
  - the road height above the car's road plane at 9, 20 and 38 m, divided by
    10 m and clipped to -1..1 (0 where there is no road).

  That makes 75 floats (`LIDAR_DIM`), added after the attitude in state mode
  when `env.lidar: true`. On Fall 2026 - 01, the straight-ahead height at
  38 m ranges from -18 m to +17 m along a lap, so the slopes are real
  information.

## Status

| Step | State |
|---|---|
| Plugin exporter | Works in game (v0.1.5). Fall 2026 - 01: 4825 blocks, 220 items, 683k triangles, exported in 4.4 s spread over frames |
| Python reader, placement, ray caster, lidar, lap fit | Done, unit-tested on hand-built maps |
| `tmai geometry-check`: fit to demo laps, `geometry.json`, top-down picture | Done |
| `tmai lidar-test`: live readout while you drive | Done |
| `env.lidar` in observations, demos and checkpoints | Done; off by default |
| Live test drive on Fall 2026 - 01 (180 s, 10 restarts) | Done: 1801 scans 0.1 s apart, none late; 4 of 6051 walls on the path driven next; road height at 20 m within 0.15 m for 90% of 1015 checks (the rest were jumps, plus 1 case off by about 1 m); scan 5 ms median, 9.5 ms p95 |
| First real map lined up (Fall 2026 - 01) | Done: 100% of lap samples on the road, car 0.04 m above it, grid height -120 m |
| Learning comparison: state vs state + lidar | After the export checks out |

## Next steps, in order

1. **Install and export.** Run `tmai install-plugin`, then reload TMAI Map
   Geometry in game. Start Test1 from local play, then check
   F3 > Openplanet > Log for the two `TMAI geometry:` lines. If the plugin
   fails to compile, the log names the line.
2. **Line it up.** Run `tmai --set run.track=Test1 geometry-check`. It should
   report OK, meaning at least 95% of lap samples have a surface under the car
   and 90% of samples are within 0.5 m of the median ride height. Look at
   `runs/tracks/Test1/geometry.png`: the red laps should lie on grey road.
3. **Live check.** Run `tmai --set run.track=Test1 lidar-test` and drive along
   a wall and up to an edge. The numbers should shrink as you approach. Note
   the scan time.
4. **Baseline:** a long training run on Test1 without lidar. Record its lap
   time, finish rate and crash/stuck rate.
5. **Track-ahead features** from the same geometry: room to the left and right
   and the surface type at each reference point ahead, so the AI can plan
   turns and grip changes, not only react to the walls next to it.
6. **Comparison:** the same training time on Test1 with the new features, under
   a new `run_name`. Keep only what makes laps faster or finishes more
   reliable.

Scope for now is one map (Test1). Everything is computed from that map's own
geometry, so the same features can later carry over to training on many maps.

## Known limits

- **Assumed conventions.** The block variant and mobil choice, the Euler
  order and the item pivot follow the plugins named above.
  `geometry-check` tries the alternatives and keeps whichever fits the laps
  best. Maps with many rotated free blocks or items may still need work.
- **Approximated and skipped shapes.** Convex hulls and boxes are exported as
  their bounding boxes. The stadium decoration (outer grass floor) is not
  exported; on maps that use it, the car can read a drop where the grass is.
- **Moving items** are exported in their resting pose.
- **Loops and wallrides.** Probes are cast along the car's own down
  direction, so walls and loops are handled, but nothing has been measured on
  them yet.
- **Sparse rays.** The 15 directions can miss thin posts between rays.
- **Speed.** On Fall 2026 - 01 (683k triangles) a scan takes about 5 ms
  (95% under 10 ms), measured on every frame of the demo laps. Triangles are
  bucketed in 4 m cubes, and the lookup takes every cube a ray passes
  through, so the result does not depend on the cube size.

## Earlier approach: engine raycast (superseded)

Kept for the record. Inspection of the installed `Openplanet.h` found
`NHmsCollision_SMgr` (counts and `UpdateStatic()` only) and `CHmsPicker`
(read-only and driven by the mouse cursor). Neither offers an arbitrary
raycast. `OpenplanetCore.json` has no ray or pick functions. Two REA
deep-analysis attempts on `Trackmania.exe` (SHA-256
`3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda`) timed
out before a result. Invoking an engine function found that way would have
needed `Dev::` calls that SAFE_USE rule 5 rules out.

That work also found that observations could reuse an old telemetry packet,
for example 30 s old after the game disconnected. This is fixed (2026-11-25).
Each step now acts only on a packet received within
`env.max_telemetry_age_s` (0.1 s). If none arrives, the controls are released
before the step times out, and a disconnect clears the last packet.
