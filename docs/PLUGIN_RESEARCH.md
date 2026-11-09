# Growing the game state: what to add to the plugin

In `state` mode the AI drives only from what the TMAI Telemetry plugin sends,
so every useful field the plugin adds is something new the AI can see. This
page lists candidates, cheapest first, and the rules for finding new ones with
a reverse-engineering tool such as REA.

## Rules for reverse-engineering

These follow from [SAFE_USE.md](SAFE_USE.md) rule 5 (do not modify the game).

- **Read the result through Openplanet only.** A finding becomes a plugin field,
  read with Openplanet's API (`Dev::GetOffset*` on a class Openplanet already
  gives us, as the `VehicleState` dependency already does for RPM and side
  speed). Nothing outside Openplanet reads or writes the game's memory.
- **Static analysis only.** Look at `Trackmania.exe` and its files on disk, or
  at Openplanet's own Nod explorer (F3 > Developer > Nod explorer) while the
  game runs. Never attach a debugger, Frida, rizin or gdb to the running game:
  that is injection, which rule 5 rules out, and it is what anti-cheat looks for.
- **Offline, local maps, Developer mode**, as always.
- Ubisoft's terms generally forbid reverse-engineering their software. Keep
  the work private, as with the plugins (SAFE_USE.md rule 7). Do not publish
  offsets or tools.
- Offsets move with game updates. Every `Dev::` field needs a sanity check
  in the plugin (a value in a plausible range), and the plugin should send a
  flag when the check fails, so the AI never trains on garbage.

## Step 1: fields Openplanet already names (no REA needed)

`CSceneVehicleVisState` (what `VehicleState::ViewingPlayerState()` returns)
has more named fields than we send today. Adding them is a plugin and
protocol change, with no reverse-engineering:

| Field(s) | Why it helps |
|---|---|
| `FLSteerAngle` .. `RRSteerAngle`, `FLWheelRotSpeed` .. | how much the wheels actually turned, and wheel spin vs ground speed |
| `FLDamperLen` .. `RRDamperLen` | suspension compression: landing, bumps, how loaded each wheel is |
| `IsTopContact` | car on its roof |
| `IsTurbo`, `TurboTime`, `ReactorBoostLvl`, `ReactorBoostType` | turbo and reactor boost pads |
| `FLIcing01`, `WetnessValue01`, `WaterImmersionCoef` | ice, wet roads and water |
| `IsWheelsBurning` | burning-wheels state from special blocks |

Exact names differ a little between game versions; check them in the Nod
explorer before using them.

## Step 2: track geometry from the map (no REA needed)

The map's block and item lists are named in Openplanet
(`GetApp().RootMap.Blocks`, `.AnchoredObjects`: block model name, grid
coordinate, direction). A plugin can write them to a file once per map. Python
can then turn them into "what kind of road is around the car" features: road
type and surface (tarmac, dirt, ice, grass), walls or open edges, slopes and
loops, for the cells around and ahead of the car. This is the part of the
camera image that game state does not cover yet.

## Step 3: where REA helps

Things that are not named in Openplanet and are worth finding with static
analysis, then reading through `Dev::GetOffset*`:

1. **Angular velocity** of the car (yaw, pitch and roll rates). The physics
   state has it; `VisState` does not show it by name. It says whether a slide
   is getting worse before the position shows it.
2. **Wheel contact normals and per-wheel load**, for banked turns and walls.
3. **Distance to the nearest wall or road edge** in a few directions, if the
   game keeps such a value for its own camera or collision logic. This would
   be a clean replacement for the old pixel lidar.
4. **Block model geometry** (road edge positions inside each block model), to
   make step 2 exact rather than one 32 m cell per block.

For each finding, record the class, offset, type, the game version it was
found on, and a check that proves it (for example "matches `Dir` when the car
drives straight"), in a private note.
