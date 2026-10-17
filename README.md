# Trackmania-AI

A driving agent for Trackmania (2020) that trains continuously on a track and
keeps lowering its lap time, sized for an RTX 4050 Laptop GPU (6 GB) running
the game on the same machine.

- **Method**: SAC with a DroQ critic ensemble, seeded with your demo laps,
  reading telemetry from an Openplanet plugin and screen frames, driving
  through a virtual Xbox controller. Why: [docs/DESIGN.md](docs/DESIGN.md).
- **Plan**: [docs/PLAN.md](docs/PLAN.md)
- **Laptop setup**: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)
- **Avoiding bans**: [docs/SAFE_USE.md](docs/SAFE_USE.md). Read before training.

## Status

Phase 1 (game connection) in testing: [docs/PHASE1_TESTING.md](docs/PHASE1_TESTING.md).

```powershell
tmai doctor           # check drivers, CUDA, RAM, Openplanet, disk
tmai install-plugin   # copy the TMAI Telemetry plugin into Openplanet
tmai get-map          # download the tmrl-test map into My Maps
tmai telemetry        # print live telemetry
tmai gamepad-test     # steer, brake, restart through the virtual pad
tmai capture-test     # capture rate + image of what the agent sees
tmai focus-test       # drive and capture while the game is in the background
tmai drive-test       # full-throttle loop at 20 Hz with timing report
tmai --set env.restart_button=BACK config   # override any value
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
