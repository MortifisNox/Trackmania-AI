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

Phase 0 (setup). Available commands:

```powershell
tmai doctor   # check drivers, CUDA, RAM, Openplanet, disk
tmai config   # print the resolved configuration
tmai --set train.batch_size=128 config   # override any value
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
