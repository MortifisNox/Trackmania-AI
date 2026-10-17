# Plan

Each phase ends with something you can see working on your laptop. Design
details are in [DESIGN.md](DESIGN.md).

| # | Phase | Done when |
|---|---|---|
| 0 | **Setup** (this PR): repo skeleton, config, `tmai doctor`, setup and safety docs | `tmai doctor` is all green on your laptop |
| 1 | **Game I/O**: Openplanet telemetry plugin, virtual gamepad, screen capture, fixed 20 Hz loop, respawn/restart | A scripted driver goes forward, telemetry and frames stream at 20 Hz, resets work |
| 2 | **Track model and reward**: record your demo lap, build the reference line, progress reward, checkpoint splits, stuck/wrong-way detection | Driving manually shows correct progress, splits and resets |
| 3 | **Learner**: SAC + DroQ, demo-seeded replay, bf16, VRAM cap | Agent finishes a simple track on its own |
| 4 | **Dashboard and controls**: live telemetry, curves, pause/resume, live hyperparameters, manual reset | You can watch and steer a run from the browser |
| 5 | **Long-run robustness**: atomic checkpoints and replay snapshots, resume, game-crash watchdog, thermal governor, safe stop | A run survives a killed game, Ctrl+C and a restart without losing progress |
| 6 | **Continuous improvement and reporting**: self-improving reference line, evaluation laps, best-ghost handling, run report | Lap times keep dropping over days; stopping produces the report and best replay |
| 7 | **Arbitrary tracks**: multi-track training, per-track fine-tuning, track library | A new track goes from demo lap to competitive time without code changes |

## What happens next

1. You do [SETUP_WINDOWS.md](SETUP_WINDOWS.md) and paste `tmai doctor` output.
2. I build phase 1 (plugin + gamepad + capture) on a branch.
3. We test it together on your laptop: you run the commands, or I run them in
   the project folder on your laptop if you allow it.
