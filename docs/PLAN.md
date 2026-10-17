# Plan

Each phase ends with something you can see working on your laptop. Design
details are in [DESIGN.md](DESIGN.md).

| # | Phase | Done when |
|---|---|---|
| 0 | **Setup** (done): repo skeleton, config, `tmai doctor`, setup and safety docs | `tmai doctor` is all green on your laptop |
| 1 | **Game I/O** (done, see [PHASE1_TESTING.md](PHASE1_TESTING.md)): Openplanet telemetry plugin, virtual gamepad, screen capture, fixed 20 Hz loop, respawn/restart | A scripted driver goes forward, telemetry and frames stream at 20 Hz, resets work |
| 2 | **Track model and reward** (done, see [PHASE2_TESTING.md](PHASE2_TESTING.md)): record your demo lap, build the reference line, progress reward, checkpoint splits, stuck/wrong-way detection | Driving manually shows correct progress, splits and resets |
| 3 | **Learner** (done, see [PHASE3_TESTING.md](PHASE3_TESTING.md)): SAC + DroQ, demo-seeded replay, bf16, VRAM cap | Agent finishes a simple track on its own |
| 4 | **Dashboard and controls** (done, see [DASHBOARD.md](DASHBOARD.md)): live telemetry, curves, pause/resume, live hyperparameters, manual reset | You can watch and steer a run from the browser |
| 5 | **Long-run robustness** (partly done, see [RUN_LIFECYCLE.md](RUN_LIFECYCLE.md)): atomic checkpoints and replay snapshots, resume and safe stop are built; game-crash watchdog and thermal governor are not yet | A run survives a killed game, Ctrl+C and a restart without losing progress |
| 6 | **Continuous improvement and reporting** (mostly built): self-improving reference line, test laps, pace reward against recent laps, best-lap replay, run report | Lap times keep dropping over days; stopping produces the report and best replay |
| 7 | **Arbitrary tracks**: multi-track training, per-track fine-tuning, track library | A new track goes from demo lap to competitive time without code changes |

## Using it now

Day-to-day steps: [EVERYDAY_USE.md](EVERYDAY_USE.md). Problems:
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).
