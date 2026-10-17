# Trackmania-AI

An AI that teaches itself to drive a Trackmania (2020) map and keeps getting
faster for as long as you let it train. It runs on one Windows gaming laptop
(RTX 4050 Laptop GPU, 6 GB) next to the game.

You drive a few laps of a map yourself; the AI starts from your driving and
then learns from its own attempts, about 20 decisions per second. You watch it
in a browser dashboard, and when you stop it you get a report and the best
lap as an in-game replay.

## Start here

| I want to... | Read |
|---|---|
| set the laptop up for the first time | [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) |
| know the rules that keep my account safe (read before training) | [docs/SAFE_USE.md](docs/SAFE_USE.md) |
| update, train, watch and stop (day to day) | [docs/EVERYDAY_USE.md](docs/EVERYDAY_USE.md) |
| fix something that went wrong | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| use the dashboard | [docs/DASHBOARD.md](docs/DASHBOARD.md) |
| understand stopping, resuming, the report and the replay | [docs/RUN_LIFECYCLE.md](docs/RUN_LIFECYCLE.md) |
| record demo laps for a new map | [docs/PHASE2_TESTING.md](docs/PHASE2_TESTING.md) |
| see how it works and why | [docs/DESIGN.md](docs/DESIGN.md), [docs/PLAN.md](docs/PLAN.md) |

## Everyday use in short

In PowerShell, with no `tmai` window open:

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git pull
pip install -e ".[game,dashboard,dev]"
tmai install-plugin
```

In the game: **F3 > Developer > (Re)load plugin**, for **TMAI Telemetry**
and **TMAI Ghost**. Start your map from local play, ghosts off. Then:

```powershell
tmai --set run.track=Test1 --set env.restart_button=BACK train
```

Watch it with `tmai dashboard` in a second window, at
<http://127.0.0.1:8080>. Stop it with **Ctrl+C** (once, then wait) or the
dashboard's **Stop safely** button. The report is in
`runs\default\report\report.html`.

Always offline, on a local map, in Openplanet Developer mode. Never online.

## All commands

`tmai --help` lists them too. Any setting in
[configs/default.yaml](configs/default.yaml) can be changed for one command
with `--set section.key=value`.

| Command | What it does |
|---|---|
| `tmai doctor` | checks drivers, CUDA, RAM, Openplanet and disk |
| `tmai install-plugin` | copies the TMAI Telemetry and TMAI Ghost plugins into Openplanet |
| `tmai get-map` | downloads the tmrl-test map into My Maps |
| `tmai --set run.track=NAME record-demo` | records your laps of a map and builds its reference line |
| `tmai --set run.track=NAME progress-test` | you drive; it prints progress, reward and splits live |
| `tmai --set run.track=NAME --set env.restart_button=BACK train` | the AI trains on the map (Ctrl+C stops and saves) |
| `tmai dashboard` | the training dashboard at http://127.0.0.1:8080 |
| `tmai stop` | asks a running training to stop and save |
| `tmai --set run.track=NAME report` | rebuilds the run report from saved files |
| `tmai config` | prints every setting with its current value |
| `tmai telemetry` | prints live telemetry from the game |
| `tmai gamepad-test` | steers, brakes and restarts through the virtual pad |
| `tmai capture-test` | measures screen capture and saves what the AI sees |
| `tmai focus-test` | checks driving and capture while the game is in the background |
| `tmai drive-test` | drives full throttle at 20 Hz and reports timing |

## How it works, in one paragraph

The AI uses SAC (a reinforcement learning method) with a DroQ critic
ensemble. It sees four small grayscale frames of the game plus the car's
state from an Openplanet plugin, and drives through a virtual Xbox
controller. It is rewarded for moving along the track and for beating the
median of its own last 20 laps, so it keeps improving after it beats your
demo. Every 10th attempt is a test lap without experimenting. Details and the
alternatives that were rejected: [docs/DESIGN.md](docs/DESIGN.md).

## Development

```bash
pip install -e ".[dev,dashboard]"
pytest
```
