# Contributing

Thanks for helping. Bug reports, ideas, docs fixes and code are all welcome.

## Ground rules

- **Offline only.** The agent trains and drives on local maps, in Openplanet
  Developer (or School) mode, never online and never on leaderboards. Changes
  that would let it play online, submit records, touch the game's memory or
  change the game's speed will not be merged. See
  [docs/SAFE_USE.md](docs/SAFE_USE.md).
- Keep the 6 GB VRAM / 16 GB RAM laptop budget in mind
  ([src/tmai/budget.py](src/tmai/budget.py) checks it).
- Write docs for someone who is not a programmer: short sentences, the exact
  command to type, and what they should see.

## Dev setup

The game parts need Windows and Trackmania, but everything else (the RL code,
track model, run lifecycle and dashboard) runs and is tested on Linux or macOS
too.

```bash
git clone https://github.com/MortifisNox/Trackmania-AI.git
cd Trackmania-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,dashboard]"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel, see docs/SETUP_WINDOWS.md
pytest -q
```

Without PyTorch the trainer and end-to-end tests are skipped; CI installs CPU
PyTorch so they run. `tmai dashboard --demo` feeds the dashboard a fake trainer, and
the tests use fake game data, so you can work on most of the code without the
game open.

For the full Windows gaming setup see [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md).

## Code layout

| Path | What lives there |
|---|---|
| `src/tmai/cli.py` | the `tmai` command and its subcommands |
| `src/tmai/config.py`, `configs/default.yaml` | every setting, overridable with `--set section.key=value` |
| `src/tmai/game/` | game I/O: telemetry client, virtual gamepad, screen capture, episode restarts |
| `src/tmai/track/` | demo laps, reference line, progress, reward and splits |
| `src/tmai/rl/` | SAC + DroQ agent, networks, replay buffer, training loop |
| `src/tmai/run/` | crash-safe files, checkpoints, safe stop, run report, best-lap replay |
| `src/tmai/dashboard/` | FastAPI server and the browser dashboard (`static/index.html`) |
| `src/tmai/doctor.py`, `budget.py` | `tmai doctor` checks and the VRAM/RAM budget |
| `openplanet/` | the TMAI Telemetry and TMAI Ghost Openplanet plugins (AngelScript) |
| `tests/` | pytest suite |
| `docs/` | user guides and the design notes ([DESIGN.md](docs/DESIGN.md), [PLAN.md](docs/PLAN.md)) |

## Pull requests

1. Fork the repo (or branch, if you have access) and make your change on a
   branch off `main`.
2. Add or update tests for behaviour changes, and run `pytest -q`.
3. If you change a setting, a command or what the user sees, update the docs
   in the same PR.
4. If you change an Openplanet plugin, bump `version` in its `info.toml`.
5. Open the PR with the template filled in. Titles are plain sentences that
   say what changes for the user, like "Stop training when the restart button
   respawns at a checkpoint".
6. CI (`pytest` on Ubuntu) must be green. PRs are squash-merged.

If you tested on the real game, say which map and what you saw. That is the
most useful thing a reviewer can get, since CI cannot run Trackmania.

## Releases

Write the notes in `docs/releases/vX.Y.Z.md`, bump `version` in
`pyproject.toml` and `src/tmai/__init__.py`, merge, then push a tag
`vX.Y.Z` on `main`. The Release workflow publishes the GitHub release from
that notes file.

## Reporting bugs

Open an issue with the bug template. Include the `tmai` command you ran, the
output of `tmai doctor`, and the last lines of the console. Please remove
personal paths or account names before posting logs.
