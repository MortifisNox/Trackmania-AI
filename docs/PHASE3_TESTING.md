# Phase 3: the agent learns your map

The agent drives your map by itself, 20 decisions per second, and learns from
every run. Half of each training batch comes from your demo runs, so it starts
from your driving instead of from scratch.

## 1. Update

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git fetch origin
git checkout -B main origin/main
pip install -e ".[game,dashboard,dev]"
```

## 2. Before you start

- Play offline, on your local map file, in Developer/School mode (see
  [SAFE_USE.md](SAFE_USE.md)). Never train online.
- Turn **ghosts off** (your personal-best ghost too): the agent learns from the
  screen and a second car confuses it.
- Load the map from local play, not the editor's test drive.
- Use the same track name you recorded with, e.g. `Test1`.

## 3. Train

```powershell
tmai --set run.track=Test1 train
```

The game can stay in the background (not minimised). One line prints per run:

```
ep    12  finished   24.31s  progress 100.0%  reward   171.2  3120 updates  alpha 0.084  late p99 2 ms  lap 24.310s  NEW BEST
```

- `finished`, `stuck`, `timeout` or `wrong_way`: how the run ended.
- `progress`: how far along your line it got. This should climb first.
- `warming up` for the first ~1000 steps (about a minute): it drives
  randomly-ish before learning starts.
- `late p99`: control timing. Under 25 ms is fine.

Press **Ctrl+C** to stop. It saves `runs\default\checkpoints\latest.pt`, and
the next `train` continues from there (`--fresh` starts over). Every run is
also logged to `runs\default\episodes.csv`. `--minutes 30` stops by itself.

## 4. What to send back

After 30 to 60 minutes, paste the last ~20 lines. Phase 3 is done when it
finishes the map on its own. Expect wobbly, slow driving at first; the time
keeps dropping in later phases.
