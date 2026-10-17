# Everyday use

The short version of how to update, train, watch and stop. First time on this
laptop? Do [SETUP_WINDOWS.md](SETUP_WINDOWS.md) first. Something went wrong?
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Always train **offline, on a local map, in Openplanet Developer mode**. Never
online. The full rules are in [SAFE_USE.md](SAFE_USE.md).

## One click

Double-click **`start_training.bat`** in the repo folder. It:

1. updates the code (`git pull`, and `pip install` when something new came in),
2. reinstalls the game plugins,
3. lists the in-game checks and waits for you to press Enter,
4. records your demo laps first if this map has none yet,
5. starts the dashboard in its own window and opens it in your browser,
6. starts training.

The first time, it creates **`tmai_settings.json`** from
[tmai_settings.example.json](../tmai_settings.example.json) and opens it in
Notepad. Write your map in it, save, and double-click again. The settings
you will usually touch:

| Setting | What to write |
|---|---|
| `map_name` | the map's name as the game shows it, e.g. `"Test1"` |
| `run_name` | the folder under `runs\` for this training; leave `""` to use the map name, so every map keeps its own run. Your Test1 training from before the one-click start is in `runs\default`: write `"default"` here to continue it |
| `map_has_checkpoints` | `true` or `false` |
| `restart_button` | `"auto"` picks `BACK` (Give up, which you bind to View/Back in **Settings > Controls**) on maps with checkpoints and `B` on maps without; or name a button: `B`, `BACK`, `START`, `X`, `Y`, `A`, `LB`, `RB`, `DPAD_UP`... |
| `fresh_start` | `true` once to start the run over; set it back to `false` after |
| `train_minutes` | a number to stop by itself after that long, `null` to train until you stop it |
| `extra_settings` | any other setting from [configs/default.yaml](../configs/default.yaml), e.g. `{"train.batch_size": 128}` |

The rest (skipping the update, the plugin copy, the Enter prompt, the
dashboard or the browser) are explained in the file's `_help` section. The
file is yours: git ignores it, so `git pull` never overwrites it. A typo in
a setting name or value stops the start with a message saying which one.

Stopping is the same as below: **Ctrl+C** once in the training window, or
**Stop safely** on the dashboard. The results are in `runs\<run_name>\`.

The same settings work with any command through `--settings`, for example
`tmai --settings tmai_settings.json stop` or
`tmai --settings tmai_settings.json report`.

## The five steps by hand

What `start_training.bat` does, if you prefer to type it.

### 1. Get the latest code

Close every window that is running `tmai` (training or dashboard) first, then
open PowerShell:

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git pull
pip install -e ".[game,dashboard,dev]"
```

The prompt should start with `(.venv)` after the second line. If `git pull`
refuses because of local changes you did not make, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#git-pull-refuses-to-run).

### 2. Reinstall the game plugins

Do this after every `git pull`, because the plugins in the game are copies and
do not update by themselves:

```powershell
tmai install-plugin
```

Then in the game press **F3**, open **Developer > (Re)load plugin** and click
**TMAI Telemetry**, then do the same for **TMAI Ghost** (or simply restart the
game). Open **F3 > Openplanet > Log** and check for:

- `Loaded plugin 'TMAI_Ghost' (version 0.2.0)` (or newer), followed by
  `TMAI Ghost: watching for the agent's finishes`
- `TMAI: waiting for tmai on port 9000` from the telemetry plugin

### 3. Start training

In the game: Developer mode on, **ghosts off** (your own personal-best ghost
too), the map started from **local play** (not the editor's test drive), the
game windowed and not minimised. Then:

```powershell
tmai --set run.track=Test1 --set env.restart_button=BACK train
```

- `run.track=Test1` is the name you recorded your demo laps under.
- `env.restart_button=BACK` makes the agent restart with the **View/Back**
  button, which you bound to **Give up** in the game's controls. Without it
  the agent presses B, which on maps with checkpoints only respawns at the
  last checkpoint, and training stops with an error.

Training picks up where the last run stopped. Add `--fresh` to start over,
or `--minutes 60` to stop by itself after an hour. You can click away from
the game; it keeps driving in the background.

### 4. Watch it

Open a **second** PowerShell window:

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
tmai dashboard
```

and open <http://127.0.0.1:8080> in your browser. It shows the live car,
lap times, splits and learning curves, and has buttons to pause, resume,
reset the track, save, change settings and stop. More in
[DASHBOARD.md](DASHBOARD.md). Closing the dashboard never affects training.

### 5. Stop it safely

Any one of these:

- press **Ctrl+C** once in the training window and wait,
- click **Stop safely** on the dashboard,
- or run `tmai stop` in the second window.

Wait until it prints `report: runs\default\report\report.html`. Do **not**
just close the training window; that can cut the final save short.

When it is done you have:

- `runs\default\report\report.html`: best lap, splits against your demo,
  total steps and the learning curves. Opens in any browser.
- `runs\default\report\best-<time>ms.Replay.Gbx`: the best lap as a replay.
  In game: **Create > Replay editor**, open **TMAI > (map name)**, or race it
  with **Play > Local > Against replay**. Keep it to yourself; never upload it
  as a human run.

## Reading the training log

One line per attempt:

```
ep  2176  finished   20.84s  progress 100.0%  reward   172.9  1180538 updates  alpha 0.003  late p99 1 ms  lap 20.840s  baseline 21.06s
```

| Part | Meaning |
|---|---|
| `finished` | how the attempt ended: `finished`, `stuck`, `timeout` or `wrong_way` |
| `20.84s` | race time of the attempt |
| `progress` | how far along the track it got |
| `reward` | the score the agent gave itself for this attempt; higher is better |
| `updates` | how many learning steps so far |
| `alpha` | how much it still experiments; falls over time |
| `late p99` | control timing; under 25 ms is fine |
| `lap` | lap time, only when it finished |
| `NEW BEST` | the fastest lap so far |
| `(test lap)` | every 10th attempt drives without experimenting, to measure real pace |
| `baseline` | the typical (median) lap of its last 20 laps; it is rewarded for beating this. Shows once 5 laps are finished |
| `STALL` | the race clock ran while the agent missed driving steps (a hitch); occasional ones are harmless |

The first ~1000 steps say `warming up`: it drives almost randomly before
learning starts. Test laps are also written to `runs\default\eval.csv`, and
every attempt to `runs\default\episodes.csv`.

## Training on a new map

1. Put the map in your **My Maps** folder and start it from local play.
2. Use the map's name as the game shows it, for example `Map2` (case and
   spaces do not matter to tmai, but a name with spaces needs quotes:
   `--set "run.track=My Map"`). The best-lap replay is found by this name.
3. Record 3 clean laps yourself:

   ```powershell
   tmai --set run.track=Map2 record-demo --laps 3
   ```

4. Train it in its own run folder, so it does not continue the Test1 run:

   ```powershell
   tmai --set run.track=Map2 --set run.name=Map2 --set env.restart_button=BACK train
   ```

   Use the same `--set run.name=Map2` for `tmai dashboard`, `tmai stop` and
   `tmai report`; that run's files then go to `runs\Map2\` instead of
   `runs\default\`.

More detail on demo laps: [PHASE2_TESTING.md](PHASE2_TESTING.md).

## Other handy commands

| Command | What it does |
|---|---|
| `tmai doctor` | checks drivers, GPU, RAM, Openplanet and disk |
| `tmai --set run.track=Test1 report` | rebuilds the run report at any time, even during a run |
| `tmai config` | prints every setting with its current value |
| `tmai telemetry` | prints live speed, gear and race time from the game |
| `tmai gamepad-test` | steers, brakes and restarts through the virtual pad |
| `tmai --help` | lists every command |

Any setting in [configs/default.yaml](../configs/default.yaml) can be changed
for one command with `--set section.key=value`, as above.
