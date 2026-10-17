# Stopping, resuming and the run report

Code: `src/tmai/run/`. The trainer uses it through one hook,
`tmai.run.shutdown.ShutdownHook`.

## Stopping a run safely

Any of these stops training cleanly:

- **Ctrl+C** in the training window (or Ctrl+Break, or SIGTERM). Pressing
  Ctrl+C again while it saves just reminds you it is saving; the third press
  abandons the save step in progress and quits. Even then nothing already
  saved is damaged, because every file is written to a temp name and renamed
  into place; you only lose what had not been saved yet.
- **The dashboard's stop button.**
- **`tmai stop`** from a second PowerShell window:

  ```powershell
  cd "D:\AI Projects\Trackmania AI"
  .\.venv\Scripts\Activate.ps1
  tmai stop
  ```

Closing the console window is not a safe stop: Windows ends the process after
a few seconds, so the final save may not finish. The last periodic checkpoint
and replay snapshot are still intact in that case.

On stop the run:

1. stops the driver and the learner and releases the virtual pad (trainer),
2. writes the final checkpoint and keeps a checksummed copy as `final.pt`
   (plus up to five `history-*.pt` copies),
3. writes a final replay buffer snapshot,
4. copies the best lap's replay next to the report,
5. builds the run report.

Each step runs even if an earlier one fails, and the result of each is listed
at the end of the report and in `control/status.json`.

## Resuming

`tmai train` again with the same `run.name` resumes from the last checkpoint
and refills the replay buffer from the last snapshot, so learning carries on
without a new warm-up. Snapshots are also taken every `replay.snapshot_every_min`
minutes in the background (the driver is paused for well under a millisecond
at a time). A damaged `latest.pt` is replaced by the newest intact copy before
loading, and a damaged snapshot falls back to the previous one.

## Files in `runs/<name>/`

| Path | What |
|---|---|
| `checkpoints/latest.pt` | weights and optimiser state the trainer resumes from |
| `checkpoints/best.pt`, `final.pt`, `history-*.pt` | checksummed copies; `best.pt` holds the weights that drove the best lap |
| `replay/snap-*/` | the last two replay buffer snapshots (~2 GB each at 500k entries) |
| `checkpoints/best_test_lap.pt` | the weights that drove the fastest test lap |
| `episodes.csv` | one row per attempt (trainer) |
| `eval.csv` | one row per test lap (every 10th attempt, driven without exploring) |
| `best_laps.jsonl` | every new best lap with its checkpoint splits |
| `control/stop.json` | a pending stop request |
| `control/status.json` | running / stopping (and which step) / stopped |
| `report/report.html` | the run report; opens offline in any browser |
| `report/summary.json` | the same numbers for scripts |
| `report/best-<time>ms.Replay.Gbx` | copy of the best lap's replay |

`tmai --set run.track=Test1 report` rebuilds the report at any time,
including during a run or after a crash. Pass the same `run.track` (and
`run.name`, if you set one) as the training run, so it finds the right demo
laps and replay.

## The run report

Best lap, the gap to your fastest demo lap, total environment steps, learner
updates and training hours, lap times over the run, a checkpoint-by-checkpoint
split and sector comparison against your demo, how attempts ended, and the
learning curves (reward, progress, critic and actor loss, Q-value, entropy
temperature, entropy) with a rolling mean, a ±1 standard deviation band and a
plain verdict per curve (stable, noisier than earlier, possible divergence,
NaN).

## The best lap as a replay in game

The **TMAI Ghost** Openplanet plugin (`openplanet/TMAI_Ghost`, installed by
`tmai install-plugin`) saves each of the agent's finishes that beats the
fastest one already saved for that map, to
`Documents\Trackmania\Replays\TMAI\<map name>\<time>ms-<date>.Replay.Gbx`.
It saves only the agent's laps: while training runs, tmai lists the agent's
finish times in `OpenplanetNext\PluginStorage\TMAI_Ghost\ai_driving.txt`,
and the plugin skips any ghost whose time is not on that list (your own laps,
personal-best and record ghosts).

To watch it: **Create > Replay editor**, open `TMAI > <map> > <file>`. To race
it as a ghost: **Play > Local > Against replay**. Keep it local: per
[SAFE_USE.md](SAFE_USE.md), never upload it or present it as a human run.

What is verified and what is not:

- *Verified (documentation):* replays live in `Documents\Trackmania\Replays`
  and open in the Replay editor
  ([Trackmania docs](https://doc.trackmania.com/play/watch-replays/)).
- *Verified (source of a public plugin, not run by us):* when a run finishes
  the game adds its ghost to
  `GetApp().Network.ClientManiaAppPlayground.DataFileMgr.Ghosts`, and
  `DataFileMgr.Replay_Save(path, map, ghost)` writes a one-ghost replay that
  plays via Play > Local > Against replay. This is what XertroV's
  [Autosave Ghosts](https://openplanet.dev/plugin/autosaveghosts)
  ([source](https://github.com/XertroV/tm-autosave-ghosts)) and
  [Archivist](https://openplanet.dev/plugin/archivist)
  ([source](https://github.com/XertroV/tm-archivist)) do. Both need Club
  access (`Permissions::CreateLocalReplay`), which you have.
- *Verified in game:* our plugin (version 0.2.0) saves the agent's best lap,
  and the safe stop copies it into the report
  (`best-20733ms.Replay.Gbx` on Test1).
- *Handled, watch for it:* after a personal best the game shows an
  end-of-race menu that ignores the restart button; tmai presses
  `env.menu_button` (A, which should pick **Improve**) to get past it.
- *If no replay is saved,* the stop message and the report say why (plugin
  not installed, old copy still loaded, no Club access, or no new best); see
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md#no-best-lap-replay-in-the-report).
- *Not done:* opening the replay in game automatically. Openplanet can start
  maps from a plugin, but I could not confirm a call that opens a replay, so
  the report gives the two menu paths instead.
