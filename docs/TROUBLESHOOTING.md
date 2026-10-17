# Troubleshooting

Fixes for problems that have actually come up. If yours is not here, copy the
last lines of the PowerShell window (and any red lines from
**F3 > Openplanet > Log**) into the project thread.

## Training stops: "respawned at a checkpoint"

```
B respawned at a checkpoint (race clock still at 12.3s) ...
```

The agent pressed **B**, which on maps with checkpoints puts the car back at
the last checkpoint instead of the start. Fix, once:

1. In Trackmania, **Settings > Controls** (gamepad), bind **Give up** to the
   **View/Back** button. That is the small button just left of the Xbox logo
   with two squares on it (Share/Create on a PlayStation pad), not the red B.
2. Always start training with `--set env.restart_button=BACK`:

   ```powershell
   tmai --set run.track=Test1 --set env.restart_button=BACK train
   ```

To check it: start the map, drive past a checkpoint, and run
`tmai --set env.restart_button=BACK gamepad-test`. The car should go back to
the start line with the timer at 0.

## Training stops: "restart did not happen after pressing ..."

- Use `--set env.restart_button=BACK` (see above).
- Make sure the game is running the map in local play, is not minimised, and
  is not sitting in a menu you opened yourself.
- After a personal best the game shows a menu (**Improve**, **Save replay**,
  ...). tmai gets past it by pressing **A**, which picks **Improve**. If it
  ever gets stuck on that menu, send a screenshot of it: the highlighted
  option may differ on your setup.

## No best-lap replay in the report

The stop message and the report say why. The usual causes:

- **Old plugin copy in the game.** `tmai install-plugin` was run before
  `git pull`, or the game was not reloaded afterwards. Run `git pull`, then
  `tmai install-plugin`, then **F3 > Developer > (Re)load plugin > TMAI Ghost**.
  The log must say `version 0.2.0` (or newer) and then
  `TMAI Ghost: watching for the agent's finishes`. Still the old version? Fully
  restart the game.
- **"the folder is named after the map's in-game name".** Replays are saved
  under the map's name as the game shows it. Use that name (case and spaces
  do not matter) as `run.track`.
- **"needs Club access".** The plugin can only save replays on a Club account.
- **No new best this run.** The plugin only saves laps faster than the best
  replay already saved for that map.

To see what the plugin last did:

```powershell
Get-Content $env:USERPROFILE\OpenplanetNext\PluginStorage\TMAI_Ghost\status.txt
```

Your replays are in `Documents\Trackmania\Replays\TMAI\<map name>\`. With
OneDrive, that Documents folder is under `C:\Users\<you>\OneDrive\Documents`.

## `git pull` refuses to run

If it says your local changes would be overwritten and you did not edit
anything yourself, those are only line-ending changes Windows made. Throw
them away and pull again:

```powershell
git checkout -- .
git pull
```

This discards any edits to files in the repo, so only do it if you did not
change files on purpose. Your `runs\` folder is not touched.

## `pip install` fails: "being used by another process" (WinError 32)

A `tmai` program is still running, so Windows will not let pip replace
`tmai.exe`. Close every window that runs `tmai` (training or dashboard). If you
cannot find one, end every `tmai.exe` and stray `python.exe` in Task Manager.
Then run `pip install -e ".[game,dashboard,dev]"` again.

`WARNING: Ignoring invalid distribution ~mai` is left over from an install
that was interrupted. It is harmless; delete the folder starting with `~mai`
in your Python's `Lib\site-packages` to make it go away.

## Packages went into the wrong Python

If pip's messages mention `...\AppData\Local\Programs\Python\Python311\...`,
the project's `.venv` was not active. Run `.\.venv\Scripts\Activate.ps1` first
(the prompt then starts with `(.venv)`) and do the install, training and
dashboard from windows where it is active.

## Dashboard is empty or does not open

- It must run in its own window: `tmai dashboard`, then open
  <http://127.0.0.1:8080>.
- If you used `--set run.name=...` for training, use the same one for the
  dashboard.
- An error like `No module named 'fastapi'`: run `pip install -e ".[game,dashboard,dev]"`.

## The agent drives oddly after a change

- Turn **ghosts off**, including your personal-best ghost. The agent learns
  from the screen and a second car confuses it.
- Keep the default chase camera and the same window size between runs.
- Do not switch the game to fullscreen or minimise it during training.

## Something crashed in the middle of training

Nothing that was already saved is damaged: every file is written to a temp
name first. Start `train` again and it resumes from the last checkpoint. To
get a report for the crashed run: `tmai --set run.track=Test1 report`.
