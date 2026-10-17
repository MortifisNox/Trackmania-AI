# Dashboard

A local web page for watching and steering a training run:
http://127.0.0.1:8080. It only listens on your own machine.

## Opening it

The dashboard runs in its own PowerShell window next to `tmai train`, so it
can be opened, closed or restarted at any time without touching training.

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
pip install -e ".[dashboard]"     # once
tmai dashboard
```

Then open http://127.0.0.1:8080 in your browser. Use the same `--set run.name=...`
(or `--config`) as the training window if you changed the run name; the
dashboard also follows whichever run connects to it.

To try it without the game, `tmai dashboard --demo` feeds it a fake trainer
(its files go to `runs/default-demo/`).

## What it shows

- Live: speed, gear, current lap time, track progress, steering, gas, brake,
  RPM, and this lap's checkpoint splits against the best lap.
- Best lap, episodes, driving steps and learner updates.
- Curves: episode reward with a 20-episode mean, finished lap times with the
  best so far, progress per episode, and every learner metric (critic and
  actor loss, Q value, temperature, entropy). Hover a chart for values.
- Run state: training, warming up, paused, or the steps of a safe stop.
  Once stopped, a **Run report** link opens `report.html`.

## Controls

| Button | What the trainer does |
|---|---|
| Pause | ends the current attempt, releases the pad and waits |
| Resume | carries on |
| Reset track | ends the current attempt and restarts the map |
| Save checkpoint | writes a checkpoint when the current attempt ends |
| Stop safely | same as `tmai stop`: final checkpoint, replay snapshot, report |
| Hyperparameters | learning rate, updates per step, entropy target and demo share change immediately; they are not saved into the config. Setting the entropy target or demo share turns off its automatic schedule for the rest of the run |

## History

Every learner metric and episode is also appended to `metrics.jsonl` and
`episodes.jsonl` in `runs/<name>/`, which is what the page loads on start, so
curves survive restarts. With torch installed and `dashboard.tensorboard: true`
the same values go to TensorBoard too:

```powershell
tensorboard --logdir runs
```

## How it connects (for developers)

`tmai.dashboard.link.DashboardHook` is a `tmai.run.hooks.TrainerHooks` that
`tmai train` adds after `ShutdownHook`. It sends telemetry, metrics and
episodes to the dashboard as JSON over localhost UDP (`dashboard.link_port`,
8081) and receives pause, resume, reset, save and set-parameter commands the
same way, so the 20 Hz control loop never waits on the web page. The stop
button writes `control/stop.json` through `tmai.run.stop.request_stop`, and the
page reads shutdown progress from `control/status.json`. The message format is
documented at the top of `src/tmai/dashboard/link.py`.
