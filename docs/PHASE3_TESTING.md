# Phase 3: the agent learns your map (done)

Phase 3 is finished: on Test1 the agent finishes the map on its own and its
best laps (20.55 s) beat the 20.83 s demo lap.

How to run training now is in [EVERYDAY_USE.md](EVERYDAY_USE.md): update,
reinstall the plugins, start, watch, stop. In short:

```powershell
tmai --set run.track=Test1 --set env.restart_button=BACK train
```

What the log lines mean is in
[EVERYDAY_USE.md](EVERYDAY_USE.md#reading-the-training-log), and fixes for
problems that came up while testing are in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).
