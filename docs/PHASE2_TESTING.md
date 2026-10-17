# Phase 2: demo laps, reference line and reward

You drive a few clean laps; the fastest becomes the **reference line** the
agent's progress and lookahead are measured against. Your laps are also kept
to give the learner a head start in phase 3.

## 1. Update

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git fetch origin
git checkout -B main origin/main
```

## 2. Name your track

Pick a short name for your map (letters, digits, spaces, `-`, `_`) and use it
with `--set run.track=NAME` on every command below, for example
`--set run.track=MyMap`. Data goes to `runs\tracks\NAME\`.

## 3. Record demo laps

Start the map from local play, then:

```powershell
tmai --set run.track=MyMap record-demo --laps 3
```

Drive 3 complete laps with your own controller or keyboard. Restarting
mid-lap discards that attempt. Drive cleanly rather than at your absolute
limit: the line only needs to follow the track. When the 3rd lap finishes,
the reference line is built from the fastest one.

## 4. Watch progress and reward live

```powershell
tmai --set run.track=MyMap progress-test --seconds 60
```

Drive the map again. Twice a second it prints progress along the line, how
far you are from it, the reward the agent would get, and the next lookahead
point. At each checkpoint it prints your split against your best demo lap.
Driving backwards ends the attempt with `wrong_way`.

Paste the output in the project thread.
