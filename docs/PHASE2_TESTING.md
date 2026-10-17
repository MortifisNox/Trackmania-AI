# Recording demo laps for a map

Do this once per new map, before training on it. You drive a few clean laps;
the fastest becomes the **reference line** the AI's progress is measured
along. Your laps are also kept to give the AI a head start: early in training
half of what it learns from comes from your laps, falling to a tenth over the
first few hours.

## 1. Update

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git pull
```

## 2. Name your track

Use the map's name as the game shows it (case and spaces do not matter), for
example `Test1`, with `--set run.track=NAME` on every command below and when
you train. The best-lap replay is found by that name. Data goes to
`runs\tracks\NAME\`.

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

This step is optional. Then train: [EVERYDAY_USE.md](EVERYDAY_USE.md). For a
map other than your first one, also give it its own run folder with
`--set run.name=NAME` (see
[EVERYDAY_USE.md](EVERYDAY_USE.md#training-on-a-new-map)).

If the line looks wrong, `tmai --set run.track=MyMap build-reference` rebuilds
it from your fastest saved demo lap.
