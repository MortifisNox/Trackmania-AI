# Lidar beams

The agent no longer looks at the camera picture. Instead, 20 times a second it
casts 19 beams across the game image, from a point just above the car, fanning
from straight left through straight ahead to straight right. Each beam stops at
the first dark pixels it meets, which on maps like tmrl-test is the track
border. The beam lengths (0 = border right there, 1 = clear to the edge of the
picture) join the telemetry and the racing line in the state vector.

This is the approach TMRL uses. It is lighter than a CNN on the GPU and the
agent learns from it faster, but it only works where the borders are clearly
darker than the road.

## 1. Check the beams on your map

Start the map, drive onto the road with the usual chase camera, then run:

```
tmai capture-test
```

Besides `runs\capture_test.png` it now saves `runs\lidar_test.png`, the game
image with the beams drawn in red and their end points in yellow, and prints
the 19 lengths. Each beam should end on a track border. If they don't:

- Beams all very short: the start point sits on the car or on something dark.
  Move it up, for example `tmai --set env.lidar_origin=[0.5,0.8] capture-test`.
- Beams pass through the borders: the borders are lighter than `lidar_dark`.
  Raise it, for example `tmai --set env.lidar_dark=90 capture-test`.
- Beams stop on the road: the road is darker than `lidar_dark`. Lower it.

Put the values that work into your own config file so every command uses them.

## 2. Record demo laps again

Old demo laps have no beam readings, so training skips them (the log says so).
Record a few new laps with `tmai record-demo`.

## 3. Start a fresh run

The inputs changed, so a checkpoint trained on the camera can't be resumed;
training refuses to load it and says why. Start under a new run name so the
old run stays as it is (or set `run.name: lidar` in your config file):

```
tmai --set run.name=lidar train
```

## Settings

| key | default | meaning |
| --- | --- | --- |
| `env.lidar_beams` | 19 | number of beams (0 turns the lidar off) |
| `env.lidar_origin` | [0.5, 0.9] | where the beams start, as x and y fractions of the picture |
| `env.lidar_dark` | 60 | grayscale level (0..255) below which a pixel counts as border |
| `env.lidar_width` | 256 | width in pixels of the frame the beams are measured on |
| `env.camera` | false | true adds the CNN over the camera frames back |
