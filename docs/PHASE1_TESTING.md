# Phase 1: testing the game connection

This checks that Python can read the car's state, drive it with the virtual
controller, see the screen, and keep a steady 20 steps per second.

## 1. Update the code

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git fetch origin
git checkout -B main origin/main
pip install -e ".[game,dashboard,dev]"
```

## 2. Openplanet in Developer mode

Start Trackmania, press **F3**, then **Openplanet > Signature Mode > Developer**.
Developer mode is needed to load our own (unsigned) plugin, and it also turns
on School mode, so nothing you drive can reach official leaderboards.

## 3. Install the plugin and the map

```powershell
tmai install-plugin
tmai get-map
```

In game: **F3 > Developer > (Re)load plugin > TMAI Telemetry** (or restart the
game). Open **F3 > Openplanet > Log**: you should see
`TMAI: waiting for tmai on port 9000`. If you see red errors instead, copy them
into the project thread.

## 4. Load the map

**Create > Track Editor > Edit a Track > My Local Tracks > My Maps > tmrl-test**,
then press the green flag (bottom right) to drive it. Keep the game windowed
(1280x720) and fully on screen.

## 5. Run the checks

Run each command, then click on the game window during the countdown so it has
focus. Paste all output in the project thread.

```powershell
tmai telemetry --seconds 10
tmai gamepad-test
tmai capture-test
tmai drive-test
```

- `telemetry` should print lines with speed, gear and race time changing while
  you drive with your own controller or keyboard.
- `gamepad-test` steers left and right, brakes, and restarts twice.
- `capture-test` saves `runs\capture_test.png`, a blown-up copy of the 64x64
  picture the agent will see. Attach it in the thread if you can.
- `drive-test` drives straight at full throttle for 3 attempts and reports how
  steady the 20 Hz timing is. Crashing into a wall is expected.
