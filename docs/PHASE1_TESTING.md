# Checking the game connection

Optional checks that Python can read the car's state, drive it with the
virtual controller, see the screen, and keep a steady 20 steps per second.
Useful the first time, or when something in the game connection seems off.
Day-to-day use is in [EVERYDAY_USE.md](EVERYDAY_USE.md).

## 1. Update the code

```powershell
cd "D:\AI Projects\Trackmania AI"
.\.venv\Scripts\Activate.ps1
git pull
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

In game: **F3 > Developer > (Re)load plugin**, then **TMAI Telemetry** and
**TMAI Ghost** (or restart the game). Open **F3 > Openplanet > Log**: you
should see `TMAI: waiting for tmai on port 9000` and
`TMAI Ghost: watching for the agent's finishes`. If you see red errors
instead, copy them into the project thread.

## 4. Load the map

Start the map from the game's **local play** menu (your My Maps folder), not
from the track editor's test drive: in the editor's test drive the race clock
and restarts behave differently. Keep the game windowed (1280x720) and not
minimised.

You can use your own map instead, as long as it is a local map file (see
SAFE_USE.md). If it has checkpoints, the default restart button **B** respawns
at the last checkpoint instead of the start. Bind **Give up** to **View/Back**
in the game's **Settings > Controls** (see
[SETUP_WINDOWS.md](SETUP_WINDOWS.md#7-game-settings)) and run the tests like
`tmai --set env.restart_button=BACK gamepad-test`.

## 5. Run the checks

Run each command, then click on the game window during the countdown so it has
focus. Paste all output in the project thread.

```powershell
tmai telemetry --seconds 10
tmai gamepad-test
tmai capture-test
tmai focus-test
tmai drive-test
```

- `telemetry` should print lines with speed, gear and race time changing while
  you drive with your own controller or keyboard.
- `gamepad-test` steers left and right, brakes, and restarts twice.
- `capture-test` counts only frames that changed, so drive around (or let
  someone drive) while it runs. It saves `runs\capture_test.png`, a blown-up copy of the 64x64
  picture the agent will see. Attach it in the thread if you can.
- `focus-test` asks you to click on another window first, then drives and
  captures while the game is in the background. It saves
  `runs\focus_test.png`, which should show the game even if another window
  covers it.
- `drive-test` drives straight at full throttle for 3 attempts and reports how
  steady the 20 Hz timing is. Crashing into a wall is expected.
