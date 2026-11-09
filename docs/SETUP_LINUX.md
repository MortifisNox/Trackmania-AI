# Setting up on Linux

One-time setup for Linux (written on Fedora; the Ubuntu or Debian package
names are given where they differ). Do the steps in order. Step 10 checks
the result. Read [SAFE_USE.md](SAFE_USE.md) before you train. After setup,
the day-to-day steps are in [EVERYDAY_USE.md](EVERYDAY_USE.md), using
`./start_training.sh` where they say `start_training.bat`.

## What is different from Windows

- **The AI drives from game state, not the screen.** Window capture only
  exists on Windows, so on Linux `observation` is `state`: the TMAI Telemetry
  plugin's data (speed, gears, velocity, wheel surfaces and slip, pitch and
  roll) plus 20 points of your racing line reaching about 290 m ahead. No
  camera, no CNN. `tmai_settings.json` picks this for you (`"observation": "auto"`).
- **The game runs under Proton** (Steam), so Openplanet and the game's
  Documents folder live inside the Proton prefix:
  `~/.steam/steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/`.
  `tmai` finds it by itself; set `TMAI_GAME_USER_DIR` to that `steamuser`
  folder if it does not.
- **The virtual gamepad uses `/dev/uinput`** instead of the ViGEmBus driver.

## 1. NVIDIA driver

Fedora: enable RPM Fusion (free and nonfree), then:

```bash
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
```

Wait a few minutes for the kernel module to build, reboot, then run
`nvidia-smi`. It should list the RTX 4050 Laptop GPU. (Ubuntu: Software &
Updates > Additional Drivers, pick the newest proprietary driver.)

## 2. Steam and Trackmania

1. Install Steam (`sudo dnf install steam` with RPM Fusion nonfree, or the
   Flathub version).
2. Steam > Settings > Compatibility: turn on Steam Play for all titles, using
   **Proton Experimental**.
3. Install **Trackmania** (free on Steam). Start it once, log in to your
   Ubisoft account (the one with Club) in the Ubisoft Connect window that
   opens, and reach the main menu. Then quit.
4. Trackmania > Properties > **Controller**: pick **Disable Steam Input**, so
   the game reads the AI's virtual pad directly.
5. Install **protontricks** (`sudo dnf install protontricks`, or the
   Flathub `com.github.Matoking.protontricks`).

## 3. Openplanet

1. Download the Openplanet installer for Trackmania from
   <https://openplanet.dev/download>.
2. Install the Visual C++ runtime into the game's prefix:
   ```bash
   protontricks 2225070 vcrun2022
   ```
3. Run the Openplanet installer inside the same prefix:
   ```bash
   protontricks-launch --appid 2225070 ~/Downloads/<the Openplanet installer>.exe
   ```
   When it asks where to install, pick the game folder, normally `Z:\home\<you>\.steam\steam\steamapps\common\Trackmania`.
4. In Steam: Trackmania > Properties > General > **Launch options**:
   ```
   WINEDLLOVERRIDES="dinput8=n,b" %command%
   ```
   Openplanet loads as `dinput8.dll`, and Proton ignores it without this
   ([Openplanet: Linux](https://openplanet.dev/docs/help/linux)).
5. Start the game. The Openplanet bar should appear at the top when you press
   **F3**. Then **Openplanet > Signature Mode > Developer**, as on Windows
   (it lets our plugins load and blocks online play and record submission).

Steam game updates can undo steps 2 and 3. If Openplanet stops showing up
after an update, redo them.

## 4. Virtual gamepad access (uinput)

The AI presses a virtual Xbox 360 pad created through `/dev/uinput`. Give
your user access once:

```bash
sudo dnf install libevdev            # Ubuntu: sudo apt install libevdev2
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
sudo modprobe uinput
echo 'KERNEL=="uinput", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/50-uinput.rules
sudo udevadm control --reload && sudo udevadm trigger
```

Log out and back in. `ls -l /dev/uinput` then `getfacl /dev/uinput` should
list your user.

## 5. Python 3.11 and Git

```bash
sudo dnf install python3.11 git     # Ubuntu: sudo apt install python3.11 python3.11-venv git
```

3.11 is the tested version; newer Pythons work but `tmai doctor` warns.

## 6. Project and PyTorch

```bash
mkdir -p ~/Projects && cd ~/Projects
git clone https://github.com/MortifisNox/Trackmania-AI.git
cd Trackmania-AI
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[game,dashboard,dev]"
```

The PyTorch wheel brings its own CUDA, so only the driver from step 1 is
needed. Check the prompt starts with `(.venv)` before the `pip` lines.

**Your Windows runs.** If you still have the Windows folder
(`D:\AI Projects\Trackmania AI`), copy its `runs/tracks/` folder into the new
`runs/tracks/`. That keeps your demo laps and reference lines, and the AI
learns from them in state mode too. The old checkpoints learned from the
camera and cannot continue in state mode. With `run_name` left empty, state
training goes to `runs/<map>-state/` by itself, so the two never mix.

## 7. Plugins and maps

```bash
tmai install-plugin
tmai get-map            # optional: the tmrl-test map, into the prefix's My Maps
```

In game: **F3 > Developer > (Re)load plugin > TMAI Telemetry**, then
**TMAI Ghost** (or restart the game). Your own maps go in
`~/.steam/steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/Documents/Trackmania/Maps/My Maps/`.

## 8. Game settings

- Windowed or fullscreen both work. The AI no longer looks at the screen, so
  resolution and ghosts do not matter to it. Ghosts still cost frame time,
  so keep them off.
- Graphics preset **Very low / performance**, frame limiter **60 FPS**:
  training shares the GPU with the game.
- **Settings > Controls** (gamepad): bind **Give up** to **View/Back**. The
  game keeps bindings per controller and the AI's virtual pad is its own
  controller, so bind it from that pad: run `tmai press-button BACK`, switch
  to the game, click the box next to Give up, and the pad presses Back after
  15 seconds. Without this, Back respawns at a checkpoint and training stops
  with "BACK respawned at a checkpoint".
- **Unplug your own controller** before binding and while the AI trains. With
  a real pad connected, the game treats that pad as the first controller, so
  bindings and the AI's presses go to different devices and Back keeps
  respawning instead of giving up.
- Keep the game window focused while training, at least until
  `tmai gamepad-test` shows it takes input in the background.

## 9. Laptop power and heat

- Plug in, and set the power profile to **Performance**
  (`powerprofilesctl set performance`, or the GNOME/KDE power menu).
- Raise the back of the laptop for airflow during long runs.
- Turn off automatic suspend while plugged in.
- Check the game runs on the NVIDIA GPU: `nvidia-smi` should list the
  Trackmania process (often `wine64-preloader` or `Trackmania.exe`) while the
  game is running. If it does not, add `prime-run` or
  `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` before
  `%command%` in the launch options.

## 10. Check everything

```bash
tmai doctor
```

Every line should be `[ ok ]` (Python can say `[warn]` on 3.12+). If one is
`[FAIL]` or `[warn]`, paste the output in the project thread. Then, with the
game running on a map, check the plugin connection and the pad:

```bash
tmai telemetry          # numbers should change as you drive
tmai gamepad-test       # the car should steer, brake and restart
```

`tmai capture-test` and `tmai focus-test` are for the camera on Windows and do
not apply here.

## 11. First training run

```bash
./start_training.sh
```

The first time it creates `tmai_settings.json` and opens it: write your map
name, save, and run it again. From then on it works as described in
[EVERYDAY_USE.md](EVERYDAY_USE.md). If the map has no demo laps yet, it asks
you to drive 3 laps first.
