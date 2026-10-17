# Setting up the gaming laptop

One-time setup of the Windows laptop (RTX 4050 Laptop GPU). Do the steps in
order; step 9 checks the result and step 10 gets you to the first training
run. Read [SAFE_USE.md](SAFE_USE.md) before you train. After setup, the
day-to-day steps are in [EVERYDAY_USE.md](EVERYDAY_USE.md).

## 1. NVIDIA driver

Install the latest Game Ready or Studio driver (through the NVIDIA App or
nvidia.com). Then open PowerShell and run:

```powershell
nvidia-smi
```

It should list "NVIDIA GeForce RTX 4050 Laptop GPU" and a driver version.

## 2. Visual C++ runtime

Openplanet needs the Microsoft Visual C++ Redistributable (x64). Install the
latest one from Microsoft's "Latest supported Visual C++ Redistributable
downloads" page.

## 3. Trackmania

1. Install Trackmania (2020) from Ubisoft Connect, Steam or Epic and log in.
2. Start it once and reach the main menu.

You need **Club** access. Our two plugins are self-made (unsigned), and
Openplanet only loads those in Developer mode, which needs Club
([Openplanet: Club](https://openplanet.dev/next/club)). Saving the best lap
as a replay needs Club too.

## 4. Openplanet

1. Download Openplanet for Trackmania from <https://openplanet.dev/download>
   and install it. If Windows SmartScreen warns about the certificate, choose
   "More info", then "Run anyway" (TMRL's install guide documents the same
   warning).
2. Start the game once, so `C:\Users\<you>\OpenplanetNext` is created.
3. Press **F3** to open the Openplanet overlay, then
   **Openplanet > Signature Mode** and pick **Developer**. It lets our plugins
   load, and it also blocks online play and record submission to official
   leaderboards, which keeps training away from them. See SAFE_USE.md.

## 5. Python and Git

1. Install **Python 3.11 (64-bit)** from python.org. Tick "Add python.exe to PATH".
2. Install **Git for Windows**.

## 6. Project and PyTorch

In PowerShell (these commands also work if the folder already has files in it):

```powershell
cd "D:\AI Projects\Trackmania AI"
git init
git remote add origin https://github.com/MortifisNox/Trackmania-AI.git
git fetch origin
git checkout -b main origin/main
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, answer Y; lets the venv activate
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[game,dashboard,dev]"
```

Check that the prompt starts with `(.venv)` before the `pip` lines. Without it,
packages go into your global Python instead of the project.

Installing `vgamepad` opens the **ViGEmBus** driver installer (the virtual Xbox
360 controller the agent drives with). Accept it. If it does not appear, install
ViGEmBus from its GitHub releases page and reboot.

## 7. Game settings

The agent sees a small grayscale picture of the game window plus telemetry, so
the game should be light and consistent:

- Display mode **Windowed**, resolution **1280x720** (lower is fine).
- Graphics preset **Very low / performance**, motion blur off, V-sync off.
- Frame limiter **60 FPS**. This caps GPU heat and leaves compute for training.
- Keep the default chase camera; do not switch cameras during a run.
- Turn **ghosts off**, your own personal-best ghost included. The AI learns
  from the screen and a second car confuses it.
- Keep the game window open (not minimised) while training. It can be behind
  other windows.
- **Settings > Controls** (gamepad): bind **Give up** to the **View/Back**
  button (the small button left of the Xbox logo with two squares on it, not
  the red B). The AI restarts runs with it; that is what
  `--set env.restart_button=BACK` refers to.

## 8. Laptop power and heat

- Plug the laptop in. Windows power mode **Best performance**.
- In your laptop's vendor tool (Armoury Crate, Lenovo Vantage, etc.) pick the
  performance fan profile.
- NVIDIA Control Panel > Manage 3D settings > Program settings: set
  `Trackmania.exe` and `python.exe` (in `.venv\Scripts`) to the
  **High-performance NVIDIA processor**. On Optimus laptops this matters.
- Raise the back of the laptop for airflow during long runs.
- Turn on Focus assist / Do not disturb, and set sleep to "Never" while plugged in.

## 9. Check everything

```powershell
tmai doctor
```

Every line should be `[ ok ]`. If one is `[FAIL]` or `[warn]`, paste the
output in the project thread.

## 10. First training run

1. Install the plugins: `tmai install-plugin`, then in game
   **F3 > Developer > (Re)load plugin > TMAI Telemetry** and **TMAI Ghost**
   (or restart the game).
2. Optional first check that the game connection works:
   [PHASE1_TESTING.md](PHASE1_TESTING.md).
3. Record 3 demo laps of your map: [PHASE2_TESTING.md](PHASE2_TESTING.md).
4. Train, watch and stop: [EVERYDAY_USE.md](EVERYDAY_USE.md).
