"""`tmai doctor`: checks the gaming machine is ready before any training starts."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tmai import budget

OK, WARN, FAIL = "ok", "warn", "fail"
GIB = 1024**3


@dataclass
class Check:
    name: str
    status: str
    detail: str


def check_python(version: tuple = sys.version_info[:3]) -> Check:
    text = ".".join(map(str, version))
    if version < (3, 10):
        return Check("Python", FAIL, f"{text}; install Python 3.11 (64-bit)")
    if version[:2] != (3, 11):
        return Check("Python", WARN, f"{text}; 3.11 is the tested version")
    return Check("Python", OK, text)


def check_os(system: str = platform.system()) -> Check:
    if system == "Windows":
        return Check("OS", OK, f"Windows {platform.release()}")
    if system == "Linux":
        return Check("OS", OK, f"Linux {platform.release()}; the game runs under Proton or Wine")
    return Check("OS", WARN, f"{system}; only Windows and Linux are supported")


def check_observation(cfg: dict, system: str = platform.system()) -> Check:
    mode = cfg["env"]["observation"]
    if mode == "image" and system != "Windows":
        return Check("Observation", FAIL, "image needs Windows window capture; set observation to state "
                     "(tmai_settings.json, or --set env.observation=state)")
    if mode == "image":
        return Check("Observation", OK, "image: camera frames plus game state")
    return Check("Observation", OK, "state: game state from the plugin, no camera")


def check_uinput(path: Path = Path("/dev/uinput")) -> Check:
    """Linux: vgamepad creates its virtual pad through /dev/uinput."""
    if not path.exists():
        return Check("uinput", FAIL, "/dev/uinput missing; run: sudo modprobe uinput (see docs/SETUP_LINUX.md)")
    if not os.access(path, os.R_OK | os.W_OK):
        return Check("uinput", FAIL, "no access to /dev/uinput; add the udev rule in docs/SETUP_LINUX.md step 4, "
                     "then log out and in")
    return Check("uinput", OK, "read/write")


def check_proton_launch_option(home: Path) -> Check:
    """Openplanet loads as dinput8.dll, which Proton ignores unless told to load it."""
    from tmai.game.paths import STEAM_APP_ID, steam_roots

    configs = [c for root in steam_roots(home) for c in (root / "userdata").glob("*/config/localconfig.vdf")]
    if not configs:
        return Check("Launch option", WARN, "no Steam config found; if you play through Steam, set the launch "
                     'option WINEDLLOVERRIDES="dinput8=n,b" %command%')
    for vdf in configs:
        text = vdf.read_text(errors="replace")
        at = text.find(f'"{STEAM_APP_ID}"')
        if at >= 0 and "dinput8" in text[at:at + 4000]:
            return Check("Launch option", OK, "Steam loads Openplanet (dinput8 override set)")
    return Check("Launch option", WARN, 'Trackmania > Properties > Launch options in Steam should be '
                 'WINEDLLOVERRIDES="dinput8=n,b" %command% (restart Steam first if you just set it)')


def check_torch() -> list[Check]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return [Check("PyTorch", FAIL, "not installed; see docs/SETUP_WINDOWS.md step 6")]
    checks = [Check("PyTorch", OK, torch.__version__)]
    if not torch.cuda.is_available():
        checks.append(Check("CUDA", FAIL, "torch cannot see the GPU; reinstall the CUDA wheel"))
        return checks
    props = torch.cuda.get_device_properties(0)
    checks.append(Check("CUDA", OK, f"{props.name}, {props.total_memory / GIB:.1f} GiB VRAM"))
    if props.total_memory < 5.5 * GIB:
        checks.append(Check("VRAM", WARN, "under 6 GiB; lower gpu.max_vram_fraction"))
    bf16 = torch.cuda.is_bf16_supported()
    checks.append(Check("bf16", OK if bf16 else WARN, "supported" if bf16 else "use precision: fp16"))
    return checks


def check_module(module: str, label: str, hint: str) -> Check:
    try:
        importlib.import_module(module)
    except Exception as exc:  # vgamepad raises non-ImportError when ViGEmBus is missing
        return Check(label, FAIL, f"{hint} ({type(exc).__name__}: {exc})")
    return Check(label, OK, "importable")


def check_ram(cfg: dict, total_ram: int | None = None) -> Check:
    if total_ram is None:
        import psutil

        total_ram = psutil.virtual_memory().total
    from tmai.rl.features import frame_size, state_dim

    frame, sdim = frame_size(cfg["env"]), state_dim(cfg["env"])
    capacity = cfg["replay"]["capacity"]
    need = budget.replay_bytes(capacity, frame, sdim)
    suggested = budget.suggest_replay_capacity(total_ram, frame, state_dim=sdim)
    detail = (
        f"{total_ram / GIB:.0f} GiB RAM; replay of {capacity:,} needs {need / GIB:.1f} GiB, "
        f"suggested capacity {suggested:,}"
    )
    return Check("RAM", WARN if capacity > suggested else OK, detail)


def check_openplanet(home: Path) -> list[Check]:
    root = home / "OpenplanetNext"
    if not root.is_dir():
        return [Check("Openplanet", FAIL, f"{root} missing; install Openplanet and start the game once")]
    plugins = root / "Plugins"
    names = sorted(p.name for p in plugins.iterdir()) if plugins.is_dir() else []
    ours = any(n.lower().startswith("tmai") for n in names)
    tmrl = any(n.lower().startswith("tmrl") for n in names)
    if ours:
        plugin = Check("Telemetry plugin", OK, "tmai plugin found")
    elif tmrl:
        plugin = Check("Telemetry plugin", WARN, "only TMRL_GrabData found; fine for phase 1")
    else:
        plugin = Check("Telemetry plugin", WARN, "none yet; installed in phase 1")
    return [Check("Openplanet", OK, str(root)), plugin]


def check_disk(path: Path, min_free_gib: float = 20.0) -> Check:
    target = path if path.exists() else path.resolve().anchor or Path.cwd()
    free = shutil.disk_usage(target).free / GIB
    if free < min_free_gib:
        return Check("Disk", WARN, f"{free:.0f} GiB free; keep {min_free_gib:.0f}+ GiB for replay snapshots")
    return Check("Disk", OK, f"{free:.0f} GiB free")


def check_nvidia_smi() -> Check:
    found = shutil.which("nvidia-smi")
    if not found:
        return Check("NVIDIA driver", FAIL, "nvidia-smi not on PATH; install the latest driver")
    return Check("NVIDIA driver", OK, found)


def run_checks(cfg: dict, home: Path | None = None, system: str = platform.system()) -> list[Check]:
    from tmai.game.paths import game_user_dir

    linux = system == "Linux"
    home = home or game_user_dir(system=system)
    checks = [check_python(), check_os(system), check_observation(cfg, system), check_nvidia_smi()]
    checks += check_torch()
    if linux:
        checks.append(check_uinput())
        checks.append(check_module("vgamepad", "Virtual gamepad",
                                   'pip install -e ".[game]" (needs libevdev: sudo dnf install libevdev)'))
    else:
        checks.append(check_module("vgamepad", "Virtual gamepad", "pip install vgamepad and accept the ViGEmBus driver"))
    if cfg["env"]["observation"] == "image":
        checks.append(check_module("dxcam", "Screen capture", 'pip install -e ".[game]"'))
    checks.append(check_module("cv2", "OpenCV", 'pip install -e ".[game]"'))
    checks.append(check_ram(cfg))
    checks += check_openplanet(home)
    if linux:
        checks.append(Check("Game folder", OK if (home / "OpenplanetNext").is_dir() else WARN,
                            f"{home} (set TMAI_GAME_USER_DIR to use another prefix)"))
        checks.append(check_proton_launch_option(Path.home()))
    checks.append(check_disk(Path(cfg["run"]["output_dir"])))
    return checks


def format_report(checks: list[Check]) -> str:
    marks = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    width = max(len(c.name) for c in checks)
    return "\n".join(f"{marks[c.status]} {c.name:<{width}}  {c.detail}" for c in checks)
