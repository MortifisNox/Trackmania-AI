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
    if system != "Windows":
        return Check("OS", WARN, f"{system}; the game side only runs on Windows")
    return Check("OS", OK, f"Windows {platform.release()}")


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
    frame = cfg["env"]["frame_size"]
    capacity = cfg["replay"]["capacity"]
    need = budget.replay_bytes(capacity, frame)
    suggested = budget.suggest_replay_capacity(total_ram, frame)
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


def run_checks(cfg: dict, home: Path | None = None) -> list[Check]:
    home = home or Path(os.environ.get("USERPROFILE") or Path.home())
    checks = [check_python(), check_os(), check_nvidia_smi()]
    checks += check_torch()
    checks.append(check_module("vgamepad", "Virtual gamepad", "pip install vgamepad and accept the ViGEmBus driver"))
    checks.append(check_module("dxcam", "Screen capture", 'pip install -e ".[game]"'))
    checks.append(check_module("cv2", "OpenCV", 'pip install -e ".[game]"'))
    checks.append(check_ram(cfg))
    checks += check_openplanet(home)
    checks.append(check_disk(Path(cfg["run"]["output_dir"])))
    return checks


def format_report(checks: list[Check]) -> str:
    marks = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    width = max(len(c.name) for c in checks)
    return "\n".join(f"{marks[c.status]} {c.name:<{width}}  {c.detail}" for c in checks)
