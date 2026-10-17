"""Run configuration: YAML defaults plus command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in; unknown keys are errors."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise ConfigError(f"unknown config key: {key}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must be a mapping")
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_override(text: str) -> dict:
    """Turn `section.key=value` into a nested dict, typing the value as YAML."""
    if "=" not in text:
        raise ConfigError(f"override must look like section.key=value: {text!r}")
    dotted, raw = text.split("=", 1)
    parts = [p for p in dotted.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"empty key in override: {text!r}")
    value: Any = yaml.safe_load(raw) if raw.strip() else ""
    for part in reversed(parts):
        value = {part: value}
    return value


def validate(cfg: dict) -> None:
    gpu, train, env = cfg["gpu"], cfg["train"], cfg["env"]
    if not 0.1 <= gpu["max_vram_fraction"] <= 0.9:
        raise ConfigError("gpu.max_vram_fraction must be between 0.1 and 0.9")
    if gpu["temp_soft_c"] >= gpu["temp_hard_c"]:
        raise ConfigError("gpu.temp_soft_c must be below gpu.temp_hard_c")
    if not 0.0 <= train["demo_ratio"] < 1.0:
        raise ConfigError("train.demo_ratio must be in [0, 1)")
    if train["batch_size"] % 8:
        # Tensor cores want dimensions that are multiples of 8.
        raise ConfigError("train.batch_size must be a multiple of 8")
    if env["step_hz"] <= 0:
        raise ConfigError("env.step_hz must be positive")
    from tmai.game.gamepad import BUTTONS

    if env["restart_button"] not in BUTTONS:
        raise ConfigError(f"env.restart_button must be one of {sorted(BUTTONS)}")
    if env["capture_backend"] not in ("window", "screen"):
        raise ConfigError("env.capture_backend must be window or screen")
    if cfg["dashboard"]["host"] not in ("127.0.0.1", "localhost"):
        raise ConfigError("dashboard.host must stay on localhost")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for text in overrides or []:
        cfg = deep_merge(cfg, parse_override(text))
    validate(cfg)
    return cfg
