"""Run configuration: YAML defaults plus command-line overrides."""

from __future__ import annotations

import copy
import json
import math
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
    """Turn `section.key=value` into a nested dict, accepting JSON and YAML values."""
    if "=" not in text:
        raise ConfigError(f"override must look like section.key=value: {text!r}")
    dotted, raw = text.split("=", 1)
    parts = [p for p in dotted.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"empty key in override: {text!r}")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = yaml.safe_load(raw) if raw.strip() else ""
    for part in reversed(parts):
        value = {part: value}
    return value


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _require_bool(section: dict, key: str, full: str, default: bool | None = None) -> bool:
    v = section.get(key, default) if default is not None and key not in section else section[key]
    if not isinstance(v, bool):
        raise ConfigError(f"{full} must be a boolean (true or false), got {v!r}")
    return v


def _require_int(section: dict, key: str, full: str, *, lo: int | None = None, hi: int | None = None,
                 default: int | None = None) -> int:
    v = section.get(key, default) if default is not None and key not in section else section[key]
    if not _is_int(v):
        raise ConfigError(f"{full} must be an integer, got {v!r}")
    if lo is not None and v < lo:
        raise ConfigError(f"{full} must be at least {lo}")
    if hi is not None and v > hi:
        raise ConfigError(f"{full} must be at most {hi}")
    return v


def _require_number(section: dict, key: str, full: str, *, lo: float | None = None, hi: float | None = None,
                    lo_open: bool = False, hi_open: bool = False, default: float | None = None) -> float:
    v = section.get(key, default) if default is not None and key not in section else section[key]
    if not _is_number(v):
        raise ConfigError(f"{full} must be a finite number, got {v!r}")
    if lo is not None and (v <= lo if lo_open else v < lo):
        op = ">" if lo_open else ">="
        raise ConfigError(f"{full} must be {op} {lo}")
    if hi is not None and (v >= hi if hi_open else v > hi):
        op = "<" if hi_open else "<="
        raise ConfigError(f"{full} must be {op} {hi}")
    return float(v)


def validate(cfg: dict) -> None:
    run, gpu, train, env = cfg["run"], cfg["gpu"], cfg["train"], cfg["env"]
    track, rw, m, replay, dash = cfg["track"], cfg["reward"], cfg["model"], cfg["replay"], cfg["dashboard"]

    _require_int(run, "seed", "run.seed")

    _require_number(gpu, "max_vram_fraction", "gpu.max_vram_fraction", lo=0.1, hi=0.9)
    soft_c = _require_number(gpu, "temp_soft_c", "gpu.temp_soft_c")
    hard_c = _require_number(gpu, "temp_hard_c", "gpu.temp_hard_c")
    if soft_c >= hard_c:
        raise ConfigError("gpu.temp_soft_c must be below gpu.temp_hard_c")

    for key in ("demo_ratio", "demo_ratio_end"):
        _require_number(train, key, f"train.{key}", lo=0.0, hi=1.0, hi_open=True)
    _require_number(train, "gamma", "train.gamma", lo=0.0, hi=1.0, lo_open=True)
    _require_number(train, "tau", "train.tau", lo=0.0, hi=1.0, lo_open=True)
    _require_number(train, "learning_rate", "train.learning_rate", lo=0.0, lo_open=True)
    _require_int(train, "n_step", "train.n_step", lo=1)
    _require_int(train, "updates_per_step", "train.updates_per_step", lo=0)
    _require_int(train, "warmup_steps", "train.warmup_steps", lo=0)
    _require_int(train, "demo_anneal_steps", "train.demo_anneal_steps", lo=0)
    _require_int(train, "best_laps_kept", "train.best_laps_kept", lo=0)
    _require_int(train, "entropy_anneal_steps", "train.entropy_anneal_steps", lo=0)
    _require_int(train, "eval_every", "train.eval_every", lo=0)
    _require_number(train, "target_entropy", "train.target_entropy")
    _require_number(train, "target_entropy_end", "train.target_entropy_end")
    _require_number(train, "checkpoint_every_min", "train.checkpoint_every_min", lo=0.0, lo_open=True)
    batch_size = _require_int(train, "batch_size", "train.batch_size", lo=8)
    if batch_size % 8:
        # Tensor cores want dimensions that are multiples of 8.
        raise ConfigError("train.batch_size must be a multiple of 8")

    _require_number(env, "step_hz", "env.step_hz", lo=0.0, lo_open=True)
    _require_int(env, "frame_size", "env.frame_size", lo=1)
    _require_int(env, "frame_stack", "env.frame_stack", lo=1)
    lookahead_points = _require_int(env, "lookahead_points", "env.lookahead_points", lo=1)
    _require_number(env, "lookahead_spacing_m", "env.lookahead_spacing_m", lo=0.0, lo_open=True)
    _require_number(env, "lookahead_growth", "env.lookahead_growth", lo=1.0, hi=1.5)
    _require_bool(env, "lidar", "env.lidar", default=True)
    _require_bool(env, "launch_hold_gas", "env.launch_hold_gas", default=True)
    _require_bool(env, "digital_pedals", "env.digital_pedals", default=True)
    _require_bool(gpu, "allow_cpu", "gpu.allow_cpu", default=False)
    _require_number(env, "episode_timeout_s", "env.episode_timeout_s", lo=0.0, lo_open=True)
    _require_number(env, "stuck_timeout_s", "env.stuck_timeout_s", lo=0.0, lo_open=True)
    _require_number(env, "stuck_speed_kmh", "env.stuck_speed_kmh", lo=0.0)
    _require_int(env, "telemetry_port", "env.telemetry_port", lo=1, hi=65535)
    _require_number(env, "max_telemetry_age_s", "env.max_telemetry_age_s", lo=0.0, hi=1.0, lo_open=True, default=0.1)
    from tmai.game.gamepad import BUTTONS

    if env["restart_button"] not in BUTTONS:
        raise ConfigError(f"env.restart_button must be one of {sorted(BUTTONS)}")
    if env.get("menu_button", "A") not in BUTTONS:
        raise ConfigError(f"env.menu_button must be one of {sorted(BUTTONS)}")
    if env["observation"] not in ("image", "state"):
        raise ConfigError("env.observation must be image or state")
    if env["observation"] == "image" and lookahead_points != 10:
        raise ConfigError("env.lookahead_points must be 10 with observation image; state mode allows any count")
    # env.lidar is on by default; it only applies in state mode (image mode ignores it).
    if env["capture_backend"] not in ("window", "screen"):
        raise ConfigError("env.capture_backend must be window or screen")

    _require_number(track, "reference_spacing_m", "track.reference_spacing_m", lo=0.0, lo_open=True)
    _require_number(track, "max_progress_jump_m", "track.max_progress_jump_m", lo=0.0, lo_open=True)
    _require_number(track, "wrong_way_m", "track.wrong_way_m", lo=0.0, lo_open=True)
    _require_number(track, "off_track_m", "track.off_track_m", lo=0.0)
    _require_number(track, "crash_speed_drop_kmh", "track.crash_speed_drop_kmh", lo=0.0, lo_open=True)
    _require_int(track, "adopt_best_gain_ms", "track.adopt_best_gain_ms", lo=0)

    _require_number(rw, "progress_per_m", "reward.progress_per_m")
    _require_number(rw, "time_penalty", "reward.time_penalty", lo=0.0)
    _require_number(rw, "pace_per_s", "reward.pace_per_s", lo=0.0)
    _require_int(rw, "pace_baseline_laps", "reward.pace_baseline_laps", lo=0)
    _require_int(rw, "pace_baseline_min_laps", "reward.pace_baseline_min_laps", lo=1)
    _require_number(rw, "pace_baseline_percentile", "reward.pace_baseline_percentile", lo=0.0, hi=100.0, default=25.0)
    _require_number(rw, "pace_loss_cap", "reward.pace_loss_cap", hi=1.0, default=0.5)
    _require_number(rw, "pb_bonus_per_s", "reward.pb_bonus_per_s", lo=0.0)
    _require_number(rw, "pb_bonus_cap", "reward.pb_bonus_cap", lo=0.0)
    _require_number(rw, "crash_penalty", "reward.crash_penalty", lo=0.0)
    _require_number(rw, "split_bonus_per_s", "reward.split_bonus_per_s", lo=0.0)
    _require_number(rw, "split_bonus_cap", "reward.split_bonus_cap", lo=0.0)
    _require_number(rw, "finish_bonus", "reward.finish_bonus", lo=0.0)

    _require_int(m, "hidden_dim", "model.hidden_dim", lo=1)
    _require_int(m, "num_critics", "model.num_critics", lo=1)
    _require_number(m, "critic_dropout", "model.critic_dropout", lo=0.0, hi=1.0, hi_open=True)
    _require_bool(m, "actor_layer_norm", "model.actor_layer_norm", default=True)
    head = m.get("critic_head", "scalar")
    if head not in ("scalar", "hl_gauss"):
        raise ConfigError("model.critic_head must be scalar or hl_gauss")
    _require_int(m, "q_bins", "model.q_bins", lo=2, default=141)
    q_min = _require_number(m, "q_min", "model.q_min", default=-150.0)
    q_max = _require_number(m, "q_max", "model.q_max", default=200.0)
    if not (math.isfinite(q_max - q_min) and q_min < q_max):
        raise ConfigError("model.q_min must be below model.q_max with finite span")
    _require_number(m, "hl_sigma_ratio", "model.hl_sigma_ratio", lo=0.0, lo_open=True, default=0.75)

    if train.get("optimizer", "adam") not in ("adam", "adamw"):
        raise ConfigError("train.optimizer must be adam or adamw")
    _require_number(train, "weight_decay", "train.weight_decay", lo=0.0, default=0.0)
    critic_loss = train.get("critic_loss", "mse")
    if critic_loss not in ("mse", "huber"):
        raise ConfigError("train.critic_loss must be mse or huber")
    if head == "scalar" and critic_loss == "huber":
        _require_number(train, "huber_delta", "train.huber_delta", lo=0.0, lo_open=True, default=1.0)
    _require_int(train, "reset_every_updates", "train.reset_every_updates", lo=0, default=0)
    _require_number(train, "reset_shrink", "train.reset_shrink", lo=0.0, hi=1.0, default=0.8)

    _require_int(replay, "capacity", "replay.capacity", lo=1)
    _require_number(replay, "snapshot_every_min", "replay.snapshot_every_min", lo=0.0, lo_open=True)

    if dash["host"] not in ("127.0.0.1", "localhost"):
        raise ConfigError("dashboard.host must stay on localhost")
    port = _require_int(dash, "port", "dashboard.port", lo=0, hi=65535)
    link_port = _require_int(dash, "link_port", "dashboard.link_port", lo=0, hi=65535)
    if port == link_port:
        raise ConfigError("dashboard.port and dashboard.link_port must differ")
    _require_bool(dash, "tensorboard", "dashboard.tensorboard", default=True)


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
