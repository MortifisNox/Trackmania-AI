import pytest

from tmai.config import ConfigError, deep_merge, load_config, parse_override


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert cfg["env"]["step_hz"] == 20
    assert cfg["gpu"]["max_vram_fraction"] == 0.5


def test_override_is_typed_as_yaml():
    cfg = load_config(overrides=["train.batch_size=128", "model.precision=fp16"])
    assert cfg["train"]["batch_size"] == 128
    assert cfg["model"]["precision"] == "fp16"


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["train.batchsize=128"])


def test_invalid_values_are_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["train.batch_size=250"])
    with pytest.raises(ConfigError):
        load_config(overrides=["dashboard.host=0.0.0.0"])
    with pytest.raises(ConfigError):
        load_config(overrides=["gpu.temp_soft_c=90"])


def test_parse_override_nests_keys():
    assert parse_override("a.b.c=3") == {"a": {"b": {"c": 3}}}
    with pytest.raises(ConfigError):
        parse_override("no-equals-sign")


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    merged = deep_merge(base, {"a": {"b": 2}})
    assert base["a"]["b"] == 1 and merged["a"]["b"] == 2
