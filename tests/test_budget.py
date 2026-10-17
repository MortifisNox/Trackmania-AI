from tmai import budget

GIB = 1024**3


def test_default_replay_fits_in_a_quarter_of_16gb():
    assert budget.replay_bytes(500_000, 64) < 4 * GIB


def test_suggested_capacity_scales_with_ram_and_is_rounded():
    small = budget.suggest_replay_capacity(8 * GIB, 64)
    large = budget.suggest_replay_capacity(32 * GIB, 64)
    assert small < large
    assert small % 50_000 == 0 and large % 50_000 == 0
    assert budget.replay_bytes(large, 64) <= 32 * GIB * 0.25


def test_suggested_capacity_has_a_floor():
    assert budget.suggest_replay_capacity(1, 64) == 50_000
