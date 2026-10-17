"""Memory budgeting for the replay buffer (system RAM) and the learner (VRAM)."""

from __future__ import annotations

# Per transition: one uint8 frame (stacks are rebuilt from neighbouring frames),
# a float32 state vector, a 3-float action, reward, done flag and episode index.
STATE_DIM = 48
ACTION_DIM = 3


def bytes_per_transition(frame_size: int, state_dim: int = STATE_DIM) -> int:
    frame = frame_size * frame_size
    state = state_dim * 4
    action = ACTION_DIM * 4
    return frame + state + action + 4 + 1 + 4


def replay_bytes(capacity: int, frame_size: int) -> int:
    return capacity * bytes_per_transition(frame_size)


def suggest_replay_capacity(total_ram_bytes: int, frame_size: int, ram_share: float = 0.25) -> int:
    """Largest capacity (rounded down to 50k) that uses at most `ram_share` of RAM.

    The game, the OS and the browser dashboard need the rest; a quarter of RAM
    keeps a 16 GB laptop clear of swapping during multi-day runs.
    """
    budget = total_ram_bytes * ram_share
    capacity = int(budget // bytes_per_transition(frame_size))
    return max(50_000, (capacity // 50_000) * 50_000)


def vram_budget_bytes(total_vram_bytes: int, max_fraction: float) -> int:
    return int(total_vram_bytes * max_fraction)
