"""Replay buffer in system RAM. Each frame is stored once; stacks of the last
`stack` frames are rebuilt when sampling, which keeps a transition near 4 KB."""

from __future__ import annotations

import numpy as np

from tmai.budget import ACTION_DIM, STATE_DIM


class ReplayBuffer:
    """Stores observations in order. Entry i holds the observation at step i and
    the action, reward and terminal flag of the transition that starts there;
    its successor is entry i+1. The last observation of an episode is added with
    `has_next=False`, so it is only ever used as a next observation."""

    def __init__(self, capacity: int, frame_size: int, stack: int, state_dim: int = STATE_DIM):
        self.capacity, self.stack = capacity, stack
        self.frames = np.zeros((capacity, frame_size, frame_size), dtype=np.uint8)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.float32)
        self.has_next = np.zeros(capacity, dtype=bool)
        # Pace-term inputs (lap fraction before and after, race seconds); NaN where
        # the transition has none, and `reward` then already holds everything.
        self.pace_u0 = np.full(capacity, np.nan, dtype=np.float32)
        self.pace_u1 = np.full(capacity, np.nan, dtype=np.float32)
        self.pace_dt = np.full(capacity, np.nan, dtype=np.float32)
        self.episode = np.full(capacity, -1, dtype=np.int64)
        self.pos = 0
        self.size = 0
        self._episode = 0
        self.transitions = 0     # total transitions ever added

    def __len__(self) -> int:
        return self.size

    def start_episode(self) -> None:
        self._episode += 1

    def add(self, frame, state, action=None, reward: float = 0.0, terminal: bool = False,
            pace: tuple[float, float, float] | None = None) -> None:
        """Add the observation (frame, state) and, unless this is the episode's
        last observation (action None), the transition that starts from it.
        With `pace`, `reward` leaves out the pace term, which sampling scores
        from these inputs against the baseline of the day."""
        i = self.pos
        self.frames[i] = frame
        self.state[i] = state
        self.has_next[i] = action is not None
        if action is not None:
            self.action[i] = action
            self.reward[i] = reward
            self.terminal[i] = float(terminal)
            self.pace_u0[i], self.pace_u1[i], self.pace_dt[i] = pace if pace is not None else (np.nan,) * 3
            self.transitions += 1
        self.episode[i] = self._episode
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _valid(self, idx: np.ndarray) -> np.ndarray:
        nxt = (idx + 1) % self.capacity
        ok = self.has_next[idx] & (self.episode[nxt] == self.episode[idx])
        return ok & (nxt != self.pos)   # the successor has not been written yet

    def _stack(self, idx: np.ndarray) -> np.ndarray:
        """(B, stack, H, W): frames idx-stack+1 .. idx, repeating the episode's
        first frame where the episode is shorter than the stack."""
        out = np.empty((len(idx), self.stack) + self.frames.shape[1:], dtype=np.uint8)
        cur = idx.copy()
        out[:, -1] = self.frames[cur]
        for k in range(self.stack - 2, -1, -1):
            prev = (cur - 1) % self.capacity
            same = (self.episode[prev] == self.episode[idx]) & (prev != (self.pos - 1) % self.capacity)
            cur = np.where(same, prev, cur)
            out[:, k] = self.frames[cur]
        return out

    def rewards(self, idx: np.ndarray, pace_fn=None) -> np.ndarray:
        """Rewards of the transitions at idx, with the pace term scored by
        pace_fn(u0, u1, dt) where the transition stored its inputs."""
        r = self.reward[idx].astype(np.float32)
        if pace_fn is not None:
            has = ~np.isnan(self.pace_dt[idx])
            if has.any():
                j = idx[has]
                r[has] += pace_fn(self.pace_u0[j].astype(np.float64), self.pace_u1[j].astype(np.float64),
                                  self.pace_dt[j].astype(np.float64))
        return r

    def sample(self, n: int, rng: np.random.Generator, n_step: int = 1, gamma: float = 0.99,
               pace_fn=None) -> dict:
        """A batch of n-step transitions. `reward` is the discounted sum of up to
        n_step rewards and `discount` is what the bootstrapped value is scaled by:
        gamma**k after k steps, or 0 when the episode ended in a terminal state.
        A return is cut short where the episode stops, so it never crosses into
        another attempt."""
        if self.size < 2:
            raise ValueError("replay buffer is empty")
        idx = np.empty(0, dtype=np.int64)
        for _ in range(50):
            cand = rng.integers(0, self.size, size=2 * n)
            idx = np.concatenate([idx, cand[self._valid(cand)]])
            if len(idx) >= n:
                break
        else:
            raise ValueError("replay buffer has too few complete transitions")
        idx = idx[:n]
        ret = np.zeros(n, dtype=np.float32)
        disc = np.ones(n, dtype=np.float32)
        cur, active = idx.copy(), np.ones(n, dtype=bool)
        for k in range(n_step):
            ret[active] += disc[active] * self.rewards(cur[active], pace_fn)
            disc[active] *= gamma
            ended = active & (self.terminal[cur] > 0)
            disc[ended] = 0.0
            active &= ~ended
            if k == n_step - 1:
                break
            nxt = (cur + 1) % self.capacity
            active &= self._valid(nxt)    # the attempt goes on with another transition
            cur = np.where(active, nxt, cur)
        nxt = (cur + 1) % self.capacity
        return {
            "frames": self._stack(idx), "state": self.state[idx], "action": self.action[idx],
            "reward": ret, "discount": disc,
            "next_frames": self._stack(nxt), "next_state": self.state[nxt],
        }

    def stacked(self, i: int) -> np.ndarray:
        return self._stack(np.array([i]))[0]


def concat_batches(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in a}
