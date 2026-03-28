"""Uniform replay buffer — fixed-size circular buffer with random sampling."""

import random
from collections import deque

import numpy as np

from .base import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    """Every stored transition has equal probability of being sampled."""

    def __init__(self, capacity: int = 50_000, seed: int = 42):
        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, duration: int) -> None:
        self._buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            float(done),
            int(duration),
        ))

    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        if not self.is_ready(batch_size):
            raise RuntimeError(
                f"Buffer has {len(self)} transitions — "
                f"need at least {batch_size} to sample."
            )
        batch = self._rng.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones, durations = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
            np.array(durations, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        self._buffer.clear()
