"""
modeling/components/replay_buffer/uniform.py
----------------------------------------------
Uniform random replay buffer implementation.

Stores transitions in a fixed-size circular buffer.
Sampling is uniformly random — every stored transition has
equal probability of being selected for a learning update.

This is the standard replay buffer used with DQN.
"""

import random
from collections import deque
from typing import Any

import numpy as np

from .base import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    """
    Fixed-capacity circular buffer with uniform random sampling.

    Args:
        capacity:  Maximum number of transitions to store.
                   Once full, oldest transitions are overwritten.
        seed:      Random seed for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        seed:     int = 42,
    ):
        self._capacity = capacity
        self._buffer:  deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store a single transition."""
        self._buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Sample a random batch of transitions.

        Returns:
            states:      (batch_size, obs_dim)  float32
            actions:     (batch_size,)           int64
            rewards:     (batch_size,)           float32
            next_states: (batch_size, obs_dim)  float32
            dones:       (batch_size,)           float32
        """
        if not self.is_ready(batch_size):
            raise RuntimeError(
                f"Buffer has {len(self)} transitions — "
                f"need at least {batch_size} to sample."
            )

        batch = self._rng.sample(self._buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        """Empty the buffer. Useful between train and test phases."""
        self._buffer.clear()