"""
modeling/components/replay_buffer/base.py
-------------------------------------------
Abstract base class for all replay buffers.

A replay buffer stores (state, action, reward, next_state, done)
transitions and provides sampling for learning updates.
Swap this to change the sampling strategy
(uniform, prioritised experience replay, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseReplayBuffer(ABC):

    @abstractmethod
    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """
        Store a single transition in the buffer.

        Args:
            state:      Observation before the action.
            action:     Discrete action taken.
            reward:     Scalar reward received.
            next_state: Observation after the action.
            done:       True if this transition ends the episode.
        """
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[
        np.ndarray,   # states
        np.ndarray,   # actions
        np.ndarray,   # rewards
        np.ndarray,   # next_states
        np.ndarray,   # dones
    ]:
        """
        Sample a batch of transitions for a learning update.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of numpy arrays:
            (states, actions, rewards, next_states, dones)
            All float32 except actions (int64) and dones (float32 0/1).
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of transitions currently stored.
        Used by the policy to check if enough samples exist before updating.
        """
        ...

    @abstractmethod
    def is_ready(self, batch_size: int) -> bool:
        """
        Returns True if the buffer contains enough transitions
        to produce a valid batch of the given size.

        Args:
            batch_size: Minimum number of transitions required.
        """
        ...