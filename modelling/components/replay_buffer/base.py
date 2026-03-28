"""Abstract base for replay buffers.

Stores and samples (s, a, r, s', done, duration) transitions.
`duration` is the number of environment primitive steps elapsed between the decision
that produced the transition and the next decision epoch.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseReplayBuffer(ABC):

    @abstractmethod
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, duration: int) -> None:
        """Store one transition."""
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Return a random batch of (states, actions, rewards, next_states, dones, durations)."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def is_ready(self, batch_size: int) -> bool:
        """True if enough transitions are stored to sample a batch."""
        ...
