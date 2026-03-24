"""Abstract base for replay buffers. Stores and samples (s, a, r, s', done) transitions."""

from abc import ABC, abstractmethod
import numpy as np


class BaseReplayBuffer(ABC):

    @abstractmethod
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Store one transition."""
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        """Return a random batch of (states, actions, rewards, next_states, dones)."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def is_ready(self, batch_size: int) -> bool:
        """True if enough transitions are stored to sample a batch."""
        ...
