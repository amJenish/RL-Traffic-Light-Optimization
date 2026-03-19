"""
modeling/components/policy/base.py
------------------------------------
Abstract base class for all policies.

A policy maps an observation vector to a discrete action.
Swap this to change the learning algorithm (DQN, DDQN, PPO, etc.)
without touching the environment, reward, or trainer.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BasePolicy(ABC):

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """
        Select an action given the current observation.
        During training this should use exploration (e.g. epsilon-greedy).
        During evaluation call with exploration disabled.

        Args:
            obs: Observation vector from the observation builder.
                 Shape must match what was passed to __init__.

        Returns:
            int — discrete action index.
            Action space for a 2-phase intersection:
                0 = keep current phase
                1 = switch to next phase
        """
        ...

    @abstractmethod
    def update(self, replay_buffer: Any) -> float | None:
        """
        Perform one learning update using transitions sampled
        from the replay buffer.

        Args:
            replay_buffer: A BaseReplayBuffer instance to sample from.

        Returns:
            float loss value if an update was performed, else None
            (e.g. buffer not yet full enough to sample).
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save policy weights/state to disk.

        Args:
            path: File path to save to (e.g. 'models/dqn_ep100.pt').
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load policy weights/state from disk.

        Args:
            path: File path to load from.
        """
        ...

    def set_eval_mode(self) -> None:
        """
        Switch to evaluation mode — disables exploration.
        Override for policies that have a training/eval distinction.
        Default implementation does nothing.
        """
        pass

    def set_train_mode(self) -> None:
        """
        Switch back to training mode — re-enables exploration.
        Override for policies that have a training/eval distinction.
        Default implementation does nothing.
        """
        pass