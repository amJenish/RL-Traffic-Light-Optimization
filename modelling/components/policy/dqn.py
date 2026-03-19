"""
modeling/components/policy/dqn.py
-----------------------------------
Deep Q-Network (DQN) policy implementation.

Architecture:
    Linear(obs_dim → 128) → ReLU
    Linear(128 → 128)     → ReLU
    Linear(128 → n_actions)

Training:
    - Epsilon-greedy exploration with linear decay
    - Experience replay via the passed replay buffer
    - Periodic target network synchronisation
    - Bellman target: r + gamma * (1 - done) * max_a' Q_target(s', a')
    - MSE loss
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BasePolicy
from ..replay_buffer.base import BaseReplayBuffer



# NEURAL NETWORK

class _QNetwork(nn.Module):
    """Simple 2-hidden-layer Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,  hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# DQN POLICY

class DQNPolicy(BasePolicy):
    """
    Deep Q-Network policy with epsilon-greedy exploration.

    Args:
        obs_dim:          Size of the observation vector. Must match
                          BaseObservation.size() of observation builder.
        n_actions:        Number of discrete actions (2 for keep/switch).
        lr:               Learning rate for Adam optimiser.
        gamma:            Discount factor.
        epsilon_start:    Initial exploration rate (1.0 = fully random).
        epsilon_end:      Minimum exploration rate.
        epsilon_decay:    Multiplicative decay applied each update step.
        target_update:    Number of update steps between target network syncs.
        batch_size:       Transitions sampled per update.
        hidden:           Hidden layer size.
        device:           'cuda', 'cpu', or 'auto' (picks cuda if available).
        min_green_steps:  Minimum number of decision steps a phase must stay
                          green before the agent can switch away.
                          Hard stop — switch action is blocked until met.
    """

    def __init__(
        self,
        obs_dim:         int,
        n_actions:       int   = 2,
        lr:              float = 1e-3,
        gamma:           float = 0.99,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   float = 0.995,
        target_update:   int   = 100,
        batch_size:      int   = 64,
        hidden:          int   = 128,
        device:          str   = "auto",
        min_green_steps: int   = 15,
        max_green_steps: int   = 90,
    ):
        self._obs_dim       = obs_dim
        self._n_actions     = n_actions
        self._gamma         = gamma
        self._epsilon       = epsilon_start
        self._epsilon_end   = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._target_update = target_update
        self._batch_size    = batch_size
        self._min_green_steps = min_green_steps
        self._max_green_steps = max_green_steps
        self._update_count  = 0
        self._eval_mode     = False

        # Phase constraint tracking — keyed by tls_id
        self._steps_since_switch: dict[str, int] = {}

        # Device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Networks
        self._online = _QNetwork(obs_dim, n_actions, hidden).to(self._device)
        self._target = _QNetwork(obs_dim, n_actions, hidden).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optimiser = optim.Adam(self._online.parameters(), lr=lr)
        self._loss_fn   = nn.MSELoss()

    # ABSTRACT METHOD IMPLEMENTATIONS

    def select_action(self, obs: np.ndarray, tls_id: str = "default") -> int:
        """
        Select action using epsilon-greedy policy.

        Enforces min_red constraint — action 1 (switch) is blocked
        if the phase has not been red long enough.

        Args:
            obs:     Observation vector.
            tls_id:  Traffic light ID (used for min_red tracking).

        Returns:
            0 = keep current phase
            1 = switch to next phase
        """
        # Initialise step counter for new tls_ids
        if tls_id not in self._steps_since_switch:
            self._steps_since_switch[tls_id] = self._min_green_steps

        can_switch  = self._steps_since_switch[tls_id] >= self._min_green_steps
        must_switch = self._steps_since_switch[tls_id] >= self._max_green_steps

        # Hard override — must switch if max_green exceeded
        if must_switch:
            action = 1
        # Hard override — cannot switch if min_green not yet met
        elif not can_switch:
            action = 0
        # Exploration
        elif not self._eval_mode and np.random.random() < self._epsilon:
            action = np.random.randint(self._n_actions)
        else:
            # Greedy
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32,
                                 device=self._device).unsqueeze(0)
                q_values = self._online(t).squeeze(0)
                action = int(q_values.argmax().item())

        # Update step counter
        if action == 1:
            self._steps_since_switch[tls_id] = 0
        else:
            self._steps_since_switch[tls_id] += 1

        return action

    def update(self, replay_buffer: BaseReplayBuffer) -> float | None:
        """
        One DQN update step using a batch from the replay buffer.

        Returns loss value or None if buffer not ready.
        """
        if not replay_buffer.is_ready(self._batch_size):
            return None

        states, actions, rewards, next_states, dones = \
            replay_buffer.sample(self._batch_size)

        states      = torch.tensor(states,      device=self._device)
        actions     = torch.tensor(actions,     device=self._device)
        rewards     = torch.tensor(rewards,     device=self._device)
        next_states = torch.tensor(next_states, device=self._device)
        dones       = torch.tensor(dones,       device=self._device)

        # Current Q values
        q_current = self._online(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Bellman target
        with torch.no_grad():
            q_next  = self._target(next_states).max(1).values
            q_target = rewards + self._gamma * (1 - dones) * q_next

        loss = self._loss_fn(q_current, q_target)

        self._optimiser.zero_grad()
        loss.backward()
        self._optimiser.step()

        # Decay epsilon
        if not self._eval_mode:
            self._epsilon = max(
                self._epsilon_end,
                self._epsilon * self._epsilon_decay,
            )

        # Sync target network
        self._update_count += 1
        if self._update_count % self._target_update == 0:
            self._target.load_state_dict(self._online.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save online network weights and training state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online":        self._online.state_dict(),
            "target":        self._target.state_dict(),
            "optimiser":     self._optimiser.state_dict(),
            "epsilon":       self._epsilon,
            "update_count":  self._update_count,
            "obs_dim":       self._obs_dim,
            "n_actions":     self._n_actions,
        }, path)

    def load(self, path: str) -> None:
        """Load weights and training state from disk."""
        ck = torch.load(path, map_location=self._device)
        self._online.load_state_dict(ck["online"])
        self._target.load_state_dict(ck["target"])
        self._optimiser.load_state_dict(ck["optimiser"])
        self._epsilon     = ck.get("epsilon",      self._epsilon_end)
        self._update_count = ck.get("update_count", 0)

    def set_eval_mode(self) -> None:
        """Disable exploration for evaluation runs."""
        self._eval_mode = True
        self._online.eval()

    def set_train_mode(self) -> None:
        """Re-enable exploration for training runs."""
        self._eval_mode = False
        self._online.train()

    def reset_phase_tracking(self) -> None:
        """Clear min_red counters at the start of each episode."""
        self._steps_since_switch.clear()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def device(self) -> torch.device:
        return self._device