"""Double DQN — online network selects the best next action, target network evaluates it."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BasePolicy
from ..replay_buffer.base import BaseReplayBuffer


class _QNetwork(nn.Module):
    """Two hidden-layer feedforward Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,  hidden), nn.ReLU(),
            nn.Linear(hidden,   hidden), nn.ReLU(),
            nn.Linear(hidden,   n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQNPolicy(BasePolicy):
    """Double DQN. Reduces overestimation bias vs vanilla DQN."""

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
    ):
        self._obs_dim       = obs_dim
        self._n_actions     = n_actions
        self._gamma         = gamma
        self._epsilon       = epsilon_start
        self._epsilon_end   = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._target_update = target_update
        self._batch_size    = batch_size
        self._update_count  = 0
        self._eval_mode     = False

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._online = _QNetwork(obs_dim, n_actions, hidden).to(self._device)
        self._target = _QNetwork(obs_dim, n_actions, hidden).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._optimiser = optim.Adam(self._online.parameters(), lr=lr)
        self._loss_fn   = nn.SmoothL1Loss()  # Huber loss — robust to large TD errors

    def select_action(self, obs: np.ndarray, tls_id: str = "default") -> int:
        """Epsilon-greedy: random with prob epsilon, else greedy from Q-values."""
        if not self._eval_mode and np.random.random() < self._epsilon:
            return np.random.randint(self._n_actions)

        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32,
                             device=self._device).unsqueeze(0)
            q_values = self._online(t).squeeze(0)
            return int(q_values.argmax().item())

    def update(self, replay_buffer: BaseReplayBuffer) -> float | None:
        """Double DQN update — online picks best action, target evaluates it."""
        if not replay_buffer.is_ready(self._batch_size):
            return None

        states, actions, rewards, next_states, dones, durations = \
            replay_buffer.sample(self._batch_size)

        states      = torch.tensor(states,      dtype=torch.float32, device=self._device)
        actions     = torch.tensor(actions,     dtype=torch.long,    device=self._device)
        rewards     = torch.tensor(rewards,     dtype=torch.float32, device=self._device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self._device)
        dones       = torch.tensor(dones,       dtype=torch.float32, device=self._device)
        durations   = torch.tensor(durations,   dtype=torch.float32, device=self._device)

        q_current = self._online(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            best_next_actions = self._online(next_states).argmax(1)
            q_next = self._target(next_states).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)
            gamma_t  = torch.tensor(self._gamma, dtype=torch.float32, device=self._device)
            discount = torch.pow(gamma_t, durations)
            q_target = rewards + discount * (1 - dones) * q_next

        loss = self._loss_fn(q_current, q_target)

        self._optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=1.0)
        self._optimiser.step()

        if not self._eval_mode:
            self._epsilon = max(
                self._epsilon_end,
                self._epsilon * self._epsilon_decay,
            )

        self._update_count += 1
        if self._update_count % self._target_update == 0:
            self._target.load_state_dict(self._online.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        """Persist both networks, optimiser state, epsilon, and update count."""
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
        """Restore from a checkpoint .pt file."""
        ck = torch.load(path, map_location=self._device)
        self._online.load_state_dict(ck["online"])
        self._target.load_state_dict(ck["target"])
        self._optimiser.load_state_dict(ck["optimiser"])
        self._epsilon      = ck.get("epsilon",      self._epsilon_end)
        self._update_count = ck.get("update_count", 0)

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        self._online.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self._online.train()

    def reset_phase_tracking(self) -> None:
        pass

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def device(self) -> torch.device:
        return self._device
