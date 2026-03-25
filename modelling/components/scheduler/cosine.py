"""Cosine annealing LR scheduler — smooth decay from lr_start to lr_min over total_steps."""

import torch.optim as optim
from .base import BaseScheduler


class CosineScheduler(BaseScheduler):
    """Wraps PyTorch CosineAnnealingLR. Decays slowly at first, faster mid-training,
    then gently approaches lr_min — good fit for DQN where early updates are exploratory."""

    def __init__(self, optimizer: optim.Optimizer, total_steps: int, lr_min: float = 1e-5):
        self._optimizer = optimizer
        self._total_steps = total_steps
        self._lr_min = lr_min
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr_min,
        )

    def step(self) -> None:
        self._scheduler.step()

    def get_lr(self) -> float:
        return self._optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return self._scheduler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._scheduler.load_state_dict(state)
