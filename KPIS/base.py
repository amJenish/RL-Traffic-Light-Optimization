"""Abstract episode-level KPI (independent of training reward signal)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EpisodeKpi(ABC):
    """Accumulates per-SUMO-step statistics and exposes episode-level scalars."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        ...

    @abstractmethod
    def contribute_episode_metrics(
        self, metrics: dict[str, Any], elapsed_s: float
    ) -> None:
        """Mutate ``metrics`` with this KPI's keys (``kpi_*`` and legacy aliases)."""
