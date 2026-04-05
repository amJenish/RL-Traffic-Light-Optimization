"""Throughput + mean queue + cross-lane queue imbalance (std)."""

import statistics
from typing import Any

from .base import BaseReward
from .throughput import _departure_lanes


class ThroughputCompositeReward(BaseReward):
    """R = γ·ΔV − α·Q̄ − β·σ_Q per decision interval (SMDP).

    - ΔV: vehicles newly seen on departure (post-intersection) lanes since the last
      :meth:`compute`, accumulated in :meth:`on_simulation_step` (same as
      ThroughputQueueReward).
    - Q̄: mean of per-lane halting counts on TLS controlled lanes at decision time.
    - σ_Q: population std of those per-lane halting counts (0 if fewer than two lanes).

    ΔV and Q̄ are correlated (clearing reduces queues); both terms reinforce clearing
    while β penalizes starving some lanes. If the policy over-optimizes throughput at
    the expense of fairness, increase β before reducing γ.

    Designed for SMDP-style decision intervals: call :meth:`on_simulation_step` every
    SUMO step, :meth:`compute` only at decision epochs.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.1,
        normalise: bool = True,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._gamma = float(gamma)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._normalise = normalise
        self._scale = scale

        self._prev_on_departure: dict[str, frozenset[str]] = {}
        self._accumulated_throughput: dict[str, int] = {}
        self._departure_lanes_cache: dict[str, list[str]] = {}

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        if tls_id not in self._departure_lanes_cache:
            self._departure_lanes_cache[tls_id] = _departure_lanes(traci, tls_id)

        dep_lanes = self._departure_lanes_cache[tls_id]
        current: set[str] = set()
        for lane in dep_lanes:
            try:
                current.update(traci.lane.getLastStepVehicleIDs(lane))
            except traci.exceptions.TraCIException:
                continue

        current_f = frozenset(current)
        if tls_id not in self._prev_on_departure:
            self._prev_on_departure[tls_id] = current_f
            return

        prev = self._prev_on_departure[tls_id]
        passed = len(current - set(prev))
        self._prev_on_departure[tls_id] = current_f

        if accumulate:
            self._accumulated_throughput[tls_id] = (
                self._accumulated_throughput.get(tls_id, 0) + passed
            )

    def compute(
        self, traci: Any, tls_id: str, *, switched: bool = False
    ) -> float:
        delta_v = float(self._accumulated_throughput.pop(tls_id, 0))

        controlled_lanes = list(
            dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id))
        )
        per_lane_q: list[float] = []
        for lane in controlled_lanes:
            try:
                per_lane_q.append(
                    float(traci.lane.getLastStepHaltingNumber(lane))
                )
            except traci.exceptions.TraCIException:
                per_lane_q.append(0.0)

        if per_lane_q:
            mean_q = float(sum(per_lane_q) / len(per_lane_q))
            std_q = (
                float(statistics.pstdev(per_lane_q))
                if len(per_lane_q) > 1
                else 0.0
            )
        else:
            mean_q = 0.0
            std_q = 0.0

        dep_lanes = self._departure_lanes_cache.get(tls_id, [])
        dv_term = float(delta_v)
        if self._normalise:
            if dep_lanes:
                dv_term /= len(dep_lanes)

        reward = (
            self._gamma * dv_term
            - self._alpha * mean_q
            - self._beta * std_q
        )
        return float(reward * self._scale)

    def reset(self) -> None:
        self._prev_on_departure.clear()
        self._accumulated_throughput.clear()
        self._departure_lanes_cache.clear()
