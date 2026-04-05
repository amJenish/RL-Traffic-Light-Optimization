"""Throughput delta plus mean vehicle waiting time (and optional cross-lane wait imbalance)."""

import statistics
from typing import Any

from .base import BaseReward
from .throughput import _departure_lanes


def _vehicle_mean_waiting(traci: Any, veh_ids: list[str]) -> float:
    if not veh_ids:
        return 0.0
    total = 0.0
    n = 0
    for vid in veh_ids:
        try:
            total += float(traci.vehicle.getWaitingTime(vid))
            n += 1
        except traci.exceptions.TraCIException:
            continue
    return total / n if n else 0.0


class ThroughputWaitTimeReward(BaseReward):
    """R = gamma*DeltaV - alpha*Wbar - beta*sigma_W per decision interval (SMDP).

    DeltaV: departure-lane throughput since last compute (same as ThroughputQueueReward).
    Wbar: mean vehicle.getWaitingTime over vehicles on TLS controlled lanes (once each).
    sigma_W: pstdev of per-lane mean vehicle waiting times; beta=0 omits this term.
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

        all_veh: set[str] = set()
        per_lane_mean_wait: list[float] = []
        for lane in controlled_lanes:
            try:
                vids = list(traci.lane.getLastStepVehicleIDs(lane))
            except traci.exceptions.TraCIException:
                vids = []
            all_veh.update(vids)
            per_lane_mean_wait.append(_vehicle_mean_waiting(traci, vids))

        if all_veh:
            w_bar = _vehicle_mean_waiting(traci, list(all_veh))
        else:
            w_bar = 0.0

        if len(per_lane_mean_wait) > 1:
            std_w = float(statistics.pstdev(per_lane_mean_wait))
        else:
            std_w = 0.0

        dep_lanes = self._departure_lanes_cache.get(tls_id, [])
        dv_term = float(delta_v)
        if self._normalise:
            if dep_lanes:
                dv_term /= len(dep_lanes)

        reward = self._gamma * dv_term - self._alpha * w_bar
        if self._beta != 0.0:
            reward -= self._beta * std_w
        return float(reward * self._scale)

    def reset(self) -> None:
        self._prev_on_departure.clear()
        self._accumulated_throughput.clear()
        self._departure_lanes_cache.clear()
