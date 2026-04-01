"""Queue-based observation: lane halting counts + phase info + time-of-day encoding."""

import math
import numpy as np
from typing import Any

from .base import BaseObservation


class QueueObservation(BaseObservation):

    def __init__(
        self,
        max_lanes: int = 16,
        max_phase: int = 3,
        max_phase_time: float = 120.0,
        max_vehicles: int = 20,
        max_green_s: float | None = None,
    ):
        self._max_lanes = max_lanes
        self._max_phase = max_phase
        self._max_phase_time = max_phase_time
        self._max_vehicles = max_vehicles
        self._max_green_s = max_green_s
        self._phase_start: dict[str, float] = {}
        self._current_phase: dict[str, int] = {}

    def _time_in_phase_norms(self, time_in_phase: float) -> tuple[float, float | None]:
        """Generic time cap + optional elapsed / max_green (aligns with Agent overshoot)."""
        time_norm = min(time_in_phase / self._max_phase_time, 1.0)
        green_ratio = None
        if self._max_green_s is not None and self._max_green_s > 0:
            green_ratio = min(time_in_phase / self._max_green_s, 2.0)
        return time_norm, green_ratio

    def build(self, traci: Any, tls_id: str) -> np.ndarray:
        """Build [lane_queues | phase | time_norm | (+ green_elapsed_ratio) | time encodings]."""
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        halting = np.zeros(self._max_lanes, dtype=np.float32)
        for i, lane in enumerate(lanes[: self._max_lanes]):
            count = traci.lane.getLastStepHaltingNumber(lane)
            halting[i] = min(count / self._max_vehicles, 1.0)

        phase = traci.trafficlight.getPhase(tls_id)
        sim_time = traci.simulation.getTime()

        if tls_id not in self._current_phase or self._current_phase[tls_id] != phase:
            self._current_phase[tls_id] = phase
            self._phase_start[tls_id] = sim_time

        time_in_phase = sim_time - self._phase_start.get(tls_id, sim_time)
        phase_norm = phase / max(self._max_phase, 1)
        time_norm, green_ratio = self._time_in_phase_norms(time_in_phase)

        hour = (sim_time % 86400) / 3600.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # dow derived from sim_time is always 0 (sim starts at 28800s < 86400s),
        # so dow_sin/cos carry no information — kept as zeros to preserve obs_dim.
        dow_sin = 0.0
        dow_cos = 0.0

        tail = [phase_norm, time_norm, hour_sin, hour_cos, dow_sin, dow_cos]
        if green_ratio is not None:
            tail.append(green_ratio)
        obs = np.concatenate([
            halting,
            np.array(tail, dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def size(self) -> int:
        n = self._max_lanes + 6
        if self._max_green_s is not None:
            n += 1
        return n

    def reset(self) -> None:
        self._phase_start.clear()
        self._current_phase.clear()

    def set_day_of_week(self, dow: int) -> None:
        """Override the day-of-week for the current episode (0=Mon .. 6=Sun)."""
        self._dow = dow

    def build_with_dow(self, traci: Any, tls_id: str, dow: int) -> np.ndarray:
        """Same as build() but uses an explicit day-of-week instead of deriving it."""
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        halting = np.zeros(self._max_lanes, dtype=np.float32)
        for i, lane in enumerate(lanes[: self._max_lanes]):
            count = traci.lane.getLastStepHaltingNumber(lane)
            halting[i] = min(count / self._max_vehicles, 1.0)

        phase = traci.trafficlight.getPhase(tls_id)
        sim_time = traci.simulation.getTime()

        if tls_id not in self._current_phase or self._current_phase[tls_id] != phase:
            self._current_phase[tls_id] = phase
            self._phase_start[tls_id] = sim_time

        time_in_phase = sim_time - self._phase_start.get(tls_id, sim_time)
        phase_norm = phase / max(self._max_phase, 1)
        time_norm, green_ratio = self._time_in_phase_norms(time_in_phase)

        hour = (sim_time % 86400) / 3600.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * dow / 7)
        dow_cos = math.cos(2 * math.pi * dow / 7)

        tail = [phase_norm, time_norm, hour_sin, hour_cos, dow_sin, dow_cos]
        if green_ratio is not None:
            tail.append(green_ratio)
        obs = np.concatenate([
            halting,
            np.array(tail, dtype=np.float32),
        ])
        return obs.astype(np.float32)
