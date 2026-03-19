"""
modeling/components/observation/queue_observation.py
------------------------------------------------------
Queue-based observation implementation.

Builds a feature vector containing:
  - Halting vehicle count per controlled lane (normalised)
  - Current phase index (normalised)
  - Time spent in current phase (normalised)
  - Time of day encoded as sin/cos (captures daily traffic patterns)
  - Day of week encoded as sin/cos (captures Mon-Sun variation)

The temporal features allow the agent to learn that Monday 08:30
is different from Sunday 08:30 — essential for Option B deployment
where the policy must generalise across all days of the week.
"""

import math
import numpy as np
from typing import Any

from .base import BaseObservation


class QueueObservation(BaseObservation):
    """
    Observation vector:
      [halting_per_lane (max_lanes),
       phase_norm,
       time_in_phase_norm,
       hour_sin, hour_cos,
       dow_sin,  dow_cos]

    Total size = max_lanes + 6

    Args:
        max_lanes:        Maximum number of controlled lanes expected.
                          Observation is zero-padded to this length so the
                          vector size is fixed regardless of intersection size.
        max_phase:        Maximum phase index (number of phases - 1).
        max_phase_time:   Maximum time in phase before normalisation clips to 1.
                          Set to your cycle length in seconds.
        max_vehicles:     Maximum vehicles per lane for normalisation.
    """

    def __init__(
        self,
        max_lanes:      int   = 16,
        max_phase:      int   = 3,
        max_phase_time: float = 120.0,
        max_vehicles:   int   = 20,
    ):
        self._max_lanes      = max_lanes
        self._max_phase      = max_phase
        self._max_phase_time = max_phase_time
        self._max_vehicles   = max_vehicles

        # Track time in phase internally
        self._phase_start:   dict[str, float] = {}
        self._current_phase: dict[str, int]   = {}

    def build(self, traci: Any, tls_id: str) -> np.ndarray:
        """
        Build observation vector for one traffic light.

        Returns float32 array of length self.size().
        """
        # --- halting counts per lane ---
        lanes   = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tls_id)
        ))
        halting = np.zeros(self._max_lanes, dtype=np.float32)
        for i, lane in enumerate(lanes[:self._max_lanes]):
            count      = traci.lane.getLastStepHaltingNumber(lane)
            halting[i] = min(count / self._max_vehicles, 1.0)

        # --- current phase ---
        phase    = traci.trafficlight.getPhase(tls_id)
        sim_time = traci.simulation.getTime()

        # Track phase changes to compute time in phase
        if tls_id not in self._current_phase or self._current_phase[tls_id] != phase:
            self._current_phase[tls_id] = phase
            self._phase_start[tls_id]   = sim_time

        time_in_phase = sim_time - self._phase_start.get(tls_id, sim_time)

        # --- normalise phase features ---
        phase_norm = phase / max(self._max_phase, 1)
        time_norm  = min(time_in_phase / self._max_phase_time, 1.0)

        # --- temporal features from sim time ---
        # sim_time is seconds since midnight
        hour        = (sim_time % 86400) / 3600.0
        hour_sin    = math.sin(2 * math.pi * hour / 24)
        hour_cos    = math.cos(2 * math.pi * hour / 24)

        # Day of week from sim time
        # sim_time counts up across the full episode (one day = 86400s)
        # We use day index from the environment — passed via traci simulation day
        # Fallback: derive from sim_time if > 86400 (multi-day sim)
        dow         = int((sim_time // 86400)) % 7
        dow_sin     = math.sin(2 * math.pi * dow / 7)
        dow_cos     = math.cos(2 * math.pi * dow / 7)

        obs = np.concatenate([
            halting,
            np.array([
                phase_norm,
                time_norm,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
            ], dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def size(self) -> int:
        """Length of the observation vector: max_lanes + 6 temporal features."""
        return self._max_lanes + 6

    def reset(self) -> None:
        """Clear internal phase tracking at the start of each episode."""
        self._phase_start.clear()
        self._current_phase.clear()

    def set_day_of_week(self, dow: int) -> None:
        """
        Explicitly set the day of week for the current episode.
        Call this from the Agent at the start of each episode
        so the observation reflects the correct day.

        Args:
            dow: 0=Monday ... 6=Sunday
        """
        self._dow = dow

    def build_with_dow(self, traci: Any, tls_id: str, dow: int) -> np.ndarray:
        """
        Build observation with an explicitly provided day of week.
        Use this instead of build() when the day CSV has a day_of_week column.

        Args:
            traci:   Active TraCI connection.
            tls_id:  Traffic light ID.
            dow:     Day of week (0=Mon ... 6=Sun) from the day CSV.
        """
        lanes   = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tls_id)
        ))
        halting = np.zeros(self._max_lanes, dtype=np.float32)
        for i, lane in enumerate(lanes[:self._max_lanes]):
            count      = traci.lane.getLastStepHaltingNumber(lane)
            halting[i] = min(count / self._max_vehicles, 1.0)

        phase    = traci.trafficlight.getPhase(tls_id)
        sim_time = traci.simulation.getTime()

        if tls_id not in self._current_phase or self._current_phase[tls_id] != phase:
            self._current_phase[tls_id] = phase
            self._phase_start[tls_id]   = sim_time

        time_in_phase = sim_time - self._phase_start.get(tls_id, sim_time)
        phase_norm    = phase / max(self._max_phase, 1)
        time_norm     = min(time_in_phase / self._max_phase_time, 1.0)

        hour     = (sim_time % 86400) / 3600.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin  = math.sin(2 * math.pi * dow / 7)
        dow_cos  = math.cos(2 * math.pi * dow / 7)

        obs = np.concatenate([
            halting,
            np.array([
                phase_norm,
                time_norm,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
            ], dtype=np.float32),
        ])
        return obs.astype(np.float32)