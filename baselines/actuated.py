"""
baselines/actuated.py
----------------------
Actuated (queue-responsive) traffic signal policy.

How it works
------------
The controller monitors the overall queue length every decision step by
reading the normalised halting-vehicle counts from the observation vector.
It holds the current green phase as long as vehicles are still waiting
(queue above threshold), and switches as soon as the queue clears or the
maximum green timer expires.

Why this is a useful baseline
------------------------------
Actuated control is the standard "smart but not learning" approach used
in modern real-world traffic engineering.  Unlike fixed-time, it reacts
to what is actually happening in the intersection.  It represents the
best a well-designed deterministic rule can do.

Comparing DQN against actuated answers the harder question:
  "Is learned behaviour better than a competent hand-crafted heuristic?"

Observation vector layout (from QueueObservation)
---------------------------------------------------
The first ``max_lanes`` elements of the observation vector are the
normalised halting-vehicle counts for each controlled lane (0.0–1.0,
where 1.0 = max_vehicles halting on that lane).  All remaining elements
(phase info, temporal encoding) are ignored by this policy.

Queue metric
-------------
The policy computes the mean normalised halting count across all
active lanes (non-padding entries) and compares it to ``queue_threshold``.

Using the global mean rather than per-phase lanes keeps the policy
self-contained: it works from the observation vector alone without
needing a separate TraCI call or lane-to-phase mapping.  In practice
this is equivalent to "switch when the intersection as a whole is
draining" — a valid and commonly deployed actuated strategy.

Phase constraints (same as DQN and FixedTimePolicy)
-----------------------------------------------------
min_green_steps — hard lower bound; switch is blocked even if the queue
                  has already cleared.  Prevents rapid oscillation.
max_green_steps — hard upper bound; switch is forced even if the queue
                  is still above threshold.  Prevents starvation of the
                  opposing approach.

Usage
-----
    from baselines.actuated import ActuatedPolicy

    policy = ActuatedPolicy(
        max_lanes = 16, # must match QueueObservation max_lanes
        min_green_steps = 1, # min_green_s / decision_gap
        max_green_steps = 3, # max_green_s / decision_gap
        queue_threshold = 0.1, # ~2 vehicles out of max_vehicles=20
    )
    agent = Agent(
        environment = env,
        observation = obs_builder,
        reward = reward,
        policy = policy,
        replay_buffer = replay_buffer,
    )
"""

import json
import os
import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import BasePolicy without modifying any existing file.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modelling.components.policy.base import BasePolicy                  
from modelling.components.replay_buffer.base import BaseReplayBuffer     


class ActuatedPolicy(BasePolicy):
    """
    Queue-responsive actuated signal controller.

    Holds green while the mean normalised queue exceeds ``queue_threshold``.
    Switches as soon as the queue drops below the threshold (vehicles have
    cleared) or ``max_green_steps`` is reached.  Never switches before
    ``min_green_steps`` (prevents oscillation).

    Args:
        max_lanes:  Number of lane slots in the observation vector.
                    Must match ``QueueObservation.max_lanes``
                    (default 16 from config.json).
        min_green_steps: Minimum decision steps before any switch is allowed.
                         Derive from intersection.json: min_green_s ÷ decision_gap.
        max_green_steps: Maximum decision steps before a switch is forced.
                         Derive from intersection.json: max_green_s ÷ decision_gap.
        queue_threshold: Mean normalised queue below which the phase is
                         considered "drained" and a switch is triggered.
                         Normalised units: 0.0 = empty, 1.0 = max_vehicles
                         halting on every lane.
                         Default 0.1 ≈ 2 vehicles per lane (with max_vehicles=20).
    """

    def __init__(
        self,
        max_lanes:       int   = 16,
        min_green_steps: int   = 1,
        max_green_steps: int   = 3,
        queue_threshold: float = 0.1,
    ):
        if max_lanes < 1:
            raise ValueError("max_lanes must be >= 1")
        if min_green_steps < 1:
            raise ValueError("min_green_steps must be >= 1")
        if max_green_steps < min_green_steps:
            raise ValueError("max_green_steps must be >= min_green_steps")
        if not (0.0 <= queue_threshold <= 1.0):
            raise ValueError("queue_threshold must be in [0.0, 1.0]")

        self._max_lanes = max_lanes
        self._min_green_steps = min_green_steps
        self._max_green_steps = max_green_steps
        self._queue_threshold = queue_threshold

        # Step counter per traffic light ID, reset each episode.
        self._steps_since_switch: dict[str, int] = {}

    # ------------------------------------------------------------------
    # QUEUE EXTRACTION
    # ------------------------------------------------------------------

    def _mean_queue(self, obs: np.ndarray) -> float:
        """
        Compute the mean normalised halting count across all active lanes.

        Padding lanes (value == 0.0 at the end of the obs slice) are
        excluded so a sparse intersection doesn't artificially deflate
        the metric.  If every lane reads 0.0 the queue is genuinely
        empty and the mean is returned as 0.0.

        Args:
            obs: Full observation vector from QueueObservation.

        Returns:
            Mean normalised queue in [0.0, 1.0].
        """
        queue_slice = obs[: self._max_lanes]

        # Identify lanes that have ever reported a non-zero count.
        # In a fresh episode the vector is all zeros; treat that as empty.
        active = queue_slice[queue_slice > 0.0]
        if active.size == 0:
            return float(queue_slice.mean())
        return float(active.mean())

    # ------------------------------------------------------------------
    # CORE INTERFACE
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, tls_id: str = "default") -> int:
        """
        Return action 0 (keep) or 1 (switch) for one traffic light.

        Decision logic (evaluated in priority order)
        ---------------------------------------------
        1. steps >= max_green_steps → must switch   (action 1)
        2. steps < min_green_steps → cannot switch  (action 0)
        3. mean_queue < queue_threshold → queue drained, switch (action 1)
        4. otherwise → queue still active, hold (action 0)

        Args:
            obs:    Observation vector from QueueObservation. The first
                    ``max_lanes`` elements are the per-lane halting counts.
            tls_id: Traffic light ID for per-signal step tracking.

        Returns:
            0 = keep current phase, 1 = switch to next phase.
        """
        if tls_id not in self._steps_since_switch:
            self._steps_since_switch[tls_id] = self._min_green_steps

        steps = self._steps_since_switch[tls_id]
        mean_queue = self._mean_queue(obs)

        # Priority 1 — hard upper bound
        if steps >= self._max_green_steps:
            action = 1

        # Priority 2 — hard lower bound
        elif steps < self._min_green_steps:
            action = 0

        # Priority 3 — queue has drained
        elif mean_queue < self._queue_threshold:
            action = 1

        # Priority 4 — queue still active, hold green
        else:
            action = 0

        if action == 1:
            self._steps_since_switch[tls_id] = 0
        else:
            self._steps_since_switch[tls_id] += 1

        return action

    def update(self, replay_buffer: BaseReplayBuffer) -> float | None:
        """No-op. Actuated control does not learn."""
        return None

    # ------------------------------------------------------------------
    # SAVE / LOAD  (saves config as JSON, no weights to store)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy configuration to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        config = {
            "policy": "ActuatedPolicy",
            "max_lanes": self._max_lanes,
            "min_green_steps": self._min_green_steps,
            "max_green_steps": self._max_green_steps,
            "queue_threshold": self._queue_threshold,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str) -> None:
        """Load policy configuration from a JSON file."""
        with open(path) as f:
            config = json.load(f)
        self._max_lanes = config["max_lanes"]
        self._min_green_steps = config["min_green_steps"]
        self._max_green_steps = config["max_green_steps"]
        self._queue_threshold = config["queue_threshold"]

    # ------------------------------------------------------------------
    # MODE SWITCHING  (no-op — no exploration to enable/disable)
    # ------------------------------------------------------------------

    def set_eval_mode(self) -> None:
        pass

    def set_train_mode(self) -> None:
        pass

    # ------------------------------------------------------------------
    # EPISODE RESET  (matches DQNPolicy.reset_phase_tracking interface)
    # ------------------------------------------------------------------

    def reset_phase_tracking(self) -> None:
        """Clear per-signal step counters at the start of each episode."""
        self._steps_since_switch.clear()

    # ------------------------------------------------------------------
    # PROPERTIES  (epsilon=None keeps Trainer logging compatible)
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> None:
        """No epsilon — this policy never explores."""
        return None

    # ------------------------------------------------------------------
    # REPR
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ActuatedPolicy("
            f"max_lanes={self._max_lanes}, "
            f"min_green_steps={self._min_green_steps}, "
            f"max_green_steps={self._max_green_steps}, "
            f"queue_threshold={self._queue_threshold})"
        )
