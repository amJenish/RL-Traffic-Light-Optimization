"""
baselines/fixed_time.py
------------------------
Fixed-Time traffic signal policy.

How it works
------------
The controller holds each green phase for a fixed number of decision steps
(fixed_green_steps), then switches unconditionally to the next phase.
It does NOT look at queue lengths or any other traffic state — the timer
alone drives all switching decisions.

Why this is a useful baseline
------------------------------
Fixed-time control is what most real-world signals do by default.
It represents the simplest possible intelligent controller:
  "optimise once offline, then ignore what is actually happening."

Comparing the DQN agent against fixed-time answers the question:
  "Is learning to react to live queue state better than ignoring it?"

Phase constraints (same as DQN)
---------------------------------
min_green_steps — hard lower bound: the switch action is blocked even
                  if the fixed timer fires early (shouldn't happen in
                  practice if fixed_green_steps >= min_green_steps, but
                  kept for safety).
max_green_steps — hard upper bound: a switch is forced if the phase has
                  been green for this many steps regardless of the timer.
                  Prevents a phase from being held indefinitely.

Both bounds are enforced identically to DQNPolicy so comparisons are
fair — the DQN agent faces the same hard constraints.

Usage
-----
    from baselines.fixed_time import FixedTimePolicy

    policy = FixedTimePolicy(
        fixed_green_steps = 6, # hold green for 6 × 30 s = 3 min
        min_green_steps = 1, # from intersection.json min_green_s / decision_gap
        max_green_steps = 3, # from intersection.json max_green_s / decision_gap
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
# We add the project root to sys.path if needed so the import always works
# regardless of where this script is run from.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modelling.components.policy.base import BasePolicy          # noqa: E402
from modelling.components.replay_buffer.base import BaseReplayBuffer  # noqa: E402


class FixedTimePolicy(BasePolicy):
    """
    Deterministic fixed-time signal controller.

    Switches to the next phase every ``fixed_green_steps`` decision steps,
    regardless of queue state.  Hard min/max green guards are applied
    identically to DQNPolicy so the comparison is fair.

    Args:
        fixed_green_steps: Target number of decision steps to hold each
                           green phase before switching.  One decision step
                           equals ``decision_gap × step_length`` simulation
                           seconds (default config: 30 × 5 s = 150 s).
        min_green_steps: Minimum steps that must pass before a switch is
                         allowed.  Should be derived from intersection.json
                         ``min_green_s`` ÷ decision_gap.
        max_green_steps: Maximum steps before a switch is forced.
                         Should be derived from intersection.json
                         ``max_green_s`` ÷ decision_gap.
    """

    def __init__(
        self,
        fixed_green_steps: int = 6,
        min_green_steps:   int = 1,
        max_green_steps:   int = 3,
    ):
        if fixed_green_steps < 1:
            raise ValueError("fixed_green_steps must be >= 1")
        if min_green_steps < 1:
            raise ValueError("min_green_steps must be >= 1")
        if max_green_steps < min_green_steps:
            raise ValueError("max_green_steps must be >= min_green_steps")

        self._fixed_green_steps = fixed_green_steps
        self._min_green_steps = min_green_steps
        self._max_green_steps = max_green_steps

        # Step counter per traffic light ID, reset each episode.
        self._steps_since_switch: dict[str, int] = {}

    # ------------------------------------------------------------------
    # CORE INTERFACE
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, tls_id: str = "default") -> int:
        """
        Return action 0 (keep) or 1 (switch) for one traffic light.

        Decision logic
        --------------
        1. If steps_since_switch >= max_green_steps  → must switch (action 1).
        2. If steps_since_switch <  min_green_steps  → cannot switch (action 0).
        3. If steps_since_switch >= fixed_green_steps → timer fired (action 1).
        4. Otherwise                                 → hold (action 0).

        Args:
            obs: Observation vector (not used — fixed-time ignores queue state).
            tls_id: Traffic light ID for per-signal step tracking.

        Returns:
            0 = keep current phase, 1 = switch to next phase.
        """
        if tls_id not in self._steps_since_switch:
            self._steps_since_switch[tls_id] = self._min_green_steps

        steps = self._steps_since_switch[tls_id]

        # Hard overrides — identical to DQNPolicy
        if steps >= self._max_green_steps:
            action = 1
        elif steps < self._min_green_steps:
            action = 0
        elif steps >= self._fixed_green_steps:
            action = 1
        else:
            action = 0

        if action == 1:
            self._steps_since_switch[tls_id] = 0
        else:
            self._steps_since_switch[tls_id] += 1

        return action

    def update(self, replay_buffer: BaseReplayBuffer) -> float | None:
        """No-op. Fixed-time control does not learn."""
        return None

    # ------------------------------------------------------------------
    # SAVE / LOAD  (saves config as JSON, no weights to store)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy configuration to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        config = {
            "policy": "FixedTimePolicy",
            "fixed_green_steps": self._fixed_green_steps,
            "min_green_steps": self._min_green_steps,
            "max_green_steps": self._max_green_steps,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str) -> None:
        """Load policy configuration from a JSON file."""
        with open(path) as f:
            config = json.load(f)
        self._fixed_green_steps = config["fixed_green_steps"]
        self._min_green_steps = config["min_green_steps"]
        self._max_green_steps = config["max_green_steps"]

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
            f"FixedTimePolicy("
            f"fixed_green_steps={self._fixed_green_steps}, "
            f"min_green_steps={self._min_green_steps}, "
            f"max_green_steps={self._max_green_steps})"
        )
