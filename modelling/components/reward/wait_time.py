"""
modeling/components/reward/wait_time.py
-----------------------------------------
Instantaneous queue reward.

Reward = negative count of halting vehicles across all controlled lanes
         at this exact decision step.

Why halting count and not waiting time:
  - getWaitingTime() is cumulative — always grows, agent can't distinguish
    good from bad decisions
  - getLastStepHaltingNumber() is instantaneous — goes up when queues build,
    goes down when vehicles clear. Agent gets a clean signal about whether
    its last action was good.

The overstay constraint (max_green) is enforced at the policy layer via
action masking, not here. Keeping reward and constraint logic separate
makes both easier to reason about and swap independently.
"""

from typing import Any
from .base import BaseReward


class WaitTimeReward(BaseReward):
    """
    Instantaneous halting-count reward.

    Args:
        normalise: Divide by lane count so reward scale is consistent
                   regardless of intersection size.
        scale:     Multiplier. Increase to make reward signal stronger
                   relative to other loss terms.
    """

    def __init__(
        self,
        normalise: bool  = True,
        scale:     float = 1.0,
    ):
        self._normalise = normalise
        self._scale     = scale

    def compute(self, traci: Any, tls_id: str) -> float:
        """
        Returns negative halting vehicle count at this decision step.
        Higher (less negative) = fewer vehicles stopped = better.
        Returns 0.0 if no lanes are controlled.
        """
        lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tls_id)
        ))

        if not lanes:
            return 0.0

        total_halting = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in lanes
        )

        if self._normalise:
            total_halting /= len(lanes)

        return -total_halting * self._scale

    def reset(self) -> None:
        """No internal state to reset."""
        pass