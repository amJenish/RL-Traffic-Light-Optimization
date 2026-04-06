"""
baselines/
----------
Rule-based traffic signal controllers used as comparison baselines
against the trained DQN agent.

Controllers
-----------
FixedTimePolicy   — cycles phases on a fixed timer, ignoring queue state.
ActuatedPolicy    — holds green while queues are above a threshold;
                    switches when the queue clears or max green is hit.
WebsterPolicy     — classical Webster (1958) timing + gap-out from queue obs.

All implement the same select_action / update / save / load interface
as DQNPolicy so they can be dropped into an existing Agent unchanged.
Neither modifies any existing modelling/ code.
"""

from .fixed_time import FixedTimePolicy
from .actuated   import ActuatedPolicy
from .webster    import WebsterPolicy
