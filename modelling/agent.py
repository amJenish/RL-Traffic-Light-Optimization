"""
modeling/agent.py
------------------
The Agent holds all RL components and exposes a clean interface
that the Trainer calls. It never builds components internally —
all components are injected from outside.

Usage:
    agent = Agent(
        environment   = SumoEnvironment(...),
        observation   = QueueObservation(...),
        reward        = WaitTimeReward(...),
        policy        = DQNPolicy(...),
        replay_buffer = UniformReplayBuffer(...),
    )
    trainer = Trainer(agent=agent, split=split_cfg)
    trainer.run()
"""

import numpy as np
from typing import Any

from modelling.components.environment.base  import BaseEnvironment
from modelling.components.observation.base  import BaseObservation
from modelling.components.reward.base       import BaseReward
from modelling.components.policy.base       import BasePolicy
from modelling.components.replay_buffer.base import BaseReplayBuffer


class Agent:
    """
    Composes all RL components and coordinates a single episode step.

    Args:
        environment:   Simulation wrapper (e.g. SumoEnvironment).
        observation:   Observation builder (e.g. QueueObservation).
        reward:        Reward function (e.g. WaitTimeReward).
        policy:        Learning policy (e.g. DQNPolicy).
        replay_buffer: Experience store (e.g. UniformReplayBuffer).
    """

    def __init__(
        self,
        environment:   BaseEnvironment,
        observation:   BaseObservation,
        reward:        BaseReward,
        policy:        BasePolicy,
        replay_buffer: BaseReplayBuffer,
    ):
        self.environment   = environment
        self.observation   = observation
        self.reward        = reward
        self.policy        = policy
        self.replay_buffer = replay_buffer

        # Episode-level metrics
        self._episode_reward: float = 0.0
        self._episode_steps:  int   = 0
        self._episode_losses: list[float] = []

    # ------------------------------------------------------------------
    # EPISODE LIFECYCLE
    # ------------------------------------------------------------------

    def start_episode(self, route_file: str) -> dict[str, np.ndarray]:
        """
        Start a new episode.

        Args:
            route_file: Path to the SUMO .rou.xml for this episode.

        Returns:
            Initial observations keyed by tls_id.
        """
        self.environment.start(route_file)
        self.observation.reset()
        self.reward.reset()
        self.policy.reset_phase_tracking()

        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_losses = []

        # Advance a few steps so vehicles have time to spawn
        self.environment.step(5)

        # Validate TraCI is live
        tls_ids = self.environment.get_tls_ids()
        if not tls_ids:
            raise RuntimeError(
                "No traffic lights found in simulation. "
                "Check that BuildNetwork produced a valid .net.xml with "
                "a traffic light at J_centre."
            )

        return self._get_observations()

    def step(self, obs: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray],   # next observations
        dict[str, float],        # rewards
        bool,                    # done
        float | None,            # loss (if update performed)
    ]:
        """
        Execute one decision step for all controlled traffic lights.

        1. Select actions for each tls_id
        2. Apply actions to the simulation
        3. Advance simulation by one decision gap
        4. Collect next observations and rewards
        5. Store transitions and perform a learning update

        Args:
            obs: Current observations keyed by tls_id.

        Returns:
            (next_obs, rewards, done, loss)
        """
        tls_ids = self.environment.get_tls_ids()

        # 1. Select actions
        actions = {
            tls_id: self.policy.select_action(obs[tls_id], tls_id)
            for tls_id in tls_ids
        }

        # 2. Apply actions to SUMO
        self._apply_actions(actions)

        # 3. Advance simulation
        self.environment.step_decision()

        done = self.environment.is_done()

        # 4. Collect next observations and rewards
        next_obs = self._get_observations()
        rewards  = {
            tls_id: self.reward.compute(self.environment.traci, tls_id)
            for tls_id in tls_ids
        }

        # Cooperative reward — all agents share the sum
        total_reward = sum(rewards.values())
        self._episode_reward += total_reward
        self._episode_steps  += 1

        # 5. Store transitions
        for tls_id in tls_ids:
            self.replay_buffer.push(
                state      = obs[tls_id],
                action     = actions[tls_id],
                reward     = total_reward,
                next_state = next_obs[tls_id],
                done       = float(done),
            )

        # 6. Learning update
        loss = self.policy.update(self.replay_buffer)
        if loss is not None:
            self._episode_losses.append(loss)

        return next_obs, rewards, done, loss

    def end_episode(self) -> dict[str, Any]:
        """
        Close the environment and return episode summary metrics.

        Returns:
            dict with keys: total_reward, steps, mean_loss, epsilon
        """
        self.environment.close()

        mean_loss = (
            float(np.mean(self._episode_losses))
            if self._episode_losses else None
        )

        return {
            "total_reward": self._episode_reward,
            "steps":        self._episode_steps,
            "mean_loss":    mean_loss,
            "epsilon":      getattr(self.policy, "epsilon", None),
        }

    # ------------------------------------------------------------------
    # EVALUATION MODE
    # ------------------------------------------------------------------

    def set_eval_mode(self) -> None:
        """Switch policy to greedy evaluation — no exploration."""
        self.policy.set_eval_mode()

    def set_train_mode(self) -> None:
        """Switch policy back to training mode with exploration."""
        self.policy.set_train_mode()

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy weights to disk."""
        self.policy.save(path)

    def load(self, path: str) -> None:
        """Load policy weights from disk."""
        self.policy.load(path)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Build observations for all traffic lights."""
        traci = self.environment.traci
        return {
            tls_id: self.observation.build(traci, tls_id)
            for tls_id in self.environment.get_tls_ids()
        }

    def _apply_actions(self, actions: dict[str, int]) -> None:
        """
        Apply actions to SUMO traffic lights.

        Action 0 = keep current phase.
        Action 1 = advance to next phase.
        """
        traci = self.environment.traci
        for tls_id, action in actions.items():
            if action == 1:
                current_phase  = traci.trafficlight.getPhase(tls_id)
                n_phases       = len(
                    traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
                )
                next_phase = (current_phase + 1) % n_phases
                traci.trafficlight.setPhase(tls_id, next_phase)