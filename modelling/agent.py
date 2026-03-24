"""Agent — composes all RL components with adaptive decision timing.
Fast-forwards through min_green lockout, decides every step after, forces switch at max_green."""

import math
import numpy as np
from typing import Any

from modelling.components.environment.base  import BaseEnvironment
from modelling.components.observation.base  import BaseObservation
from modelling.components.reward.base       import BaseReward
from modelling.components.policy.base       import BasePolicy
from modelling.components.replay_buffer.base import BaseReplayBuffer


class Agent:

    def __init__(
        self,
        environment:   BaseEnvironment,
        observation:   BaseObservation,
        reward:        BaseReward,
        policy:        BasePolicy,
        replay_buffer: BaseReplayBuffer,
        step_length:   float = 5.0,
        min_green_s:   float = 15.0,
        max_green_s:   float = 90.0,
    ):
        self.environment   = environment
        self.observation   = observation
        self.reward        = reward
        self.policy        = policy
        self.replay_buffer = replay_buffer

        self._step_length       = step_length
        self._min_green_steps   = max(1, math.ceil(min_green_s / step_length))
        self._max_green_steps   = max(1, math.ceil(max_green_s / step_length))

        self._steps_in_phase: dict[str, int] = {}
        self._episode_reward: float = 0.0
        self._episode_steps:  int   = 0
        self._episode_losses: list[float] = []

    def start_episode(self, route_file: str) -> dict[str, np.ndarray]:
        """Reset everything and launch SUMO for one episode. Returns initial observations."""
        self.environment.start(route_file)
        self.observation.reset()
        self.reward.reset()
        self.policy.reset_phase_tracking()

        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_losses = []

        self.environment.step(5)

        tls_ids = self.environment.get_tls_ids()
        if not tls_ids:
            raise RuntimeError(
                "No traffic lights found in simulation. "
                "Check that BuildNetwork produced a valid .net.xml with "
                "a traffic light at J_centre."
            )

        for tls_id in tls_ids:
            self._steps_in_phase[tls_id] = self._min_green_steps

        return self._get_observations()

    def step(self, obs: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray], dict[str, float], bool, float | None,
    ]:
        """One adaptive decision cycle: skip lockout, decide, step, learn."""
        tls_ids = self.environment.get_tls_ids()

        max_remaining = max(
            max(0, self._min_green_steps - self._steps_in_phase.get(tid, self._min_green_steps))
            for tid in tls_ids
        )

        if max_remaining > 0:
            self.environment.step(max_remaining)
            for tid in tls_ids:
                self._steps_in_phase[tid] = (
                    self._steps_in_phase.get(tid, self._min_green_steps) + max_remaining
                )
            if self.environment.is_done():
                return self._get_observations(), {}, True, None
            obs = self._get_observations()

        actions = {}
        for tid in tls_ids:
            if self._steps_in_phase.get(tid, 0) >= self._max_green_steps:
                actions[tid] = 1
            else:
                actions[tid] = self.policy.select_action(obs[tid], tid)

        self._apply_actions(actions)
        self.environment.step(1)

        for tid in tls_ids:
            if actions[tid] == 1:
                self._steps_in_phase[tid] = 0
            else:
                self._steps_in_phase[tid] = self._steps_in_phase.get(tid, 0) + 1

        done     = self.environment.is_done()
        next_obs = self._get_observations()
        rewards  = {tid: self.reward.compute(self.environment.traci, tid) for tid in tls_ids}

        total_reward = sum(rewards.values())
        self._episode_reward += total_reward
        self._episode_steps  += 1

        for tid in tls_ids:
            self.replay_buffer.push(
                state=obs[tid], action=actions[tid], reward=total_reward,
                next_state=next_obs[tid], done=float(done),
            )

        loss = self.policy.update(self.replay_buffer)
        if loss is not None:
            self._episode_losses.append(loss)

        return next_obs, rewards, done, loss

    def end_episode(self) -> dict[str, Any]:
        """Close SUMO and return episode metrics."""
        self.environment.close()
        mean_loss = float(np.mean(self._episode_losses)) if self._episode_losses else None
        return {
            "total_reward": self._episode_reward,
            "steps":        self._episode_steps,
            "mean_loss":    mean_loss,
            "epsilon":      getattr(self.policy, "epsilon", None),
        }

    def set_eval_mode(self) -> None:
        self.policy.set_eval_mode()

    def set_train_mode(self) -> None:
        self.policy.set_train_mode()

    def save(self, path: str) -> None:
        self.policy.save(path)

    def load(self, path: str) -> None:
        self.policy.load(path)

    def _get_observations(self) -> dict[str, np.ndarray]:
        traci = self.environment.traci
        return {
            tls_id: self.observation.build(traci, tls_id)
            for tls_id in self.environment.get_tls_ids()
        }

    def _apply_actions(self, actions: dict[str, int]) -> None:
        """Action 0 = keep current phase, action 1 = advance to next phase."""
        traci = self.environment.traci
        for tls_id, action in actions.items():
            if action == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                n_phases = len(
                    traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
                )
                next_phase = (current_phase + 1) % n_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
