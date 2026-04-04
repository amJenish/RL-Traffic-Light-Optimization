"""Agent — composes all RL components with adaptive decision timing.
Fast-forwards through min_green lockout, decides every step after,
soft-penalizes holding past max_green instead of forcing a switch."""

import math
import os
import numpy as np
from typing import Any, Dict, List

from modelling.components.environment.base   import BaseEnvironment
from modelling.components.observation.base   import BaseObservation
from modelling.components.reward.base        import BaseReward
from modelling.components.policy.base        import BasePolicy
from modelling.components.replay_buffer.base import BaseReplayBuffer
from modelling.components.scheduler.base     import BaseScheduler


class Agent:

    def __init__(
        self,
        environment:     BaseEnvironment,
        observation:     BaseObservation,
        reward:          BaseReward,
        policy:          BasePolicy,
        replay_buffer:   BaseReplayBuffer,
        scheduler:       BaseScheduler | None = None,
        eval_reward:     BaseReward | None    = None,
        step_length:     float = 1.0,
        min_green_s:     float = 15.0,
        max_green_s:     float = 90.0,
        overshoot_coeff: float = 4.0,
        yellow_duration_s: float = 4.0,
    ):
        self.environment   = environment
        self.observation   = observation
        self.reward        = reward
        self.policy        = policy
        self.replay_buffer = replay_buffer
        self.scheduler     = scheduler
        self.eval_reward   = eval_reward

        self._step_length       = step_length
        self._yellow_duration_s = yellow_duration_s
        self._yellow_steps      = max(1, round(yellow_duration_s / step_length))
        self._min_green_steps   = max(1, math.ceil(min_green_s / step_length))
        self._max_green_steps   = max(1, math.ceil(max_green_s / step_length))
        self._overshoot_coeff   = overshoot_coeff

        self._steps_in_phase: dict[str, int] = {}
        self._episode_reward: float = 0.0
        self._episode_steps:  int   = 0
        self._episode_losses: list[float] = []
        # Pending transition to be finalized at the next decision epoch.
        # (state, action) are measured at the decision time; reward+next_state are
        # finalized when min-green lockout ends (or the episode terminates).
        self._pending_state: dict[str, np.ndarray] | None = None
        self._pending_action: dict[str, int] | None = None
        self._pending_duration: int = 0  # primitive simulation steps since pending decision
        self._train_mode: bool = True
        self._phase_log: Dict[str, List[Dict[str, Any]]] = {}

    def _tls_rewards(self, tls_ids: list[str]) -> dict[str, float]:
        active = (
            self.eval_reward
            if (not self._train_mode and self.eval_reward is not None)
            else self.reward
        )
        pa = self._pending_action
        out = {
            tid: active.compute(
                self.environment.traci,
                tid,
                switched=(pa is not None and pa.get(tid) == 1),
            )
            for tid in tls_ids
        }
        if self._train_mode:
            for tid in tls_ids:
                out[tid] *= self._overshoot_scale(tid)
        return out

    def start_episode(self, route_file: str) -> dict[str, np.ndarray]:
        """Reset everything and launch SUMO for one episode. Returns initial observations."""
        self.environment.start(route_file)
        self.observation.reset()
        self.reward.reset()
        if self.eval_reward is not None:
            self.eval_reward.reset()
        self.policy.reset_phase_tracking()

        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_losses = []
        self._pending_state   = None
        self._pending_action  = None
        self._pending_duration = 0
        self._phase_log = {}

        warmup_steps = max(5, round(25 / self._step_length))
        for _ in range(warmup_steps):
            self.environment.step(1)
            self._notify_simulation_step(accumulate_reward=False)

        # Land on green before the first decision. _pending_state is still None here
        # (only cleared above; first step() sets it), so yellow substeps must not touch
        # _pending_duration — accumulate_reward=False is correct.
        traci_w = self.environment.traci
        for tid in list(traci_w.trafficlight.getIDList()):
            logic0 = traci_w.trafficlight.getAllProgramLogics(tid)[0]
            n0 = len(logic0.phases)
            cur0 = traci_w.trafficlight.getPhase(tid)
            if self._is_yellow_phase(logic0.phases[cur0].state):
                self._execute_yellow_clearance(
                    tid, logic0, n0, cur0, accumulate_reward=False
                )

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

    def _overshoot_scale(self, tls_id: str) -> float:
        """Reward multiplier that decays exponentially past max_green.
        At or under max_green: 1.0. Past it: exp(-coeff * (overshoot/max)^2)."""
        steps = self._steps_in_phase.get(tls_id, 0)
        overshoot = max(0, steps - self._max_green_steps)
        if overshoot == 0:
            return 1.0
        ratio = overshoot / self._max_green_steps
        return math.exp(-self._overshoot_coeff * ratio * ratio)

    def step(
        self, obs: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], bool, float | None,
    ]:
        """Semi-Markov step: finalize previous (variable-duration) transition, then act."""
        tls_ids = self.environment.get_tls_ids()

        # How many primitive steps we must fast-forward until min-green is satisfied
        # for all TLS. Decision epochs occur when this becomes 0.
        max_remaining = max(
            max(
                0,
                self._min_green_steps - self._steps_in_phase.get(
                    tid, self._min_green_steps
                ),
            )
            for tid in tls_ids
        )

        # 1) Fast-forward through lockout (variable duration transition accumulation).
        if max_remaining > 0:
            stepped = 0
            for _ in range(max_remaining):
                if self.environment.is_done():
                    break
                self.environment.step(1)

                # Accumulate reward over the entire interval for the pending decision.
                accumulate = self._pending_state is not None
                self._notify_simulation_step(accumulate_reward=accumulate)
                if accumulate:
                    self._pending_duration += 1

                stepped += 1

            # All phases are held during the lockout; time-in-phase advances for each TLS.
            for tid in tls_ids:
                self._steps_in_phase[tid] = (
                    self._steps_in_phase.get(tid, self._min_green_steps) + stepped
                )

            if self.environment.is_done():
                # Episode terminated before the next decision epoch.
                if self._pending_state is not None and self._pending_action is not None:
                    terminal_obs = self._get_observations()
                    done = True
                    pending_duration = self._pending_duration

                    rewards = self._tls_rewards(tls_ids)

                    if self._train_mode:
                        for tid in tls_ids:
                            self.replay_buffer.push(
                                state=self._pending_state[tid],
                                action=self._pending_action[tid],
                                reward=rewards[tid],
                                next_state=terminal_obs[tid],
                                done=float(done),
                                duration=pending_duration,
                            )

                    total_reward = sum(rewards.values())
                    self._episode_reward += total_reward
                    self._episode_steps += 1

                    loss = None
                    if self._train_mode:
                        loss = self.policy.update(self.replay_buffer)
                        if loss is not None:
                            self._episode_losses.append(loss)
                            if self.scheduler is not None:
                                self.scheduler.step()

                    # Clear pending and close.
                    self._pending_state = None
                    self._pending_action = None
                    self._pending_duration = 0
                    return terminal_obs, rewards, done, loss

                # No pending transition to finalize.
                return self._get_observations(), {}, True, None

            obs_now = self._get_observations()
        else:
            # When we're already at a decision epoch, the passed `obs` is correct.
            obs_now = obs

        loss: float | None = None
        rewards: dict[str, float] = {}

        # 2) Finalize the pending transition at this decision epoch boundary.
        if self._pending_state is not None and self._pending_action is not None:
            rewards = self._tls_rewards(tls_ids)

            done = float(self.environment.is_done())
            pending_duration = self._pending_duration

            if self._train_mode:
                for tid in tls_ids:
                    self.replay_buffer.push(
                        state=self._pending_state[tid],
                        action=self._pending_action[tid],
                        reward=rewards[tid],
                        next_state=obs_now[tid],
                        done=done,
                        duration=pending_duration,
                    )

            total_reward = sum(rewards.values())
            self._episode_reward += total_reward
            self._episode_steps += 1

            loss = None
            if self._train_mode:
                loss = self.policy.update(self.replay_buffer)
                if loss is not None:
                    self._episode_losses.append(loss)
                    if self.scheduler is not None:
                        self.scheduler.step()

            self._pending_state = None
            self._pending_action = None
            self._pending_duration = 0

        # 3) Choose actions at this decision epoch and execute one primitive step.
        actions: dict[str, int] = {}
        for tid in tls_ids:
            actions[tid] = self.policy.select_action(obs_now[tid], tid)

        yellow_extra, phase_advanced_tls = self._apply_actions(actions)

        # Start of the next pending interval: the current decision epoch itself.
        self._pending_state = obs_now
        self._pending_action = actions
        self._pending_duration = yellow_extra

        self.environment.step(1)
        self._notify_simulation_step(accumulate_reward=True)
        self._pending_duration += 1

        # Update time-since-switch counters after this step.
        for tid in tls_ids:
            if tid in phase_advanced_tls:
                self._steps_in_phase[tid] = 0
            else:
                self._steps_in_phase[tid] = self._steps_in_phase.get(tid, 0) + 1

        done = self.environment.is_done()
        next_obs = self._get_observations()

        # 4) If the episode ends right after the chosen action, finalize immediately.
        if done and self._pending_state is not None and self._pending_action is not None:
            terminal_rewards = self._tls_rewards(tls_ids)

            pending_duration = self._pending_duration
            if self._train_mode:
                for tid in tls_ids:
                    self.replay_buffer.push(
                        state=self._pending_state[tid],
                        action=self._pending_action[tid],
                        reward=terminal_rewards[tid],
                        next_state=next_obs[tid],
                        done=float(done),
                        duration=pending_duration,
                    )

            total_reward = sum(terminal_rewards.values())
            self._episode_reward += total_reward
            self._episode_steps += 1

            loss = None
            if self._train_mode:
                loss = self.policy.update(self.replay_buffer)
                if loss is not None:
                    self._episode_losses.append(loss)
                    if self.scheduler is not None:
                        self.scheduler.step()

            self._pending_state = None
            self._pending_action = None
            self._pending_duration = 0

            return next_obs, terminal_rewards, True, loss

        return next_obs, rewards, bool(done), loss

    def end_episode(self) -> dict[str, Any]:
        """Close SUMO and return episode metrics."""
        self.environment.close()
        mean_loss = float(np.mean(self._episode_losses)) if self._episode_losses else None
        lr = self.scheduler.get_lr() if self.scheduler else None
        metrics: dict[str, Any] = {
            "total_reward": self._episode_reward,
            "steps":        self._episode_steps,
            "mean_loss":    mean_loss,
            "epsilon":      getattr(self.policy, "epsilon", None),
            "learning_rate": lr,
        }
        return metrics

    def take_phase_sequence_export(self) -> dict[str, list[dict[str, Any]]]:
        """Copy eval phase segments and clear the buffer. Call after end_episode in eval."""
        if self._train_mode:
            return {}
        out: dict[str, list[dict[str, Any]]] = {
            k: list(v) for k, v in self._phase_log.items()
        }
        self._phase_log.clear()
        if os.environ.get("RL_DEBUG"):
            expected = self._yellow_steps * self._step_length
            for _tls_id, entries in out.items():
                for entry in entries:
                    if self._is_yellow_phase(entry["phase_state"]):
                        ds = float(entry["duration_s"])
                        assert ds <= expected + 1e-6, (
                            "Yellow phase logged with unexpected duration: "
                            f"{entry['duration_s']}"
                        )
                        assert math.isclose(ds, expected, rel_tol=0, abs_tol=1e-5) or (
                            ds < expected - 1e-6
                        ), (
                            "Yellow phase duration should be full clearance or "
                            f"truncated at episode end: {entry['duration_s']}"
                        )
        return out

    def set_eval_mode(self) -> None:
        self._train_mode = False
        self.policy.set_eval_mode()

    def set_train_mode(self) -> None:
        self._train_mode = True
        self.policy.set_train_mode()

    def save(self, path: str) -> None:
        """Save policy weights + scheduler state together."""
        self.policy.save(path)
        if self.scheduler is not None:
            sched_path = path.replace(".pt", "_scheduler.pt")
            import torch
            torch.save(self.scheduler.state_dict(), sched_path)

    def load(self, path: str) -> None:
        """Load policy weights + scheduler state if available."""
        self.policy.load(path)
        if self.scheduler is not None:
            sched_path = path.replace(".pt", "_scheduler.pt")
            if os.path.exists(sched_path):
                import torch
                state = torch.load(sched_path, map_location="cpu")
                self.scheduler.load_state_dict(state)

    def _notify_simulation_step(self, accumulate_reward: bool = True) -> None:
        """Let rewards that need per-SUMO-step accounting (e.g. throughput) update state."""
        traci = self.environment.traci
        for tls_id in self.environment.get_tls_ids():
            self.reward.on_simulation_step(traci, tls_id, accumulate=accumulate_reward)

    def _get_observations(self) -> dict[str, np.ndarray]:
        traci = self.environment.traci
        if os.environ.get("RL_DEBUG"):
            for tls_id in self.environment.get_tls_ids():
                logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                ph = logic.phases[traci.trafficlight.getPhase(tls_id)]
                assert not self._is_yellow_phase(ph.state), (
                    f"Agent received yellow phase as observation at {tls_id}: "
                    f"{ph.state}"
                )
        return {
            tls_id: self.observation.build(traci, tls_id)
            for tls_id in self.environment.get_tls_ids()
        }

    @staticmethod
    def _is_yellow_phase(phase_state: str) -> bool:
        """True if phase is yellow-only (no green links in SUMO state string)."""
        return (
            "y" in phase_state
            and "G" not in phase_state
            and "g" not in phase_state
        )

    def _sumo_steps_during_yellow(self, accumulate_reward: bool = True) -> int:
        """Advance simulation for fixed yellow duration; return steps executed."""
        steps = 0
        for _ in range(self._yellow_steps):
            if self.environment.is_done():
                break
            self.environment.step(1)
            self._notify_simulation_step(accumulate_reward=accumulate_reward)
            steps += 1
        return steps

    def _execute_yellow_clearance(
        self,
        tls_id: str,
        logic: Any,
        n_phases: int,
        yellow_idx: int,
        *,
        accumulate_reward: bool = True,
    ) -> int:
        """Leave a yellow phase with fixed-duration substeps; land on following green."""
        steps = self._sumo_steps_during_yellow(accumulate_reward=accumulate_reward)
        green_phase = (yellow_idx + 1) % n_phases
        self.environment.traci.trafficlight.setPhase(tls_id, green_phase)
        if not self._train_mode:
            ph = logic.phases[yellow_idx]
            duration_s = float(steps * self._step_length)
            sl = self._step_length
            self._phase_log.setdefault(tls_id, []).append(
                {
                    "phase": int(yellow_idx),
                    "phase_name": getattr(ph, "name", "") or "",
                    "phase_state": getattr(ph, "state", "") or "",
                    "duration_s": duration_s,
                    "duration_steps": int(duration_s / sl) if sl > 0 else 0,
                    "sim_time": float(self.environment.get_sim_time()),
                }
            )
        return steps

    def _execute_green_switch(
        self,
        tls_id: str,
        logic: Any,
        n_phases: int,
        current_phase: int,
    ) -> int:
        """Log green exit, optionally cross yellow with fixed time, land on next green."""
        traci = self.environment.traci
        if not self._train_mode:
            ph = logic.phases[current_phase]
            duration_s = float(
                self._steps_in_phase.get(tls_id, 0) * self._step_length
            )
            sl = self._step_length
            duration_steps = int(duration_s / sl) if sl > 0 else 0
            self._phase_log.setdefault(tls_id, []).append(
                {
                    "phase": int(current_phase),
                    "phase_name": getattr(ph, "name", "") or "",
                    "phase_state": getattr(ph, "state", "") or "",
                    "duration_s": duration_s,
                    "duration_steps": duration_steps,
                    "sim_time": float(self.environment.get_sim_time()),
                }
            )
        next_phase = (current_phase + 1) % n_phases
        next_state = logic.phases[next_phase].state
        extra = 0
        if self._is_yellow_phase(next_state):
            traci.trafficlight.setPhase(tls_id, next_phase)
            extra = self._sumo_steps_during_yellow()
            if not self._train_mode:
                yph = logic.phases[next_phase]
                duration_s_y = float(extra * self._step_length)
                sl = self._step_length
                self._phase_log.setdefault(tls_id, []).append(
                    {
                        "phase": int(next_phase),
                        "phase_name": getattr(yph, "name", "") or "",
                        "phase_state": getattr(yph, "state", "") or "",
                        "duration_s": duration_s_y,
                        "duration_steps": int(duration_s_y / sl) if sl > 0 else 0,
                        "sim_time": float(self.environment.get_sim_time()),
                    }
                )
            green_phase = (next_phase + 1) % n_phases
            traci.trafficlight.setPhase(tls_id, green_phase)
        else:
            traci.trafficlight.setPhase(tls_id, next_phase)
        return extra

    def _apply_actions(self, actions: dict[str, int]) -> tuple[int, set[str]]:
        """Action 0 = hold, 1 = advance to next green (auto-crossing yellow).

        Returns (extra primitive steps for SMDP pending_duration, tls_ids that changed phase).
        """
        traci = self.environment.traci
        extra_pending = 0
        advanced: set[str] = set()
        for tls_id, action in actions.items():
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(logic.phases)
            cur = traci.trafficlight.getPhase(tls_id)
            cur_state = logic.phases[cur].state

            if self._is_yellow_phase(cur_state):
                extra_pending += self._execute_yellow_clearance(
                    tls_id, logic, n_phases, cur
                )
                advanced.add(tls_id)
                continue

            if action != 1:
                continue

            extra_pending += self._execute_green_switch(
                tls_id, logic, n_phases, cur
            )
            advanced.add(tls_id)
        return extra_pending, advanced
