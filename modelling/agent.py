"""Agent — composes all RL components with adaptive decision timing.
Fast-forwards through min_green lockout, decides every step after.
Past max_green_s, holding (action 0) incurs an exponential penalty in overshoot;
rewards are also damped/amplified via overshoot_coeff. Switching still requires action 1."""

import math
import os
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Callable, Dict, List

import numpy as np

from modelling.components.environment.base   import BaseEnvironment
from modelling.components.observation.base   import BaseObservation
from modelling.components.reward.base        import BaseReward
from modelling.components.policy.base        import BasePolicy
from modelling.components.replay_buffer.base import BaseReplayBuffer
from modelling.components.scheduler.base     import BaseScheduler


class Agent:
    # Must match ``programID`` on ``tlLogic`` in the SUMO net (BuildNetwork / .tll.xml).
    _STATIC_TLS_PROGRAM_ID = "0"

    @staticmethod
    def _switch_debug_enabled() -> bool:
        v = (os.environ.get("RL_SWITCH_DEBUG") or "").strip().lower()
        return v not in ("", "0", "false", "no", "off")

    def _switch_dbg(self, tag: str, msg: str) -> None:
        if self._switch_debug_enabled():
            print(f"[SWITCH_DEBUG {tag}] {msg}", flush=True)

    def _switch_dbg_traci(
        self, tag: str, desc: str, tls_id: str, fn: Callable[[], None]
    ) -> None:
        """Run a TraCI write; log args and never swallow exceptions."""
        if self._switch_debug_enabled():
            print(
                f"[SWITCH_DEBUG {tag}] {desc} tls_id={tls_id!r} (calling)",
                flush=True,
            )
        try:
            fn()
        except Exception as e:
            if self._switch_debug_enabled():
                print(
                    f"[SWITCH_DEBUG {tag}] {desc} tls_id={tls_id!r} "
                    f"FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )
            raise

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
        min_red_s:       float = 1.0,
        decision_gap:    int   = 1,
        episode_kpis:    Sequence[Any] | None = None,
    ):
        self.environment   = environment
        self.observation   = observation
        self.reward        = reward
        self.policy        = policy
        self.replay_buffer = replay_buffer
        self.scheduler     = scheduler
        self.eval_reward   = eval_reward
        self._episode_kpis = tuple(episode_kpis or ())

        self._step_length       = step_length
        self._yellow_duration_s = yellow_duration_s
        self._yellow_steps      = max(1, round(yellow_duration_s / step_length))
        self._min_green_steps   = max(1, math.ceil(min_green_s / step_length))
        self._max_green_steps   = max(1, math.ceil(max_green_s / step_length))
        self._min_red_steps     = max(1, round(min_red_s / step_length))
        self._decision_gap      = max(1, int(decision_gap))
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
        # Target static phase after a TraCI yellow interval (online program uses phase index 0 only).
        self._tls_next_phase_after_yellow: dict[str, int] = {}
        # Training-only: completed green segments (tls_id, SUMO phase index, duration_s sim time).
        self._train_green_segments: list[tuple[str, int, float]] = []
        self._green_segment_start_time: dict[str, float] = {}
        self._switch_debug_step_count: int = 0

    @staticmethod
    def _static_program_logic(traci: Any, tls_id: str) -> Any:
        """TLS program ``0`` (multi-phase net definition), not the TraCI ``online`` program."""
        for lg in traci.trafficlight.getAllProgramLogics(tls_id):
            pid = getattr(lg, "programID", None) or getattr(lg, "subID", None)
            if str(pid) == Agent._STATIC_TLS_PROGRAM_ID:
                return lg
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        return logics[0]

    def _primitive_simulation_step(self, accumulate_reward: bool) -> bool:
        """Advance SUMO by one primitive step."""
        if self.environment.is_done():
            return False
        self.environment.step(1)
        self._notify_simulation_step(accumulate_reward=accumulate_reward)
        return True

    def _run_all_red_clearance(
        self, tls_id: str, state_len: int, *, accumulate_reward: bool
    ) -> int:
        """TraCI-only all-red (Section 3.E); returns primitive steps executed."""
        traci = self.environment.traci
        ryg = "r" * int(state_len)
        self._switch_dbg_traci(
            "T4",
            f"_run_all_red_clearance setRedYellowGreenState(all-red) len={len(ryg)}",
            tls_id,
            lambda: traci.trafficlight.setRedYellowGreenState(tls_id, ryg),
        )
        steps = 0
        for _ in range(self._min_red_steps):
            if not self._primitive_simulation_step(accumulate_reward=accumulate_reward):
                break
            steps += 1
        return steps

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
        for tid in tls_ids:
            held = pa is None or int(pa.get(tid, 0)) != 1
            out[tid] = self._apply_overshoot_to_reward(tid, out[tid], held=held)
        return out

    def start_episode(self, route_file: str) -> dict[str, np.ndarray]:
        """Reset everything and launch SUMO for one episode. Returns initial observations."""
        self.environment.start(route_file)
        self.observation.reset()
        self.reward.reset()
        if self.eval_reward is not None:
            self.eval_reward.reset()
        for kpi in self._episode_kpis:
            kpi.reset()
        self.policy.reset_phase_tracking()

        self._episode_reward = 0.0
        self._episode_steps  = 0
        self._episode_losses = []
        self._pending_state   = None
        self._pending_action  = None
        self._pending_duration = 0
        self._phase_log = {}
        self._tls_next_phase_after_yellow.clear()
        self._switch_debug_step_count = 0

        warmup_steps = max(5, round(25 / self._step_length))
        for _ in range(warmup_steps):
            self._primitive_simulation_step(accumulate_reward=False)

        # Land on green before the first decision. _pending_state is still None here
        # (only cleared above; first step() sets it), so yellow substeps must not touch
        # _pending_duration — accumulate_reward=False is correct.
        traci_w = self.environment.traci
        for tid in list(traci_w.trafficlight.getIDList()):
            # setRedYellowGreenState switches SUMO to the one-phase ``online`` program;
            # use setProgram + setPhase so phase indices 0..n-1 stay valid for switching.
            traci_w.trafficlight.setProgram(tid, self._STATIC_TLS_PROGRAM_ID)
            traci_w.trafficlight.setPhase(tid, 0)

        tls_ids = self.environment.get_tls_ids()
        if not tls_ids:
            raise RuntimeError(
                "No traffic lights found in simulation. "
                "Check that BuildNetwork produced a valid .net.xml with "
                "a traffic light at J_centre."
            )

        for tls_id in tls_ids:
            self._steps_in_phase[tls_id] = self._min_green_steps
            self._mark_green_segment_start(tls_id)

        if self._switch_debug_enabled():
            traci_d = self.environment.traci
            all_ids = list(traci_d.trafficlight.getIDList())
            for tid in tls_ids:
                logic = self._static_program_logic(traci_d, tid)
                nph = len(logic.phases)
                self._switch_dbg(
                    "EPISODE",
                    f"tls_id={tid!r} in getIDList={tid in all_ids} "
                    f"program0 n_phases={nph} sumo_phase={traci_d.trafficlight.getPhase(tid)}",
                )

        return self._get_observations()

    # Cap overshoot ratio so exp(coeff * ratio) stays finite in float64.
    _OVERSHOOT_RATIO_CAP = 8.0

    def _overshoot_ratio(self, tls_id: str) -> float:
        """Normalized time past max_green: (steps - max) / max_green_steps, or 0 if not past."""
        steps = self._steps_in_phase.get(tls_id, 0)
        overshoot = max(0, steps - self._max_green_steps)
        if overshoot == 0 or self._max_green_steps <= 0:
            return 0.0
        return min(overshoot / self._max_green_steps, self._OVERSHOOT_RATIO_CAP)

    def _overshoot_scale(self, tls_id: str) -> float:
        """Gaussian dampening factor exp(-coeff * ratio^2); 1.0 if not past max_green."""
        ratio = self._overshoot_ratio(tls_id)
        if ratio == 0.0:
            return 1.0
        return math.exp(-self._overshoot_coeff * ratio * ratio)

    def _apply_overshoot_to_reward(
        self, tls_id: str, reward: float, *, held: bool
    ) -> float:
        """Past max_green_s: shape reward; if the agent **held** (no switch), add exponential cost.

        ``exp(overshoot_coeff * ratio) - 1`` grows exponentially in overshoot (ratio is
        overshoot / max_green_steps). Applied in **train and eval** so GUI matches training.
        """
        if self._steps_in_phase.get(tls_id, 0) <= self._max_green_steps:
            return reward
        scale = self._overshoot_scale(tls_id)
        out = reward * scale if reward >= 0.0 else reward / max(scale, 1e-6)
        if held:
            ratio = self._overshoot_ratio(tls_id)
            if ratio > 0.0:
                out -= math.exp(self._overshoot_coeff * ratio) - 1.0
        return out

    def step(
        self, obs: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray], dict[str, float], bool, float | None,
    ]:
        """Semi-Markov step: finalize previous (variable-duration) transition, then act.

        Min-green is enforced by fast-forwarding primitive SUMO steps before the policy
        runs; rewards over that interval are attributed to the *previous* decision's*
        duration (standard SMDP), not to a synthetic KEEP choice (Section 4.1 note).
        """
        tls_ids = self.environment.get_tls_ids()
        self._switch_debug_step_count += 1
        traci_dbg = self.environment.traci
        sim_t = float(self.environment.get_sim_time())
        if self._switch_debug_enabled():
            all_tls = list(traci_dbg.trafficlight.getIDList())
            parts = []
            for tid in tls_ids:
                sip = self._steps_in_phase.get(tid, 0)
                ph = traci_dbg.trafficlight.getPhase(tid)
                parts.append(
                    f"{tid}: sumo_phase={ph} steps_in_phase={sip} "
                    f"(~{sip * self._step_length:.3f}s)"
                )
            pend = self._pending_action
            self._switch_dbg(
                "T1",
                f"step_count={self._switch_debug_step_count} sim_t={sim_t:.2f}s "
                f"pending_prev_action={pend!r} tls_list={tls_ids!r} "
                f"getIDList={all_tls!r} | " + " | ".join(parts),
            )

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

        if self._switch_debug_enabled():
            self._switch_dbg(
                "T2",
                f"min_green_gate: max_remaining={max_remaining} primitive_steps "
                f"(defer_decision_epoch={max_remaining > 0}) | "
                f"threshold _min_green_steps={self._min_green_steps} "
                f"vs _steps_in_phase per tls (no action override after policy; "
                f"gate only fast-forwards)",
            )
            self._switch_dbg(
                "T3",
                "lockout_remaining: none (separate post-switch lockout removed; "
                "units N/A)",
            )

        # 1) Fast-forward through lockout (variable duration transition accumulation).
        if max_remaining > 0:
            stepped = 0
            for _ in range(max_remaining):
                accumulate = self._pending_state is not None
                if not self._primitive_simulation_step(accumulate_reward=accumulate):
                    break
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
            if self._switch_debug_enabled():
                self._switch_dbg(
                    "T2",
                    f"after min_green fast-forward: reached decision epoch "
                    f"(stepped_primitive={stepped})",
                )
        else:
            # When we're already at a decision epoch, the passed `obs` is correct.
            obs_now = obs
            if self._switch_debug_enabled():
                self._switch_dbg(
                    "T2",
                    "min_green_gate: already at decision epoch (max_remaining=0)",
                )

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

        # 3) Choose actions. Min-green / max-green timing is enforced via
        # _steps_in_phase + max_remaining fast-forward above, not a separate lockout.
        actions: dict[str, int] = {
            tid: int(self.policy.select_action(obs_now[tid], tid)) for tid in tls_ids
        }
        if self._switch_debug_enabled():
            self._switch_dbg(
                "T1",
                f"chosen_actions={actions!r} train_mode={self._train_mode} "
                f"decision_gap={self._decision_gap}",
            )

        yellow_extra, phase_advanced_tls = self._apply_actions(actions)

        # Start of the next pending interval: the current decision epoch itself.
        self._pending_state = obs_now
        self._pending_action = actions
        self._pending_duration = yellow_extra

        for _ in range(self._decision_gap):
            if not self._primitive_simulation_step(accumulate_reward=True):
                break
            self._pending_duration += 1

        # Update time-in-phase in primitive steps (decision_gap per decision epoch).
        for tid in tls_ids:
            if tid in phase_advanced_tls:
                self._steps_in_phase[tid] = 0
            else:
                self._steps_in_phase[tid] = (
                    self._steps_in_phase.get(tid, 0) + self._decision_gap
                )

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
        self._flush_train_green_segment_at_episode_end()
        self.environment.close()
        mean_loss = float(np.mean(self._episode_losses)) if self._episode_losses else None
        if self.scheduler is not None:
            lr = self.scheduler.get_lr()
        else:
            lr = None
            opt = getattr(self.policy, "optimizer", None)
            if opt is not None and opt.param_groups:
                lr = float(opt.param_groups[0]["lr"])
        metrics: dict[str, Any] = {
            "total_reward": self._episode_reward,
            "steps":        self._episode_steps,
            "mean_loss":    mean_loss,
            "epsilon":      getattr(self.policy, "epsilon", None),
            "learning_rate": lr,
        }
        elapsed = self._kpi_elapsed_s()
        for kpi in self._episode_kpis:
            kpi.contribute_episode_metrics(metrics, elapsed)
        return metrics

    def reset_train_green_phase_stats(self) -> None:
        """Clear accumulated training green-phase samples (call before the training loop)."""
        self._train_green_segments.clear()
        self._green_segment_start_time.clear()

    def _mark_green_segment_start(self, tls_id: str) -> None:
        self._green_segment_start_time[tls_id] = float(self.environment.get_sim_time())

    def _record_train_green_segment_end(self, tls_id: str, phase_index: int) -> None:
        if not self._train_mode:
            return
        start = self._green_segment_start_time.pop(tls_id, None)
        if start is None:
            return
        dur = float(self.environment.get_sim_time() - start)
        if dur >= 0.0:
            self._train_green_segments.append((tls_id, int(phase_index), dur))

    def _flush_train_green_segment_at_episode_end(self) -> None:
        """Append an open green segment if the episode ends on green (train mode only)."""
        if not self._train_mode:
            return
        traci = self.environment.traci
        st = float(self.environment.get_sim_time())
        for tls_id in self.environment.get_tls_ids():
            try:
                live = traci.trafficlight.getRedYellowGreenState(tls_id)
            except Exception:
                continue
            if self._is_yellow_phase(live):
                continue
            start = self._green_segment_start_time.pop(tls_id, None)
            if start is None:
                continue
            try:
                cur = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                continue
            dur = st - start
            if dur >= 0.0:
                self._train_green_segments.append((tls_id, cur, float(dur)))

    def format_train_green_phase_duration_report(self) -> list[str]:
        """Mean / stdev of green-hold times (sim seconds) over training; per SUMO phase index."""
        segs = self._train_green_segments
        if not segs:
            return [
                "Train green-phase durations: no segments recorded "
                "(need at least one training episode with a completed or open green segment)."
            ]
        all_d = [s[2] for s in segs]
        pooled_std = float(statistics.stdev(all_d)) if len(all_d) > 1 else 0.0
        lines = [
            "Train green-phase durations (SUMO sim time, seconds):",
            f"  All green segments:  n={len(all_d)}  mean={statistics.mean(all_d):.2f}s"
            f"  stdev={pooled_std:.2f}s",
            "  By phase index (tlLogic program 0):",
        ]
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        for tid, ph, d in segs:
            buckets[(tid, ph)].append(d)
        for (tid, ph) in sorted(buckets.keys()):
            ds = buckets[(tid, ph)]
            sd = float(statistics.stdev(ds)) if len(ds) > 1 else 0.0
            lines.append(
                f"    {tid}  phase {ph}:  n={len(ds)}  mean={statistics.mean(ds):.2f}s"
                f"  stdev={sd:.2f}s"
            )
        return lines

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

    def _kpi_elapsed_s(self) -> float:
        b = getattr(self.environment, "_begin", None)
        e = getattr(self.environment, "_end", None)
        if b is not None and e is not None:
            return float(e - b)
        return 1.0

    def _notify_simulation_step(self, accumulate_reward: bool = True) -> None:
        """Let rewards that need per-SUMO-step accounting (e.g. throughput) update state."""
        traci = self.environment.traci
        for tls_id in self.environment.get_tls_ids():
            self.reward.on_simulation_step(traci, tls_id, accumulate=accumulate_reward)
            for kpi in self._episode_kpis:
                kpi.on_simulation_step(traci, tls_id, accumulate=accumulate_reward)

    def _get_observations(self) -> dict[str, np.ndarray]:
        traci = self.environment.traci
        if os.environ.get("RL_DEBUG"):
            for tls_id in self.environment.get_tls_ids():
                logic = self._static_program_logic(traci, tls_id)
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

    @staticmethod
    def _yellow_clearance_state_between(prev_state: str, next_state: str) -> str:
        """Per-link yellow where green drops to red; new greens stay red during clearance."""
        out: list[str] = []
        for p, n in zip(prev_state, next_state):
            if p in "Gg" and n == "r":
                out.append("y")
            else:
                out.append("r")
        return "".join(out)

    def _sumo_steps_during_yellow(self, accumulate_reward: bool = True) -> int:
        """Advance simulation for fixed yellow duration; return steps executed."""
        steps = 0
        for _ in range(self._yellow_steps):
            if not self._primitive_simulation_step(accumulate_reward=accumulate_reward):
                break
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
        """Finish a TraCI yellow interval and land on the next green program phase."""
        traci = self.environment.traci
        try:
            y_live = traci.trafficlight.getRedYellowGreenState(tls_id)
        except Exception:
            y_live = ""
        steps = self._sumo_steps_during_yellow(accumulate_reward=accumulate_reward)
        next_phase = self._tls_next_phase_after_yellow.get(tls_id)
        if next_phase is None:
            next_phase = (yellow_idx + 1) % n_phases
        if steps < self._yellow_steps:
            return steps
        slen = len(y_live) if y_live else len(logic.phases[0].state)
        steps += self._run_all_red_clearance(
            tls_id, slen, accumulate_reward=accumulate_reward
        )
        self._switch_dbg(
            "T4",
            f"_execute_yellow_clearance setProgram({self._STATIC_TLS_PROGRAM_ID!r}) "
            f"setPhase({next_phase}) yellow_idx={yellow_idx}",
        )
        self._switch_dbg_traci(
            "T4",
            f"setProgram({self._STATIC_TLS_PROGRAM_ID!r})",
            tls_id,
            lambda: traci.trafficlight.setProgram(
                tls_id, self._STATIC_TLS_PROGRAM_ID
            ),
        )
        self._switch_dbg_traci(
            "T4",
            f"setPhase({next_phase})",
            tls_id,
            lambda: traci.trafficlight.setPhase(tls_id, next_phase),
        )
        self._tls_next_phase_after_yellow.pop(tls_id, None)
        if self._switch_debug_enabled():
            ph = traci.trafficlight.getPhase(tls_id)
            self._switch_dbg(
                "T5",
                f"after yellow clearance land-on-green getPhase={ph} tls_id={tls_id!r}",
            )
        if self._train_mode:
            self._mark_green_segment_start(tls_id)
        if not self._train_mode:
            duration_s = float(steps * self._step_length)
            sl = self._step_length
            self._phase_log.setdefault(tls_id, []).append(
                {
                    "phase": int(yellow_idx),
                    "phase_name": "yellow_clearance",
                    "phase_state": y_live or "",
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
        if self._train_mode:
            self._record_train_green_segment_end(tls_id, current_phase)
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
        prev_st = logic.phases[current_phase].state
        next_st = logic.phases[next_phase].state
        y_st = self._yellow_clearance_state_between(prev_st, next_st)
        self._tls_next_phase_after_yellow[tls_id] = next_phase
        self._switch_dbg(
            "T4",
            f"_execute_green_switch setRedYellowGreenState len={len(y_st)} "
            f"current_phase={current_phase} next_phase={next_phase}",
        )
        self._switch_dbg_traci(
            "T4",
            "setRedYellowGreenState(yellow)",
            tls_id,
            lambda: traci.trafficlight.setRedYellowGreenState(tls_id, y_st),
        )
        extra = self._sumo_steps_during_yellow()
        if not self._train_mode:
            duration_s_y = float(extra * self._step_length)
            sl = self._step_length
            self._phase_log.setdefault(tls_id, []).append(
                {
                    "phase": int(current_phase),
                    "phase_name": "yellow_clearance",
                    "phase_state": y_st,
                    "duration_s": duration_s_y,
                    "duration_steps": int(duration_s_y / sl) if sl > 0 else 0,
                    "sim_time": float(self.environment.get_sim_time()),
                }
            )
        if extra < self._yellow_steps:
            return extra
        extra += self._run_all_red_clearance(tls_id, len(y_st), accumulate_reward=True)
        self._switch_dbg(
            "T4",
            f"_execute_green_switch post-clearance setProgram({self._STATIC_TLS_PROGRAM_ID!r}) "
            f"setPhase({next_phase})",
        )
        self._switch_dbg_traci(
            "T4",
            f"setProgram({self._STATIC_TLS_PROGRAM_ID!r})",
            tls_id,
            lambda: traci.trafficlight.setProgram(
                tls_id, self._STATIC_TLS_PROGRAM_ID
            ),
        )
        self._switch_dbg_traci(
            "T4",
            f"setPhase({next_phase})",
            tls_id,
            lambda: traci.trafficlight.setPhase(tls_id, next_phase),
        )
        self._tls_next_phase_after_yellow.pop(tls_id, None)
        if self._switch_debug_enabled():
            ph = traci.trafficlight.getPhase(tls_id)
            self._switch_dbg(
                "T5",
                f"after green switch land-on-green getPhase={ph} "
                f"expected={next_phase} tls_id={tls_id!r}",
            )
        if self._train_mode:
            self._mark_green_segment_start(tls_id)
        return extra

    def _apply_actions(self, actions: dict[str, int]) -> tuple[int, set[str]]:
        """Action 0 = hold, 1 = advance to next green (auto-crossing yellow).

        Returns (extra primitive steps for SMDP pending_duration, tls_ids that changed phase).
        """
        traci = self.environment.traci
        extra_pending = 0
        advanced: set[str] = set()
        for tls_id, action in actions.items():
            logic = self._static_program_logic(traci, tls_id)
            n_phases = len(logic.phases)
            cur = traci.trafficlight.getPhase(tls_id)
            if cur >= n_phases:
                traci.trafficlight.setProgram(tls_id, self._STATIC_TLS_PROGRAM_ID)
                cur = min(traci.trafficlight.getPhase(tls_id), n_phases - 1)
            cur_state = logic.phases[cur].state
            try:
                live = traci.trafficlight.getRedYellowGreenState(tls_id)
            except Exception:
                live = cur_state

            if self._is_yellow_phase(live):
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
