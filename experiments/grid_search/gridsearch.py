"""
Grid search runner for RL traffic control.

This is a lightweight "GridSearchCV equivalent" for your setup:
- Sweep reward + policy hyperparameters using JSON search spaces.
- Run each trial end-to-end (train on train days, evaluate on test days).
- Compute KPIs via `KPIS` (crossings, throughput, neg lane-waiting integral); `schedule.json` per trial when possible.
- Write per-trial results to `results.csv` and an aggregated `leaderboard.csv`.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

# Ensure repo root is importable when running from arbitrary working directories.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import main as main_mod
from KPIS import aggregate_test_kpis, default_episode_kpis
from modelling.agent import Agent
from modelling.schedule_export import write_schedule_for_run, write_test_sequence_episode_json
from visualization.visualize_results import render_run_graphs
from modelling.components.environment.sumo_environment import SumoEnvironment
from modelling.components.observation.queue_observation import QueueObservation
from modelling.components.policy.dqn import DQNPolicy
from modelling.components.policy.double_dqn import DoubleDQNPolicy
from modelling.components.replay_buffer.uniform import UniformReplayBuffer
from modelling.components.reward.throughput import ThroughputReward
from modelling.components.reward.vehicle_count import VehicleCountReward
from modelling.components.reward.delta_vehicle_count import DeltaVehicleCountReward
from modelling.components.reward.composite_reward import CompositeReward
from modelling.components.reward.throughput_queue import ThroughputQueueReward
from modelling.components.reward.throughput_composite import ThroughputCompositeReward
from modelling.components.reward.throughput_wait_time import ThroughputWaitTimeReward
from modelling.components.reward.waiting_time import WaitingTimeReward
from modelling.components.reward.delta_waiting_time import DeltaWaitingTimeReward


REWARD_CLASS_MAP: dict[str, Any] = {
    "ThroughputReward": ThroughputReward,
    "ThroughputQueueReward": ThroughputQueueReward,
    "ThroughputCompositeReward": ThroughputCompositeReward,
    "ThroughputWaitTimeReward": ThroughputWaitTimeReward,
    "ThroughputWaitTime": ThroughputWaitTimeReward,
    "CompositeReward": CompositeReward,
    "DeltaVehicleCountReward": DeltaVehicleCountReward,
    "VehicleCountReward": VehicleCountReward,
    "WaitingTimeReward": WaitingTimeReward,
    "WaitTimeReward": WaitingTimeReward,
    "DeltaWaitTimeReward": DeltaWaitingTimeReward,
    "DeltaWaitingTimeReward": DeltaWaitingTimeReward,
}

POLICY_CLASS_MAP: dict[str, Any] = {
    "DQNPolicy": DQNPolicy,
    "DoubleDQNPolicy": DoubleDQNPolicy,
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _try_render_trial_graphs(trial_dir: str) -> None:
    """Write ``graphs/train_curves.png`` and ``graphs/test_rewards.png`` for this trial."""
    try:
        render_run_graphs(trial_dir)
    except Exception as e:
        print(f"[gridsearch] warning: could not write graphs under {trial_dir!r}: {e}")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _product_dict(d: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product for param_name -> list(values)."""
    if not d:
        return [{}]
    keys = list(d.keys())
    values = [d[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _flatten_params(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        out[key] = v
    return out


@dataclass
class Trial:
    trial_id: str
    reward_class_name: str
    reward_kwargs: dict[str, Any]
    policy_class_name: str
    policy_kwargs: dict[str, Any]
    seed: int


def _resolve_policy_kwargs(raw: dict[str, Any], total_updates: int) -> dict[str, Any]:
    resolved = dict(raw)
    required = [
        "lr",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "target_update",
        "batch_size",
        "hidden",
        "n_actions",
    ]
    missing = [k for k in required if k not in resolved]
    if missing:
        raise ValueError(f"Missing policy params: {missing}")

    eps_decay = resolved.get("epsilon_decay", "auto")
    if eps_decay == "auto":
        eps_start = float(resolved["epsilon_start"])
        eps_end = float(resolved["epsilon_end"])
        resolved["epsilon_decay"] = (eps_end / eps_start) ** (1.0 / (0.85 * total_updates))
    return resolved


def _build_agent(
    net_file: str,
    use_gui: bool,
    reward_class_name: str,
    reward_kwargs: dict[str, Any],
    policy_class_name: str,
    policy_kwargs: dict[str, Any],
    seed: int,
) -> tuple[Agent, RewardWithCrossingsKPI]:
    reward_cls = REWARD_CLASS_MAP[reward_class_name]
    policy_cls = POLICY_CLASS_MAP[policy_class_name]

    obs_builder = QueueObservation(
        max_lanes=main_mod.MAX_LANES,
        max_phase=main_mod.MAX_PHASE,
        max_phase_time=main_mod.MAX_PHASE_TIME,
        max_vehicles=main_mod.MAX_VEHICLES,
        max_green_s=main_mod.MAX_GREEN_S,
    )

    environment = SumoEnvironment(
        net_file=net_file,
        step_length=main_mod.STEP_LENGTH,
        gui=use_gui,
        sumo_home=main_mod.SUMO_HOME,
        begin=main_mod.SIM_BEGIN,
        end=main_mod.SIM_END,
    )

    primary_reward = reward_cls(**reward_kwargs)

    policy = policy_cls(
        obs_dim=obs_builder.size(),
        n_actions=policy_kwargs["n_actions"],
        lr=policy_kwargs["lr"],
        gamma=policy_kwargs["gamma"],
        epsilon_start=policy_kwargs["epsilon_start"],
        epsilon_end=policy_kwargs["epsilon_end"],
        epsilon_decay=policy_kwargs["epsilon_decay"],
        target_update=policy_kwargs["target_update"],
        batch_size=policy_kwargs["batch_size"],
        hidden=policy_kwargs["hidden"],
    )

    replay_buffer = UniformReplayBuffer(
        capacity=main_mod.BUFFER_CAPACITY,
        seed=seed,
    )

    agent = Agent(
        environment=environment,
        observation=obs_builder,
        reward=primary_reward,
        policy=policy,
        replay_buffer=replay_buffer,
        scheduler=None,
        step_length=main_mod.STEP_LENGTH,
        min_green_s=main_mod.MIN_GREEN_S,
        max_green_s=main_mod.MAX_GREEN_S,
        overshoot_coeff=main_mod.OVERSHOOT_COEFF,
        yellow_duration_s=main_mod.YELLOW_DURATION_S,
        episode_kpis=default_episode_kpis(),
    )
    return agent


def _write_trial_snapshot(trial_dir: str, trial: Trial) -> None:
    _ensure_dir(trial_dir)

    # Snapshot reward/policy configurations for reproducibility.
    reward_snapshot = {
        "default": {},
        trial.reward_class_name: trial.reward_kwargs,
    }
    policy_snapshot = {
        "default": {},
        trial.policy_class_name: trial.policy_kwargs,
    }

    with open(os.path.join(trial_dir, "reward_configuration.json"), "w", encoding="utf-8") as f:
        json.dump(reward_snapshot, f, indent=2)
    with open(os.path.join(trial_dir, "policy_configuration.json"), "w", encoding="utf-8") as f:
        json.dump(policy_snapshot, f, indent=2)


def _write_config_txt(
    trial_dir: str,
    trial: Trial,
    train_metrics_example: dict[str, Any] | None = None,
    test_metrics_example: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"Trial id:     {trial.trial_id}")
    lines.append(f"Seed:         {trial.seed}")
    lines.append("")
    lines.append("--- Config snapshots ---")
    lines.append(f"Reward config: reward_configuration.json")
    lines.append(f"Policy config: policy_configuration.json")
    lines.append("")
    lines.append("--- Reward ---")
    lines.append(f"Reward class: {trial.reward_class_name}")
    lines.append(f"Args:         {json.dumps(trial.reward_kwargs, sort_keys=True)}")
    lines.append("")
    lines.append("--- Policy ---")
    lines.append(f"Policy class: {trial.policy_class_name}")
    lines.append(f"Args:         {json.dumps(trial.policy_kwargs, sort_keys=True)}")
    lines.append("")
    lines.append("--- Episode KPI (examples) ---")
    if train_metrics_example:
        lines.append(f"Train example crossings_total: {train_metrics_example.get('crossings_total')}")
        lines.append(f"Train example crossings_rate:  {train_metrics_example.get('crossings_rate')}")
    if test_metrics_example:
        lines.append(f"Test example crossings_total:  {test_metrics_example.get('crossings_total')}")
        lines.append(f"Test example crossings_rate:   {test_metrics_example.get('crossings_rate')}")
    with open(os.path.join(trial_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _run_episode_loop(agent: Agent, route_file: str) -> dict[str, Any]:
    obs = agent.start_episode(route_file)
    done = False
    while not done:
        obs, rewards, done, loss = agent.step(obs)
    return agent.end_episode()


def _run_test_episode_save_sequence(
    agent: Agent, route_file: str, trial_dir: str, episode_idx: int
) -> dict[str, Any]:
    metrics = _run_episode_loop(agent, route_file)
    export = agent.take_phase_sequence_export()
    write_test_sequence_episode_json(trial_dir, episode_idx, export)
    return metrics


def _is_diverged(mean_loss: float | None, threshold: float) -> bool:
    if mean_loss is None:
        return False
    if isinstance(mean_loss, float) and (math.isnan(mean_loss) or math.isinf(mean_loss)):
        return True
    return float(mean_loss) > threshold


def load_trials(reward_space_path: str, policy_space_path: str) -> list[Trial]:
    reward_space = _load_json(reward_space_path)
    policy_space = _load_json(policy_space_path)

    reward_entries = reward_space.get("rewards")
    if not isinstance(reward_entries, list):
        raise ValueError("reward_search_space.json must contain `rewards: [...]`")

    policy_entries = policy_space.get("policies")
    if not isinstance(policy_entries, list):
        raise ValueError("policy_search_space.json must contain `policies: [...]`")

    # Build a flat list of trials (reward params x policy params x seeds).
    trials: list[Trial] = []
    trial_counter = 0

    for reward_entry in reward_entries:
        reward_class_name = reward_entry["RewardClass"]
        reward_params_grid = _product_dict(reward_entry.get("params", {}))

        for policy_entry in policy_entries:
            policy_class_name = policy_entry["PolicyClass"]
            policy_params_grid = _product_dict(policy_entry.get("params", {}))
            seeds = policy_entry.get("seeds", [main_mod.SEED])

            for seed in seeds:
                for reward_kwargs, policy_raw_kwargs in itertools.product(
                    reward_params_grid, policy_params_grid
                ):
                    trial_counter += 1
                    trial_id = f"trial_{trial_counter:04d}"

                    policy_kwargs = _resolve_policy_kwargs(
                        policy_raw_kwargs, total_updates=main_mod.TOTAL_UPDATES
                    )
                    trials.append(
                        Trial(
                            trial_id=trial_id,
                            reward_class_name=reward_class_name,
                            reward_kwargs=reward_kwargs,
                            policy_class_name=policy_class_name,
                            policy_kwargs=policy_kwargs,
                            seed=int(seed),
                        )
                    )

    return trials


def _write_results_header(csv_path: str, fieldnames: list[str]) -> None:
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_result_row(csv_path: str, row: dict[str, Any], fieldnames: list[str]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _config_key(trial: Trial) -> str:
    # Exclude seed so leaderboard can aggregate across seeds.
    reward_key = f"{trial.reward_class_name}|" + ",".join(
        f"{k}={trial.reward_kwargs[k]}" for k in sorted(trial.reward_kwargs.keys())
    )
    policy_key = f"{trial.policy_class_name}|" + ",".join(
        f"{k}={trial.policy_kwargs[k]}" for k in sorted(trial.policy_kwargs.keys())
    )
    return f"{reward_key}||{policy_key}||EPOCHS={main_mod.EPOCHS}"


def build_leaderboard(results_csv_path: str, leaderboard_csv_path: str) -> None:
    with open(results_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = r["config_key"]
        grouped.setdefault(key, []).append(r)

    out_rows: list[dict[str, Any]] = []
    for key, trials in grouped.items():
        rates = [float(t["test_crossings_rate_mean"]) for t in trials]
        mean_rate = float(np.mean(rates))
        std_rate = float(np.std(rates))
        tr = [float(t.get("test_throughput_rate_mean", 0)) for t in trials]
        nw = [float(t.get("test_neg_lane_waiting_integral_mean", 0)) for t in trials]

        out_rows.append(
            {
                "config_key": key,
                "n_seeds": len(trials),
                "mean_crossings_rate": mean_rate,
                "std_crossings_rate": std_rate,
                "mean_throughput_rate": float(np.mean(tr)) if tr else 0.0,
                "std_throughput_rate": float(np.std(tr)) if len(tr) > 1 else 0.0,
                "mean_neg_lane_waiting_integral": float(np.mean(nw)) if nw else 0.0,
                "std_neg_lane_waiting_integral": float(np.std(nw)) if len(nw) > 1 else 0.0,
            }
        )

    out_rows.sort(key=lambda x: (-x["mean_crossings_rate"], x["std_crossings_rate"]))

    lb_fields = [
        "config_key",
        "n_seeds",
        "mean_crossings_rate",
        "std_crossings_rate",
        "mean_throughput_rate",
        "std_throughput_rate",
        "mean_neg_lane_waiting_integral",
        "std_neg_lane_waiting_integral",
    ]
    with open(leaderboard_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=lb_fields,
        )
        writer.writeheader()
        writer.writerows(out_rows)


def _apply_training_epochs_for_grid(epochs: int) -> None:
    """Override ``main_mod.EPOCHS`` and recompute ``TOTAL_UPDATES`` (for ``epsilon_decay`` auto)."""
    main_mod.EPOCHS = int(epochs)
    sim_seconds = float(main_mod.SIM_END - main_mod.SIM_BEGIN)
    steps_per_ep = sim_seconds / main_mod.STEP_LENGTH
    divisor = float(getattr(main_mod, "DECISIONS_PER_EP_DIVISOR", 2.0) or 2.0)
    decisions_per_ep = steps_per_ep / divisor
    main_mod.TOTAL_UPDATES = max(1, int(main_mod.EPOCHS * main_mod.TRAIN_SIZE * decisions_per_ep))


def main() -> None:
    parser = argparse.ArgumentParser(description="RL grid search runner (RL version of GridSearchCV).")
    parser.add_argument("--reward_space", default=None, help="Path to reward_search_space.json")
    parser.add_argument("--policy_space", default=None, help="Path to policy_search_space.json")
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Training epochs per trial (overrides root config EPOCHS; default 600)",
    )
    parser.add_argument("--use_gui", action="store_true", help="Launch sumo-gui")
    parser.add_argument("--max_trials", type=int, default=0, help="0 = no limit")
    parser.add_argument("--diverged_loss_threshold", type=float, default=1e4, help="Mean loss threshold")
    parser.add_argument("--output_root", default="experiments/grid_search", help="Root folder for the sweep")
    args = parser.parse_args()
    _apply_training_epochs_for_grid(args.epochs)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    reward_space_path = args.reward_space or os.path.join(this_dir, "reward_search_space.json")
    policy_space_path = args.policy_space or os.path.join(this_dir, "policy_search_space.json")

    main_mod.validate_inputs(main_mod.SUMO_HOME)
    grid_timestamp = _timestamp()
    out_root = args.output_root
    grid_dir = os.path.join(out_root, "runs", grid_timestamp)
    _ensure_dir(grid_dir)

    # Generate route/network once for all trials.
    split = main_mod.run_build_route()
    net_file = main_mod.run_build_network(main_mod.SUMO_HOME)

    flows_dir = os.path.join(main_mod.OUT_DIR, "sumo", "flows")
    rewards_list = load_trials(reward_space_path, policy_space_path)
    trials = rewards_list[: args.max_trials] if args.max_trials and args.max_trials > 0 else rewards_list

    results_csv_path = os.path.join(grid_dir, "results.csv")
    leaderboard_csv_path = os.path.join(grid_dir, "leaderboard.csv")

    # Decide CSV schema.
    base_fieldnames = [
        "trial_id",
        "status",
        "seed",
        "reward_class_name",
        "policy_class_name",
        "config_key",
        "test_crossings_rate_mean",
        "test_crossings_rate_std",
        "test_crossings_total_mean",
        "test_crossings_total_std",
        "test_throughput_rate_mean",
        "test_throughput_rate_std",
        "test_throughput_total_mean",
        "test_throughput_total_std",
        "test_neg_lane_waiting_integral_mean",
        "test_neg_lane_waiting_integral_std",
    ]

    # Flatten param kwargs into fields. Use the union across all trials so the
    # CSV schema is stable even when reward/policy param sets differ.
    reward_key_union: set[str] = set()
    policy_key_union: set[str] = set()
    for t in trials:
        reward_key_union.update(t.reward_kwargs.keys())
        policy_key_union.update(t.policy_kwargs.keys())

    fieldnames = list(base_fieldnames)
    fieldnames.extend([f"reward.{k}" for k in sorted(reward_key_union)])
    fieldnames.extend([f"policy.{k}" for k in sorted(policy_key_union)])

    _write_results_header(results_csv_path, fieldnames=fieldnames)

    for i, trial in enumerate(trials, start=1):
        trial_dir = os.path.join(
            grid_dir, f"{trial.trial_id}_{trial.policy_class_name}_{trial.reward_class_name}"
        )
        _ensure_dir(trial_dir)
        agent = None
        try:
            np.random.seed(trial.seed)

            print(
                f"\n[gridsearch] {i}/{len(trials)} starting trial={trial.trial_id} seed={trial.seed}\n"
                f"  Reward={trial.reward_class_name} kwargs={trial.reward_kwargs}\n"
                f"  Policy={trial.policy_class_name} kwargs={trial.policy_kwargs}\n"
                f"  config_key={_config_key(trial)}"
            )

            _write_trial_snapshot(trial_dir, trial)

            # Build agent per trial seed.
            agent = _build_agent(
                net_file=net_file,
                use_gui=bool(args.use_gui),
                reward_class_name=trial.reward_class_name,
                reward_kwargs=trial.reward_kwargs,
                policy_class_name=trial.policy_class_name,
                policy_kwargs=trial.policy_kwargs,
                seed=trial.seed,
            )

            # Train
            train_log: list[dict[str, Any]] = []
            train_crossings_rates: list[float] = []
            train_crossings_totals: list[float] = []
            train_losses: list[float] = []
            episode_counter = 0
            checkpoints_dir = os.path.join(trial_dir, "checkpoints")
            _ensure_dir(checkpoints_dir)
            for epoch in range(1, main_mod.EPOCHS + 1):
                for day_id in split["train"]:
                    episode_counter += 1
                    route_file = os.path.join(flows_dir, f"flows_day_{day_id:02d}.rou.xml")
                    ep_stats = _run_episode_loop(agent, route_file)
                    ep_stats["crossings_total"] = float(
                        ep_stats.get("crossings_total", ep_stats.get("kpi_crossings_total", 0))
                    )
                    ep_stats["crossings_rate"] = float(
                        ep_stats.get("crossings_rate", ep_stats.get("kpi_crossings_rate", 0))
                    )
                    ep_cross_rate = ep_stats["crossings_rate"]
                    ep_stats["epoch"] = epoch
                    ep_stats["day_id"] = day_id
                    ep_stats["episode"] = episode_counter

                    train_log.append(ep_stats)
                    train_crossings_totals.append(float(ep_stats["crossings_total"]))
                    train_crossings_rates.append(ep_cross_rate)
                    train_loss = ep_stats.get("mean_loss")
                    if train_loss is not None:
                        train_losses.append(float(train_loss))

                    if _is_diverged(train_loss, threshold=args.diverged_loss_threshold):
                        raise RuntimeError(f"diverged: mean_loss={train_loss}")

                    if main_mod.SAVE_EVERY and episode_counter % main_mod.SAVE_EVERY == 0:
                        ck_path = os.path.join(
                            checkpoints_dir,
                            f"checkpoint_ep{episode_counter:04d}.pt",
                        )
                        agent.save(ck_path)

            # Test
            agent.set_eval_mode()
            test_log: list[dict[str, Any]] = []
            test_total_rewards: list[float] = []
            test_losses: list[float] = []

            test_episode_counter = 0
            for test_day_id in split["test"]:
                test_episode_counter += 1
                route_file = os.path.join(flows_dir, f"flows_day_{test_day_id:02d}.rou.xml")
                ep_stats = _run_test_episode_save_sequence(
                    agent, route_file, trial_dir, test_episode_counter
                )
                ep_stats["crossings_total"] = float(
                    ep_stats.get("crossings_total", ep_stats.get("kpi_crossings_total", 0))
                )
                ep_stats["crossings_rate"] = float(
                    ep_stats.get("crossings_rate", ep_stats.get("kpi_crossings_rate", 0))
                )
                ep_cross_rate = ep_stats["crossings_rate"]
                ep_stats["day_id"] = test_day_id
                ep_stats["episode"] = test_episode_counter

                test_log.append(ep_stats)

                test_total_rewards.append(float(ep_stats.get("total_reward", 0.0)))
                ep_loss = ep_stats.get("mean_loss")
                if ep_loss is not None:
                    test_losses.append(float(ep_loss))

            _sim_elapsed = float(main_mod.SIM_END - main_mod.SIM_BEGIN)
            kpi_agg = aggregate_test_kpis(test_log, _sim_elapsed)
            days_dir = os.path.join(main_mod.OUT_DIR, "processed", "days")
            _sched = write_schedule_for_run(trial_dir, test_log, days_dir, warn=print)
            if _sched:
                print(f"[gridsearch] schedule.json -> {_sched}")

            # Save trial artifacts (logs + final model).
            with open(os.path.join(trial_dir, "train_log.json"), "w", encoding="utf-8") as f:
                json.dump(train_log, f, indent=2)
            with open(os.path.join(trial_dir, "test_log.json"), "w", encoding="utf-8") as f:
                json.dump(test_log, f, indent=2)
            agent.save(os.path.join(trial_dir, "final_model.pt"))

            # Serialize CSV row.
            row: dict[str, Any] = {
                "trial_id": trial.trial_id,
                "status": "ok",
                "seed": trial.seed,
                "reward_class_name": trial.reward_class_name,
                "policy_class_name": trial.policy_class_name,
                "config_key": _config_key(trial),
                **kpi_agg,
            }
            for k, v in sorted(trial.reward_kwargs.items()):
                row[f"reward.{k}"] = v
            for k, v in sorted(trial.policy_kwargs.items()):
                row[f"policy.{k}"] = v

            _append_result_row(results_csv_path, row, fieldnames=fieldnames)

            # Write trial config.txt summary similar to your current runs.
            train_example = train_log[-1] if train_log else None
            test_example = test_log[-1] if test_log else None
            _write_config_txt(
                trial_dir=trial_dir,
                trial=trial,
                train_metrics_example=train_example,
                test_metrics_example=test_example,
            )

            _try_render_trial_graphs(trial_dir)

        except Exception as e:
            # Write failure row but keep sweep running.
            fail_row: dict[str, Any] = {
                "trial_id": trial.trial_id,
                "status": "failed",
                "seed": trial.seed,
                "reward_class_name": trial.reward_class_name,
                "policy_class_name": trial.policy_class_name,
                "config_key": _config_key(trial),
                "test_crossings_rate_mean": 0.0,
                "test_crossings_rate_std": 0.0,
                "test_crossings_total_mean": 0.0,
                "test_crossings_total_std": 0.0,
                "test_throughput_rate_mean": 0.0,
                "test_throughput_rate_std": 0.0,
                "test_throughput_total_mean": 0.0,
                "test_throughput_total_std": 0.0,
                "test_neg_lane_waiting_integral_mean": 0.0,
                "test_neg_lane_waiting_integral_std": 0.0,
            }
            for k, v in sorted(trial.reward_kwargs.items()):
                fail_row[f"reward.{k}"] = v
            for k, v in sorted(trial.policy_kwargs.items()):
                fail_row[f"policy.{k}"] = v

            _append_result_row(results_csv_path, fail_row, fieldnames=fieldnames)
            with open(os.path.join(trial_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())

            # Best-effort: persist partial logs + config summary.
            try:
                partial_train_log = locals().get("train_log")
                partial_test_log = locals().get("test_log")
                if partial_train_log is not None:
                    with open(os.path.join(trial_dir, "train_log.json"), "w", encoding="utf-8") as f2:
                        json.dump(partial_train_log, f2, indent=2)
                if partial_test_log is not None:
                    with open(os.path.join(trial_dir, "test_log.json"), "w", encoding="utf-8") as f2:
                        json.dump(partial_test_log, f2, indent=2)

                train_example = partial_train_log[-1] if partial_train_log else None
                test_example = partial_test_log[-1] if partial_test_log else None
                _write_config_txt(
                    trial_dir=trial_dir,
                    trial=trial,
                    train_metrics_example=train_example,
                    test_metrics_example=test_example,
                )
                if partial_train_log:
                    _try_render_trial_graphs(trial_dir)
            except Exception:
                pass

        finally:
            # Ensure agent resources are released if created.
            if agent is not None:
                try:
                    agent.environment.close()
                except Exception:
                    pass

        # Update leaderboard progressively so you can monitor.
        try:
            if os.path.exists(results_csv_path):
                build_leaderboard(results_csv_path, leaderboard_csv_path)
        except Exception:
            pass

        print(f"[{i}/{len(trials)}] finished {trial.trial_id}")


if __name__ == "__main__":
    main()

