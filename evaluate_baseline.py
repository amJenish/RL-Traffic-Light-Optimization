"""
evaluate_baseline.py
--------------------
Runs FixedTimePolicy and ActuatedPolicy through the same test-day evaluation
loop used for the DQN agent, then prints a three-way comparison table.

Uses the n_epochs=0 trick: passing n_epochs=0 to Trainer skips the training
loop entirely and runs only the held-out test-day evaluation.  No new
evaluation infrastructure is needed.

Usage:
    python evaluate_baseline.py \\
        --config        config.json \\
        --intersection  src/intersection.json \\
        --sumo-home     <SUMO_HOME>

    Baseline min/max green steps match Agent: ceil(seconds / step_length) per sim step.
"""

import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _abort(msg: str):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def _check_file(path: str, label: str):
    if not os.path.exists(path):
        _abort(f"{label} not found at: {path}")


def _banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# COMPONENT FACTORY
# ---------------------------------------------------------------------------

def _build_agent(cfg: dict, int_cfg: dict, policy, sumo_home: str, gui: bool):
    """Construct a fresh Agent wired to the given policy."""
    from modelling.components.environment.sumo_environment import SumoEnvironment
    from modelling.components.observation.queue_observation import QueueObservation
    from modelling.components.reward.wait_time              import WaitTimeReward
    from modelling.components.replay_buffer.uniform         import UniformReplayBuffer
    from modelling.agent                                     import Agent

    ocfg = cfg["observation"]
    rcfg = cfg["reward"]
    scfg = cfg["simulation"]

    net_file = os.path.join(
        cfg["output"]["out_dir"], "sumo", "network",
        f"{int_cfg['intersection_name']}.net.xml",
    )
    _check_file(net_file, f"SUMO network file")

    obs_builder = QueueObservation(
        max_lanes = ocfg["max_lanes"],
        max_phase = ocfg["max_phase"],
        max_phase_time = ocfg["max_phase_time"],
        max_vehicles = ocfg["max_vehicles"],
    )

    environment = SumoEnvironment(
        net_file = net_file,
        step_length = scfg["step_length"],
        decision_gap = scfg["decision_gap"],
        gui = gui,
        sumo_home = sumo_home,
        begin = scfg.get("begin", 27000),
        end = scfg.get("end",   64800),
    )

    reward = WaitTimeReward(
        normalise = rcfg["normalise"],
        scale = rcfg["scale"],
    )

    replay_buffer = UniformReplayBuffer(
        capacity = cfg["replay_buffer"]["capacity"],
        seed = cfg["training"]["seed"],
    )

    min_green_s = float(int_cfg.get("min_green_s", 15))
    max_green_s = float(int_cfg.get("max_green_s", 90))
    step_length = float(scfg["step_length"])

    return Agent(
        environment = environment,
        observation = obs_builder,
        reward = reward,
        policy = policy,
        replay_buffer = replay_buffer,
        step_length = step_length,
        min_green_s = min_green_s,
        max_green_s = max_green_s,
    )


# ---------------------------------------------------------------------------
# BASELINE RUNNER
# ---------------------------------------------------------------------------

def _run_baseline(
    name: str,
    policy,
    cfg: dict,
    int_cfg: dict,
    sumo_home: str,
    gui: bool,
) -> list:
    """
    Evaluate one baseline policy on all test days and return the test log.

    Each baseline gets its own temporary output directory so the DQN
    test_log.json is never overwritten.
    """
    from modelling.trainer import Trainer

    _banner(f"Evaluating — {name}")
    print(f"  Policy : {policy!r}")

    agent = _build_agent(cfg, int_cfg, policy, sumo_home, gui)

    split_path = os.path.join(cfg["output"]["out_dir"], "processed", "split.json")
    flows_dir = os.path.join(cfg["output"]["out_dir"], "sumo", "flows")
    tmp_dir = os.path.join(
        cfg["output"]["models_dir"],
        f"_tmp_{name.lower().replace(' ', '_').replace('-', '_')}",
    )

    trainer = Trainer(
        agent = agent,
        split_path = split_path,
        flows_dir = flows_dir,
        output_dir = tmp_dir,
        n_epochs = 0,
    )

    results = trainer.run()
    return results["test_log"]


# ---------------------------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------------------------

def _print_table(logs: dict) -> None:
    """Print a formatted side-by-side reward table across all test days."""
    _banner("Results Comparison")

    day_ids = sorted({m["day_id"] for entries in logs.values() for m in entries})
    col_w = 16

    # Header
    header = f"{'Day':<6}" + "".join(f"{name:>{col_w}}" for name in logs)
    sep = "-" * len(header)
    print(header)
    print(sep)

    # Per-day rows
    reward_map = {
        model: {m["day_id"]: m["total_reward"] for m in entries}
        for model, entries in logs.items()
    }
    for day_id in day_ids:
        row = f"{day_id:<6}"
        for model, day_rewards in reward_map.items():
            val = day_rewards.get(day_id)
            row += f"{val:>{col_w}.2f}" if val is not None else f"{'N/A':>{col_w}}"
        print(row)

    # Mean row
    print(sep)
    mean_row = f"{'Mean':<6}"
    for entries in logs.values():
        mean = sum(m["total_reward"] for m in entries) / len(entries)
        mean_row += f"{mean:>{col_w}.2f}"
    print(mean_row)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fixed-Time and Actuated baselines, then compare against DQN"
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--intersection", required=True,
        help="Path to intersection.json",
    )
    parser.add_argument(
        "--sumo-home",
        default=os.environ.get("SUMO_HOME", ""),
        help="Path to SUMO installation directory",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch sumo-gui instead of headless sumo",
    )
    args = parser.parse_args()

    # Ensure project root is on sys.path so modelling/* imports resolve
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Validate inputs
    _check_file(args.config, "config.json")
    _check_file(args.intersection, "intersection.json")
    if not args.sumo_home:
        _abort(
            "SUMO_HOME not set. Pass --sumo-home or set the SUMO_HOME "
            "environment variable.\nExample: C:\\Program Files (x86)\\Eclipse\\Sumo"
        )
    if not os.path.exists(args.sumo_home):
        _abort(f"SUMO not found at: {args.sumo_home}")

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.intersection) as f:
        int_cfg = json.load(f)

    try:
        from baselines.fixed_time import FixedTimePolicy
        from baselines.actuated   import ActuatedPolicy
    except ImportError as e:
        _abort(f"Baseline import failed: {e}")

    # Same units as Agent: one policy "step" = one SUMO step after min lockout.
    # min/max green in sim steps = ceil(seconds / step_length) — NOT decision_gap×step_length.
    scfg = cfg["simulation"]
    step_length = float(scfg["step_length"])
    min_green_s = float(int_cfg.get("min_green_s", 15))
    max_green_s = float(int_cfg.get("max_green_s", 90))
    min_green_steps = max(1, math.ceil(min_green_s / step_length))
    max_green_steps = max(min_green_steps, max(1, math.ceil(max_green_s / step_length)))
    # Hold green until timer fires; pick mid-span so fixed-time is not "switch every step".
    fixed_green_steps = min(
        max_green_steps,
        max(min_green_steps, (min_green_steps + max_green_steps) // 2),
    )
    max_lanes = cfg["observation"]["max_lanes"]

    fixed_policy = FixedTimePolicy(
        fixed_green_steps = fixed_green_steps,
        min_green_steps   = min_green_steps,
        max_green_steps   = max_green_steps,
    )
    actuated_policy = ActuatedPolicy(
        max_lanes       = max_lanes,
        min_green_steps = min_green_steps,
        max_green_steps = max_green_steps,
        queue_threshold = 0.1,               # ~2 vehicles per lane (max_vehicles=20)
    )

    _banner("Baseline Evaluation")
    print(f"  Config        : {args.config}")
    print(f"  Intersection  : {args.intersection}")
    print(f"  SUMO home     : {args.sumo_home}")
    print(
        f"  Phase steps   : min={min_green_steps}  max={max_green_steps}  "
        f"fixed={fixed_green_steps}  (step_length={step_length}s, same as Agent)"
    )

    # Run evaluations
    fixed_log = _run_baseline("Fixed-Time", fixed_policy,    cfg, int_cfg, args.sumo_home, args.gui)
    actuated_log = _run_baseline("Actuated",   actuated_policy, cfg, int_cfg, args.sumo_home, args.gui)

    # Persist results
    models_dir = cfg["output"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    fixed_path = os.path.join(models_dir, "baseline_fixed_time_log.json")
    actuated_path = os.path.join(models_dir, "baseline_actuated_log.json")

    with open(fixed_path, "w") as f:
        json.dump(fixed_log, f, indent=2)
    with open(actuated_path, "w") as f:
        json.dump(actuated_log, f, indent=2)

    _banner("Logs Saved")
    print(f"  {fixed_path}")
    print(f"  {actuated_path}")

    # Load DQN results for the comparison table
    logs = {}
    dqn_path = os.path.join(models_dir, "test_log.json")
    if os.path.exists(dqn_path):
        with open(dqn_path) as f:
            logs["DQN"] = json.load(f)
    else:
        print(f"\nNote: {dqn_path} not found — DQN omitted from comparison table.")

    logs["Fixed-Time"] = fixed_log
    logs["Actuated"] = actuated_log

    _print_table(logs)


if __name__ == "__main__":
    main()
