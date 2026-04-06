"""
Run one short SUMO episode; optional Agent RL_SWITCH_DEBUG tracing (T1–T5).

Usage (from repo root):
  python scripts/run_switch_debug_episode.py          # quiet (no SWITCH_DEBUG)
  python scripts/run_switch_debug_episode.py --debug  # T1–T5 prints

Requires SUMO_HOME, generated flows/net (calls run_build_route + run_build_network).
Simulation end is forced to SIM_BEGIN + 300 seconds for a fast run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Short episode for TLS switch debugging.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set RL_SWITCH_DEBUG=1 (T1–T5 traces in modelling/agent.py)",
    )
    args = parser.parse_args()
    if args.debug:
        os.environ["RL_SWITCH_DEBUG"] = "1"
    else:
        os.environ.pop("RL_SWITCH_DEBUG", None)

    import main as m

    m.validate_inputs(m.SUMO_HOME)
    m.run_build_route()
    net_file = m.run_build_network(m.SUMO_HOME)

    old_end = m.SIM_END
    m.SIM_END = m.SIM_BEGIN + 300
    try:
        reward_kwargs, _ = m._load_component_config(
            m.REWARD_CONFIG_PATH, m.RewardClass.__name__, "Reward"
        )
        reward_kwargs = m.merge_reward_kwargs_from_config(reward_kwargs)
        policy_raw, _ = m._load_component_config(
            m.POLICY_CONFIG_PATH, m.PolicyClass.__name__, "Policy"
        )
        policy_raw = m.merge_policy_kwargs_from_config(policy_raw)
        policy_kwargs = m._resolve_policy_params(policy_raw)

        run_dir = os.path.join(m.RESULTS_DIR, "_switch_debug_episode")
        os.makedirs(run_dir, exist_ok=True)
        trainer = m.build_pipeline(net_file, False, run_dir, reward_kwargs, policy_kwargs)
        agent = trainer.agent

        split_path = os.path.join(m.OUT_DIR, "processed", "split.json")
        with open(split_path, encoding="utf-8") as f:
            split = json.load(f)
        day_id = split["train"][0]
        flows_dir = os.path.join(m.OUT_DIR, "sumo", "flows")
        route_file = os.path.join(flows_dir, f"flows_day_{day_id:02d}.rou.xml")

        print(
            f"=== switch debug: SIM_BEGIN={m.SIM_BEGIN} SIM_END={m.SIM_END} "
            f"route={route_file} ===",
            flush=True,
        )
        agent.set_train_mode()
        obs = agent.start_episode(route_file)
        done = False
        while not done:
            obs, _, done, _ = agent.step(obs)
        agent.end_episode()
        print("=== switch debug episode finished ===", flush=True)
    finally:
        m.SIM_END = old_end


if __name__ == "__main__":
    main()
