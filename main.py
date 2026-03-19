"""
main.py
--------
Single entry point for the full RL traffic signal optimisation pipeline.

Steps:
  1. Load config.json and validate all input files
  2. Run BuildRoute  → day CSVs + SUMO flow files + split.json
  3. Run BuildNetwork → SUMO .net.xml
  4. Construct all components using config values
  5. Construct Agent
  6. Construct Trainer
  7. Run training + evaluation

Usage:
    python main.py \\
        --csv           src/data/synthetic_toronto_data.xls \\
        --intersection  src/intersection.json \\
        --columns       src/columns.json \\
        --sumo-home     "C:\\Program Files (x86)\\Eclipse\\Sumo"

    python main.py \\
        --csv           src/data/synthetic_toronto_data.xls \\
        --intersection  src/intersection.json \\
        --columns       src/columns.json \\
        --sumo-home     "C:\\Program Files (x86)\\Eclipse\\Sumo" \\
        --config        config.json \\
        --gui
"""

import argparse
import importlib.util
import json
import os
import sys


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: str):
    """Dynamically import a .py file by path."""
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
# STEP 1 — LOAD AND VALIDATE
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    _check_file(config_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"Config loaded from {config_path}")
    return cfg


def validate_inputs(args):
    _check_file(args.csv,          "CSV data file")
    _check_file(args.intersection, "intersection.json")
    _check_file(args.columns,      "columns.json")

    root = os.path.dirname(os.path.abspath(__file__))
    _check_file(
        os.path.join(root, "preprocessing", "BuildRoute.py"),
        "preprocessing/BuildRoute.py"
    )
    _check_file(
        os.path.join(root, "preprocessing", "BuildNetwork.py"),
        "preprocessing/BuildNetwork.py"
    )

    if not args.sumo_home:
        _abort(
            "SUMO_HOME not set. Pass --sumo-home or set the SUMO_HOME "
            "environment variable.\n"
            "Example: C:\\Program Files (x86)\\Eclipse\\Sumo"
        )
    if not os.path.exists(args.sumo_home):
        _abort(f"SUMO not found at: {args.sumo_home}")

    print("All input files validated.")


# ---------------------------------------------------------------------------
# STEP 2 — BUILD ROUTES
# ---------------------------------------------------------------------------

def run_build_route(args, cfg: dict) -> dict:
    _banner("Step 1/3 — Building routes and day CSVs")

    root       = os.path.dirname(os.path.abspath(__file__))
    br         = _load_module("BuildRoute",
                               os.path.join(root, "preprocessing", "BuildRoute.py"))
    out_dir    = cfg["output"]["out_dir"]
    test_days  = cfg["training"]["test_days"]
    seed       = cfg["training"]["seed"]

    sys.argv = [
        "BuildRoute.py",
        "--csv",          args.csv,
        "--intersection", args.intersection,
        "--columns",      args.columns,
        "--out-dir",      out_dir,
        "--test-days",    str(test_days),
        "--seed",         str(seed),
    ]
    br.main()

    split_path = os.path.join(out_dir, "processed", "split.json")
    _check_file(split_path, "split.json (BuildRoute output)")

    with open(split_path) as f:
        split = json.load(f)

    print(f"Routes built. Train: {len(split['train'])} days | Test: {len(split['test'])} days")
    return split


# ---------------------------------------------------------------------------
# STEP 3 — BUILD NETWORK
# ---------------------------------------------------------------------------

def run_build_network(args, cfg: dict) -> str:
    _banner("Step 2/3 — Building SUMO network")

    root       = os.path.dirname(os.path.abspath(__file__))
    bn         = _load_module("BuildNetwork",
                               os.path.join(root, "preprocessing", "BuildNetwork.py"))
    out_dir    = os.path.join(cfg["output"]["out_dir"], "sumo", "network")

    with open(args.intersection) as f:
        int_cfg = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    nod = bn.write_nodes(int_cfg,       out_dir)
    edg = bn.write_edges(int_cfg,       out_dir)
    con = bn.write_connections(int_cfg, out_dir)
    tll = bn.write_tll(int_cfg,         out_dir)

    name    = int_cfg["intersection_name"]
    net_file = os.path.join(out_dir, f"{name}.net.xml")

    try:
        bn.run_netconvert(int_cfg, out_dir, nod, edg, con, tll)
    except FileNotFoundError:
        _abort(
            "netconvert not found. Make sure SUMO is installed and its bin/ "
            "folder is on your PATH.\n"
            f"Expected binary in: {os.path.join(args.sumo_home, 'bin')}"
        )

    _check_file(net_file, f"{name}.net.xml (BuildNetwork output)")
    print(f"Network built → {net_file}")
    return net_file


# ---------------------------------------------------------------------------
# STEP 4 — CONSTRUCT COMPONENTS
# ---------------------------------------------------------------------------

def build_components(args, cfg: dict, split: dict, net_file: str):
    _banner("Step 3/3 — Constructing RL components")

    # Add project root to path so modeling imports resolve
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Lazy import — torch required for DQN
    try:
        from modelling.components.environment.sumo_environment import SumoEnvironment
        from modelling.components.observation.queue_observation import QueueObservation
        from modelling.components.reward.wait_time             import WaitTimeReward
        from modelling.components.policy.dqn                   import DQNPolicy
        from modelling.components.replay_buffer.uniform        import UniformReplayBuffer
        from modelling.agent   import Agent
        from modelling.trainer import Trainer
    except ImportError as e:
        _abort(
            f"Import failed: {e}\n"
            f"Make sure torch is installed: pip install torch"
        )

    ocfg  = cfg["observation"]
    rcfg  = cfg["reward"]
    pcfg  = cfg["policy"]
    rbcfg = cfg["replay_buffer"]
    scfg  = cfg["simulation"]

    # Override GUI flag if passed on command line
    use_gui = args.gui or scfg.get("gui", False)

    obs_builder = QueueObservation(
        max_lanes      = ocfg["max_lanes"],
        max_phase      = ocfg["max_phase"],
        max_phase_time = ocfg["max_phase_time"],
        max_vehicles   = ocfg["max_vehicles"],
    )

    environment = SumoEnvironment(
        net_file     = net_file,
        step_length  = scfg["step_length"],
        decision_gap = scfg["decision_gap"],
        gui          = use_gui,
        sumo_home    = args.sumo_home,
        begin        = scfg.get("begin", 27000),
        end          = scfg.get("end",   64800),
    )

    with open(args.intersection) as f:
        _int_cfg = json.load(f)

    reward = WaitTimeReward(
        normalise = rcfg["normalise"],
        scale     = rcfg["scale"],
    )

    policy = DQNPolicy(
        obs_dim       = obs_builder.size(),
        n_actions     = 2,
        lr            = pcfg["learning_rate"],
        gamma         = pcfg["gamma"],
        epsilon_start = pcfg["epsilon_start"],
        epsilon_end   = pcfg["epsilon_end"],
        epsilon_decay = pcfg["epsilon_decay"],
        target_update = pcfg["target_update"],
        batch_size    = pcfg["batch_size"],
        hidden        = pcfg["hidden"],
        min_green_steps = _int_cfg.get("min_green_s", 15),
        max_green_steps = _int_cfg.get("max_green_s", 90),
    )

    replay_buffer = UniformReplayBuffer(
        capacity = rbcfg["capacity"],
        seed     = cfg["training"]["seed"],
    )

    print(f"  Environment   : SumoEnvironment (gui={use_gui})")
    print(f"  Observation   : QueueObservation (size={obs_builder.size()})")
    print(f"  Reward        : WaitTimeReward")
    print(f"  Policy        : DQNPolicy (device={policy.device})")
    print(f"  Replay buffer : UniformReplayBuffer (capacity={rbcfg['capacity']:,})")

    agent = Agent(
        environment   = environment,
        observation   = obs_builder,
        reward        = reward,
        policy        = policy,
        replay_buffer = replay_buffer,
    )

    flows_dir  = os.path.join(cfg["output"]["out_dir"], "sumo", "flows")
    split_path = os.path.join(cfg["output"]["out_dir"], "processed", "split.json")
    models_dir = cfg["output"]["models_dir"]
    tcfg       = cfg["training"]

    trainer = Trainer(
        agent      = agent,
        split_path = split_path,
        flows_dir  = flows_dir,
        output_dir = models_dir,
        n_epochs   = tcfg["n_epochs"],
        save_every = tcfg["save_every"],
        log_every  = tcfg["log_every"],
    )

    return trainer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RL Traffic Signal Optimiser — full pipeline"
    )
    parser.add_argument("--csv",          required=True,
                        help="Path to traffic count CSV")
    parser.add_argument("--intersection", required=True,
                        help="Path to intersection.json")
    parser.add_argument("--columns",      required=True,
                        help="Path to columns.json")
    parser.add_argument("--sumo-home",    
                        default=os.environ.get("SUMO_HOME", ""),
                        help="Path to SUMO installation directory")
    parser.add_argument("--config",       default="config.json",
                        help="Path to config.json (default: config.json)")
    parser.add_argument("--gui",          action="store_true",
                        help="Launch sumo-gui instead of headless sumo")
    args = parser.parse_args()

    _banner("RL Traffic Signal Optimiser")
    print(f"  CSV           : {args.csv}")
    print(f"  Intersection  : {args.intersection}")
    print(f"  Columns       : {args.columns}")
    print(f"  SUMO home     : {args.sumo_home}")
    print(f"  Config        : {args.config}")

    # 1. Load and validate
    cfg = load_config(args.config)
    validate_inputs(args)

    # 2. Build routes
    split = run_build_route(args, cfg)

    # 3. Build network
    net_file = run_build_network(args, cfg)

    # 4. Construct components + agent + trainer
    trainer = build_components(args, cfg, split, net_file)

    # 5. Run
    _banner("Training")
    results = trainer.run()

    _banner("Complete")
    print(f"  Train episodes : {len(results['train_log'])}")
    print(f"  Test episodes  : {len(results['test_log'])}")
    if results["test_log"]:
        mean_test = sum(m["total_reward"] for m in results["test_log"]) / len(results["test_log"])
        print(f"  Mean test reward : {mean_test:.2f}")
    print(f"\nModels and logs saved to: {cfg['output']['models_dir']}")


if __name__ == "__main__":
    main()