"""
main.py — single entry point for the full RL traffic signal optimisation pipeline.
All configuration lives at the top. Edit then run: python main.py [--gui]
"""
from typing import Any
import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _default_sumo_home() -> str:
    """Fallback when SUMO_HOME is not set in the environment.

    Windows: same default as historical ``main.py`` — prefer a real install under
    ``Program Files (x86)`` or ``Program Files`` if ``bin`` exists, else the classic x86 path.
    macOS: ``pip install eclipse-sumo`` (``import sumo``), then common Homebrew layouts.
    """
    if sys.platform == "win32":
        _win_candidates = (
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
        )
        for candidate in _win_candidates:
            if os.path.isdir(os.path.join(candidate, "bin")):
                return candidate
        return _win_candidates[0]
    if sys.platform == "darwin":
        try:
            import sumo  # noqa: PLC0415

            return sumo.SUMO_HOME
        except Exception:
            pass
        for candidate in ("/opt/homebrew/opt/sumo", "/usr/local/opt/sumo"):
            if os.path.isdir(os.path.join(candidate, "bin")):
                return candidate
    try:
        import sumo  # noqa: PLC0415

        return sumo.SUMO_HOME
    except Exception:
        return r"C:\Program Files (x86)\Eclipse\Sumo"


from modelling.components.environment.sumo_environment import SumoEnvironment
from modelling.components.observation.queue_observation import QueueObservation
from modelling.components.reward.wait_time              import WaitTimeReward
from modelling.components.policy.dqn                    import DQNPolicy
from modelling.components.policy.double_dqn             import DoubleDQNPolicy
from modelling.components.replay_buffer.uniform         import UniformReplayBuffer
from modelling.components.scheduler.cosine              import CosineScheduler
from modelling.agent   import Agent
from modelling.trainer import Trainer
from modelling.components.reward.delta_wait_time import DeltaWaitTimeReward
from modelling.components.reward.composite_reward import CompositeReward
from modelling.components.reward.throughput import ThroughputReward
from modelling.components.reward.throughput_queue import ThroughputQueueReward


# ==========================================================================
#  CONFIGURATION
# ==========================================================================

# File Paths
CSV_PATH          = "src/data/synthetic_toronto_data.csv"
INTERSECTION_PATH = "src/intersection.json"
COLUMNS_PATH      = "src/columns.json"
REWARD_CONFIG_PATH = os.path.join(
    ROOT, "modelling", "components", "reward", "reward_configuration.json"
)
POLICY_CONFIG_PATH = os.path.join(
    ROOT, "modelling", "components", "policy", "policy_configuration.json"
)
SUMO_HOME         = os.environ.get("SUMO_HOME") or _default_sumo_home()

# Data Split & Training
TRAIN_SIZE = 25
TEST_SIZE  = 5
EPOCHS     = 60

# Component Selection — swap any of these to use a different implementation
# Rewards: CompositeReward | ThroughputQueueReward | ThroughputReward | DeltaWaitTimeReward | WaitTimeReward
EnvironmentClass  = SumoEnvironment
ObservationClass  = QueueObservation
RewardClass       = ThroughputReward
EvalRewardClass   = None              # optional; None = same reward for train and eval
PolicyClass       = DoubleDQNPolicy
ReplayBufferClass = UniformReplayBuffer
SchedulerClass    = CosineScheduler               # disable LR scheduling for stability

# Simulation
STEP_LENGTH  = 10.0      # seconds per SUMO step
SIM_BEGIN    = 28800      # 08:00
# Sim window matches c881448; affects TOTAL_UPDATES / epsilon schedule
SIM_END      = 50400      # 14:00

# Phase Timing (seconds — Agent converts to sim steps internally)
MIN_GREEN_S      = 15
MAX_GREEN_S      = 90
OVERSHOOT_COEFF  = 1.0     # how harshly to penalize exceeding max_green (higher = harsher)

# Observation
MAX_LANES      = 16
MAX_PHASE      = 7       # max phase index (McCowan_Finch TLS has 8 phases → 0..7)
MAX_PHASE_TIME = 120.0
MAX_VEHICLES   = 20

# Policy scheduler floor (policy hyperparameters come from policy_configuration.json)
LR_MIN         = 0.0001

# Replay Buffer
BUFFER_CAPACITY = 4096

# Misc
SEED       = 42
SAVE_EVERY = 20
LOG_EVERY  = 1

# Output
OUT_DIR  = "src/data"
LOGS_DIR = "logs"

# Estimated total update steps (for the LR scheduler and epsilon decay)
_sim_seconds     = SIM_END - SIM_BEGIN
_steps_per_ep    = _sim_seconds / STEP_LENGTH
_min_green_steps = max(1, math.ceil(MIN_GREEN_S / STEP_LENGTH))
_decisions_per_ep = _steps_per_ep / 2  # empirical: ~1 decision per 2 sim steps on average
TOTAL_UPDATES    = int(EPOCHS * TRAIN_SIZE * _decisions_per_ep)
if TOTAL_UPDATES < 1:
    TOTAL_UPDATES = 1

# ==========================================================================
#  HELPERS
# ==========================================================================

def _load_module(name: str, filepath: str):
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


def _load_component_config(path: str, class_name: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load config JSON and merge default + class-specific blocks."""
    if not os.path.exists(path):
        _abort(f"{label} config not found: {path}")
    with open(path) as f:
        all_cfg = json.load(f)
    default_cfg = all_cfg.get("default", {})
    class_cfg = all_cfg.get(class_name, {})
    if not isinstance(default_cfg, dict) or not isinstance(class_cfg, dict):
        _abort(f"{label} config must map keys to objects: {path}")
    merged = {**default_cfg, **class_cfg}
    return merged, all_cfg


def _resolve_policy_params(policy_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate and finalize policy kwargs (including optional auto epsilon decay)."""
    resolved = dict(policy_kwargs)
    required = [
        "lr", "gamma", "epsilon_start", "epsilon_end",
        "target_update", "batch_size", "hidden", "n_actions",
    ]
    missing = [k for k in required if k not in resolved]
    if missing:
        _abort(f"Missing policy params in policy config: {missing}")

    eps_decay = resolved.get("epsilon_decay", "auto")
    if eps_decay == "auto":
        eps_start = float(resolved["epsilon_start"])
        eps_end = float(resolved["epsilon_end"])
        if eps_start <= 0 or eps_end <= 0:
            _abort("epsilon_start and epsilon_end must be > 0 for auto epsilon decay.")
        resolved["epsilon_decay"] = (eps_end / eps_start) ** (1.0 / (0.85 * TOTAL_UPDATES))
    return resolved


def _create_run_dir() -> str:
    """Create a timestamped run folder inside LOGS_DIR and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{PolicyClass.__name__}_{RewardClass.__name__}"
    run_dir = os.path.join(LOGS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def _write_config_summary(
    run_dir: str,
    use_gui: bool,
    reward_kwargs: dict[str, Any],
    policy_kwargs: dict[str, Any],
) -> None:
    """Write a human-readable config.txt for this run."""
    sched_name = SchedulerClass.__name__ if SchedulerClass else "None (constant LR)"
    reward_cfg_rel = os.path.relpath(REWARD_CONFIG_PATH, ROOT)
    policy_cfg_rel = os.path.relpath(POLICY_CONFIG_PATH, ROOT)
    lines = [
        f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Components ---",
        f"Policy:        {PolicyClass.__name__}",
        f"Reward:        {RewardClass.__name__} (train) / {EvalRewardClass.__name__ if EvalRewardClass else 'same'} (eval)",
        f"Observation:   {ObservationClass.__name__}",
        f"Environment:   {EnvironmentClass.__name__}",
        f"Replay Buffer: {ReplayBufferClass.__name__}",
        f"Scheduler:     {sched_name}",
        "",
        "--- Data ---",
        f"CSV:           {CSV_PATH}",
        f"Train days:    {TRAIN_SIZE}",
        f"Test days:     {TEST_SIZE}",
        f"Epochs:        {EPOCHS}",
        f"Seed:          {SEED}",
        "",
        "--- Simulation ---",
        f"Step length:   {STEP_LENGTH}s",
        f"Sim window:    {SIM_BEGIN}s - {SIM_END}s",
        f"Min green:     {MIN_GREEN_S}s",
        f"Max green:     {MAX_GREEN_S}s (soft — overshoot_coeff={OVERSHOOT_COEFF})",
        f"GUI:           {use_gui}",
        "",
        "--- Observation ---",
        f"Max lanes:     {MAX_LANES}",
        f"Max phase:     {MAX_PHASE}",
        f"Max phase time:{MAX_PHASE_TIME}",
        f"Max green feat:{MAX_GREEN_S}s (elapsed/max_green in obs)",
        f"Max vehicles:  {MAX_VEHICLES}",
        "",
        "--- Reward ---",
        f"Config file:   {reward_cfg_rel}",
        f"Resolved args: {json.dumps(reward_kwargs, sort_keys=True)}",
        "",
        "--- Policy ---",
        f"Config file:   {policy_cfg_rel}",
        f"Learning rate: {policy_kwargs['lr']} -> {LR_MIN} ({sched_name})",
        f"Total updates: ~{TOTAL_UPDATES:,} (estimated)",
        f"Gamma:         {policy_kwargs['gamma']}",
        f"Epsilon:       {policy_kwargs['epsilon_start']} -> {policy_kwargs['epsilon_end']} (decay {policy_kwargs['epsilon_decay']})",
        f"Target update: {policy_kwargs['target_update']}",
        f"Batch size:    {policy_kwargs['batch_size']}",
        f"Hidden size:   {policy_kwargs['hidden']}",
        f"N actions:     {policy_kwargs['n_actions']}",
        f"Resolved args: {json.dumps(policy_kwargs, sort_keys=True)}",
        "",
        "--- Replay Buffer ---",
        f"Capacity:      {BUFFER_CAPACITY}",
        "",
        "--- Output ---",
        f"Save every:    {SAVE_EVERY} episodes",
        f"Log every:     {LOG_EVERY} episodes",
    ]
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ==========================================================================
#  PIPELINE STEPS
# ==========================================================================

def validate_inputs(sumo_home: str):
    """Make sure all required files and SUMO are present."""
    _check_file(CSV_PATH,          "CSV data file")
    _check_file(INTERSECTION_PATH, "intersection.json")
    _check_file(COLUMNS_PATH,      "columns.json")
    _check_file(REWARD_CONFIG_PATH, "reward_configuration.json")
    _check_file(POLICY_CONFIG_PATH, "policy_configuration.json")
    _check_file(
        os.path.join(ROOT, "preprocessing", "BuildRoute.py"),
        "preprocessing/BuildRoute.py",
    )
    _check_file(
        os.path.join(ROOT, "preprocessing", "BuildNetwork.py"),
        "preprocessing/BuildNetwork.py",
    )
    if not sumo_home:
        _abort(
            "SUMO_HOME not set. Set the SUMO_HOME environment variable or "
            "edit SUMO_HOME at the top of main.py."
        )
    if not os.path.exists(sumo_home):
        _abort(f"SUMO not found at: {sumo_home}")
    print("All input files validated.")


def run_build_route() -> dict:
    """Load CSV, trim to TRAIN_SIZE+TEST_SIZE days, generate flow files and split.json."""
    _banner("Step 1/3 — Building routes and day CSVs")

    br = _load_module(
        "BuildRoute",
        os.path.join(ROOT, "preprocessing", "BuildRoute.py"),
    )

    int_cfg      = br.load_intersection(INTERSECTION_PATH)
    col_map      = br.load_column_map(COLUMNS_PATH)
    active       = int_cfg["active_approaches"]
    name         = int_cfg["intersection_name"]
    slot_minutes = col_map.get("slot_minutes", 15)

    print(f"  Intersection : {name}")
    print(f"  Approaches   : {active}")
    print(f"  Slot duration: {slot_minutes} min")

    df = br.load_csv(CSV_PATH, col_map, active)
    df = br.assign_sim_days(df, date_mode="concat")

    n_days_available = df["sim_day"].nunique()
    needed = TRAIN_SIZE + TEST_SIZE
    if needed > n_days_available:
        _abort(
            f"TRAIN_SIZE ({TRAIN_SIZE}) + TEST_SIZE ({TEST_SIZE}) = {needed} "
            f"but only {n_days_available} days available in the CSV."
        )

    print(f"  CSV has {n_days_available} days | using {needed} "
          f"(train={TRAIN_SIZE}, test={TEST_SIZE})")

    processed_dir = os.path.join(OUT_DIR, "processed")
    days_dir      = os.path.join(processed_dir, "days")
    flows_dir     = os.path.join(OUT_DIR, "sumo", "flows")

    used_days = list(range(needed))
    df_used   = df[df["sim_day"].isin(used_days)].copy()

    br.write_day_csvs(df_used, active, days_dir)
    br.write_sumo_flows(df_used, active, slot_minutes, flows_dir, int_cfg)

    split = br.split_days(needed, TEST_SIZE, SEED)

    min_red = int_cfg.get("min_red_s", 15)
    split.update({
        "intersection_name": name,
        "active_approaches": active,
        "min_red_seconds":   min_red,
        "n_days":            needed,
        "n_slots":           df["start_time"].nunique(),
        "slot_minutes":      slot_minutes,
        "edge_ids": {
            d: {"in": br.edge_in_id(d), "out": br.edge_out_id(d)}
            for d in active
        },
    })

    split_path = os.path.join(processed_dir, "split.json")
    os.makedirs(processed_dir, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"  Train: {len(split['train'])} days | Test: {len(split['test'])} days")
    print(f"  Saved -> {split_path}")
    return split


def run_build_network(sumo_home: str) -> str:
    """Generate SUMO .net.xml from intersection.json via netconvert."""
    _banner("Step 2/3 — Building SUMO network")

    bn = _load_module(
        "BuildNetwork",
        os.path.join(ROOT, "preprocessing", "BuildNetwork.py"),
    )
    net_out_dir = os.path.join(OUT_DIR, "sumo", "network")

    with open(INTERSECTION_PATH) as f:
        int_cfg = json.load(f)

    os.makedirs(net_out_dir, exist_ok=True)

    nod = bn.write_nodes(int_cfg,       net_out_dir)
    edg = bn.write_edges(int_cfg,       net_out_dir)
    con = bn.write_connections(int_cfg,  net_out_dir)
    tll = bn.write_tll(int_cfg,         net_out_dir)

    name     = int_cfg["intersection_name"]
    net_file = os.path.join(net_out_dir, f"{name}.net.xml")

    try:
        bn.run_netconvert(int_cfg, net_out_dir, nod, edg, con, tll)
    except FileNotFoundError:
        _abort(
            "netconvert not found. Make sure SUMO is installed and its bin/ "
            "folder is on your PATH.\n"
            f"Expected binary in: {os.path.join(sumo_home, 'bin')}"
        )

    _check_file(net_file, f"{name}.net.xml (BuildNetwork output)")
    print(f"Network built -> {net_file}")
    return net_file


def build_pipeline(
    net_file: str,
    use_gui: bool,
    run_dir: str,
    reward_kwargs: dict[str, Any],
    policy_kwargs: dict[str, Any],
) -> Trainer:
    """Construct all RL components, wire them into Agent and Trainer."""
    _banner("Step 3/3 — Constructing RL components")

    with open(INTERSECTION_PATH) as f:
        int_cfg = json.load(f)

    obs_builder = ObservationClass(
        max_lanes      = MAX_LANES,
        max_phase      = MAX_PHASE,
        max_phase_time = MAX_PHASE_TIME,
        max_vehicles   = MAX_VEHICLES,
        max_green_s    = MAX_GREEN_S,
    )

    environment = EnvironmentClass(
        net_file     = net_file,
        step_length  = STEP_LENGTH,
        gui          = use_gui,
        sumo_home    = SUMO_HOME,
        begin        = SIM_BEGIN,
        end          = SIM_END,
    )

    reward = RewardClass(**reward_kwargs)
    eval_reward = None
    if EvalRewardClass is not None:
        eval_kw, _ = _load_component_config(
            REWARD_CONFIG_PATH, EvalRewardClass.__name__, "Eval reward"
        )
        eval_reward = EvalRewardClass(**eval_kw)

    policy = PolicyClass(
        obs_dim         = obs_builder.size(),
        n_actions       = policy_kwargs["n_actions"],
        lr              = policy_kwargs["lr"],
        gamma           = policy_kwargs["gamma"],
        epsilon_start   = policy_kwargs["epsilon_start"],
        epsilon_end     = policy_kwargs["epsilon_end"],
        epsilon_decay   = policy_kwargs["epsilon_decay"],
        target_update   = policy_kwargs["target_update"],
        batch_size      = policy_kwargs["batch_size"],
        hidden          = policy_kwargs["hidden"],
    )

    scheduler = None
    if SchedulerClass is not None:
        scheduler = SchedulerClass(
            optimizer    = policy.optimizer,
            total_steps  = TOTAL_UPDATES,
            lr_min       = LR_MIN,
        )

    replay_buffer = ReplayBufferClass(
        capacity = BUFFER_CAPACITY,
        seed     = SEED,
    )

    sched_name = SchedulerClass.__name__ if SchedulerClass else "None"
    print(f"  Environment   : {EnvironmentClass.__name__} (gui={use_gui})")
    print(f"  Observation   : {ObservationClass.__name__} (size={obs_builder.size()})")
    eval_name = EvalRewardClass.__name__ if EvalRewardClass else "same as train"
    print(f"  Reward        : {RewardClass.__name__} (train) / {eval_name} (eval)")
    print(f"  Policy        : {PolicyClass.__name__} (device={policy.device})")
    print(f"  Scheduler     : {sched_name} (LR {policy_kwargs['lr']} -> {LR_MIN} over ~{TOTAL_UPDATES:,} steps)")
    print(f"  Replay buffer : {ReplayBufferClass.__name__} (capacity={BUFFER_CAPACITY:,})")
    print(f"  Phase timing  : min={MIN_GREEN_S}s  max={MAX_GREEN_S}s  "
          f"(decide every {STEP_LENGTH}s after min)")
    print(f"  Run folder    : {run_dir}")

    agent = Agent(
        environment     = environment,
        observation     = obs_builder,
        reward          = reward,
        policy          = policy,
        replay_buffer   = replay_buffer,
        scheduler       = scheduler,
        eval_reward     = eval_reward,
        step_length     = STEP_LENGTH,
        min_green_s     = MIN_GREEN_S,
        max_green_s     = MAX_GREEN_S,
        overshoot_coeff = OVERSHOOT_COEFF,
    )

    split_path = os.path.join(OUT_DIR, "processed", "split.json")
    flows_dir  = os.path.join(OUT_DIR, "sumo", "flows")

    trainer = Trainer(
        agent      = agent,
        split_path = split_path,
        flows_dir  = flows_dir,
        output_dir = run_dir,
        n_epochs   = EPOCHS,
        save_every = SAVE_EVERY,
        log_every  = LOG_EVERY,
    )

    return trainer


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="RL Traffic Signal Optimiser")
    parser.add_argument("--gui", action="store_true", help="Launch sumo-gui")
    args = parser.parse_args()
    use_gui = args.gui

    _banner("RL Traffic Signal Optimiser")
    print(f"  CSV           : {CSV_PATH}")
    print(f"  Intersection  : {INTERSECTION_PATH}")
    print(f"  Columns       : {COLUMNS_PATH}")
    print(f"  SUMO home     : {SUMO_HOME}")
    print(f"  Train size    : {TRAIN_SIZE} days")
    print(f"  Test size     : {TEST_SIZE} days")
    print(f"  Epochs        : {EPOCHS}")

    validate_inputs(SUMO_HOME)

    split    = run_build_route()
    net_file = run_build_network(SUMO_HOME)

    reward_kwargs, _ = _load_component_config(
        REWARD_CONFIG_PATH, RewardClass.__name__, "Reward"
    )
    policy_kwargs_raw, _ = _load_component_config(
        POLICY_CONFIG_PATH, PolicyClass.__name__, "Policy"
    )
    policy_kwargs = _resolve_policy_params(policy_kwargs_raw)

    run_dir = _create_run_dir()
    shutil.copy2(REWARD_CONFIG_PATH, os.path.join(run_dir, "reward_configuration.json"))
    shutil.copy2(POLICY_CONFIG_PATH, os.path.join(run_dir, "policy_configuration.json"))
    _write_config_summary(run_dir, use_gui, reward_kwargs, policy_kwargs)

    trainer = build_pipeline(net_file, use_gui, run_dir, reward_kwargs, policy_kwargs)

    _banner("Training")
    results = trainer.run()

    _banner("Complete")
    print(f"  Train episodes : {len(results['train_log'])}")
    print(f"  Test episodes  : {len(results['test_log'])}")
    if results["test_log"]:
        mean_test = (
            sum(m["total_reward"] for m in results["test_log"])
            / len(results["test_log"])
        )
        print(f"  Mean test reward : {mean_test:.2f}")
    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
