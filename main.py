"""
main.py — single entry point for the full RL traffic signal optimisation pipeline.
All configuration lives at the top. Edit then run: python main.py [--gui]
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


# ==========================================================================
#  CONFIGURATION
# ==========================================================================

# File Paths
CSV_PATH          = "src/data/synthetic_toronto_data.csv"
INTERSECTION_PATH = "src/intersection.json"
COLUMNS_PATH      = "src/columns.json"
SUMO_HOME         = os.environ.get(
    "SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo"
)

# Data Split & Training
TRAIN_SIZE = 5
TEST_SIZE  = 5
EPOCHS     = 60

# Component Selection — swap any of these to use a different implementation
EnvironmentClass  = SumoEnvironment
ObservationClass  = QueueObservation
RewardClass       = CompositeReward
PolicyClass       = DoubleDQNPolicy
ReplayBufferClass = UniformReplayBuffer
SchedulerClass    = CosineScheduler      # set to None to disable LR scheduling

# Simulation
STEP_LENGTH  = 10.0      # seconds per SUMO step
SIM_BEGIN    = 28800      # 08:00
SIM_END      = 64800      # 18:00

# Phase Timing (seconds — Agent converts to sim steps internally)
MIN_GREEN_S      = 15
MAX_GREEN_S      = 90
OVERSHOOT_COEFF  = 4.0     # how harshly to penalize exceeding max_green (higher = harsher)

# Observation
MAX_LANES      = 16
MAX_PHASE      = 3
MAX_PHASE_TIME = 120.0
MAX_VEHICLES   = 20

# Reward
REWARD_NORMALISE = True
REWARD_SCALE     = 1.0
REWARD_ALPHA     = 0.65    # blend: 0.65 delta + 0.35 pressure

# Policy (DQN)
LEARNING_RATE  = 0.001     # starting LR (scheduler decays from here)
LR_MIN         = 0.0001    # floor LR the scheduler decays towards
GAMMA          = 0.99
EPSILON_START  = 1.0
EPSILON_END    = 0.05
TARGET_UPDATE  = 200
BATCH_SIZE     = 128
HIDDEN_SIZE    = 128
N_ACTIONS      = 2        # 0 = keep, 1 = switch

# Replay Buffer
BUFFER_CAPACITY = 50_000

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
_decisions_per_ep = _steps_per_ep / (_min_green_steps + 1)
TOTAL_UPDATES    = int(EPOCHS * TRAIN_SIZE * _decisions_per_ep)

# Dynamic epsilon decay — reaches EPSILON_END at ~85% of training
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1.0 / (0.85 * TOTAL_UPDATES))


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


def _create_run_dir() -> str:
    """Create a timestamped run folder inside LOGS_DIR and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{PolicyClass.__name__}_{RewardClass.__name__}"
    run_dir = os.path.join(LOGS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def _write_config_summary(run_dir: str, use_gui: bool) -> None:
    """Write a human-readable config.txt for this run."""
    sched_name = SchedulerClass.__name__ if SchedulerClass else "None (constant LR)"
    lines = [
        f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Components ---",
        f"Policy:        {PolicyClass.__name__}",
        f"Reward:        {RewardClass.__name__}",
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
        f"Max vehicles:  {MAX_VEHICLES}",
        "",
        "--- Reward ---",
        f"Normalise:     {REWARD_NORMALISE}",
        f"Scale:         {REWARD_SCALE}",
        f"Alpha:         {REWARD_ALPHA}",
        "",
        "--- Policy ---",
        f"Learning rate: {LEARNING_RATE} -> {LR_MIN} ({sched_name})",
        f"Total updates: ~{TOTAL_UPDATES:,} (estimated)",
        f"Gamma:         {GAMMA}",
        f"Epsilon:       {EPSILON_START} -> {EPSILON_END} (decay {EPSILON_DECAY})",
        f"Target update: {TARGET_UPDATE}",
        f"Batch size:    {BATCH_SIZE}",
        f"Hidden size:   {HIDDEN_SIZE}",
        f"N actions:     {N_ACTIONS}",
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


def build_pipeline(net_file: str, use_gui: bool, run_dir: str) -> Trainer:
    """Construct all RL components, wire them into Agent and Trainer."""
    _banner("Step 3/3 — Constructing RL components")

    with open(INTERSECTION_PATH) as f:
        int_cfg = json.load(f)

    obs_builder = ObservationClass(
        max_lanes      = MAX_LANES,
        max_phase      = MAX_PHASE,
        max_phase_time = MAX_PHASE_TIME,
        max_vehicles   = MAX_VEHICLES,
    )

    environment = EnvironmentClass(
        net_file     = net_file,
        step_length  = STEP_LENGTH,
        gui          = use_gui,
        sumo_home    = SUMO_HOME,
        begin        = SIM_BEGIN,
        end          = SIM_END,
    )

    reward = RewardClass(
        normalise = REWARD_NORMALISE,
        scale     = REWARD_SCALE,
        alpha     = REWARD_ALPHA,
    )

    policy = PolicyClass(
        obs_dim         = obs_builder.size(),
        n_actions       = N_ACTIONS,
        lr              = LEARNING_RATE,
        gamma           = GAMMA,
        epsilon_start   = EPSILON_START,
        epsilon_end     = EPSILON_END,
        epsilon_decay   = EPSILON_DECAY,
        target_update   = TARGET_UPDATE,
        batch_size      = BATCH_SIZE,
        hidden          = HIDDEN_SIZE,
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
    print(f"  Reward        : {RewardClass.__name__}")
    print(f"  Policy        : {PolicyClass.__name__} (device={policy.device})")
    print(f"  Scheduler     : {sched_name} (LR {LEARNING_RATE} -> {LR_MIN} over ~{TOTAL_UPDATES:,} steps)")
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

    run_dir = _create_run_dir()
    _write_config_summary(run_dir, use_gui)

    trainer = build_pipeline(net_file, use_gui, run_dir)

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
