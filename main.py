"""
main.py — single entry point for the full RL traffic signal optimisation pipeline.
Loads config.json (repo root) by default: paths, training, simulation, observation,
replay_buffer, output, components (classes), optional embedded reward_configuration /
policy_configuration objects, and optional top-level policy / reward overrides.
Per-class kwargs merge default + class block from embedded config (if non-empty) else from
optional paths.reward_configuration / paths.policy_configuration files; then config.json policy / reward
override (learning_rate maps to lr). Keys starting with _ and the reward key parameter_help
are documentation-only and are not passed to constructors.
Run: python main.py [--gui] [--config path/to/config.json]
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


def _abort(msg: str) -> None:
    print(f"\nERROR: {msg}")
    sys.exit(1)


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
from modelling.components.reward.vehicle_count import VehicleCountReward
from modelling.components.policy.dqn                    import DQNPolicy
from modelling.components.policy.double_dqn             import DoubleDQNPolicy
from modelling.components.replay_buffer.uniform         import UniformReplayBuffer
from modelling.components.scheduler.cosine              import CosineScheduler
from modelling.agent   import Agent
from modelling.trainer import Trainer
from KPIS import aggregate_test_kpis, default_episode_kpis, write_results_csv
from visualization.visualize_results import render_run_graphs
from modelling.components.reward.delta_vehicle_count import DeltaVehicleCountReward
from modelling.components.reward.composite_reward import CompositeReward
from modelling.components.reward.throughput import ThroughputReward
from modelling.components.reward.throughput_queue import ThroughputQueueReward
from modelling.components.reward.throughput_composite import ThroughputCompositeReward
from modelling.components.reward.throughput_wait_time import ThroughputWaitTimeReward
from modelling.components.reward.waiting_time import WaitingTimeReward
from modelling.components.reward.delta_waiting_time import DeltaWaitingTimeReward


# ==========================================================================
#  CONFIGURATION (from config.json — see _apply_config)
# ==========================================================================

DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config.json")

# Infrastructure component classes — defaults; overridden by config.json → components.*
EnvironmentClass  = SumoEnvironment
ObservationClass  = QueueObservation
ReplayBufferClass = UniformReplayBuffer
EvalRewardClass: type | None = None  # optional; None = same reward for train and eval

# config.json ``policy`` / ``reward`` override merged kwargs from *configuration.json (file first, then these).
POLICY_KWARGS_OVERRIDES: dict[str, Any] = {}
REWARD_KWARGS_OVERRIDES: dict[str, Any] = {}

# Reward / policy classes: defaults below; overridden by config.json → components.*
RewardClass: type = ThroughputQueueReward
PolicyClass: type = DoubleDQNPolicy
SchedulerClass: type | None = None

# Paths and scalars — filled only by _apply_config(); numeric defaults are the .get(..., default) values inside it.
CSV_PATH = ""
INTERSECTION_PATH = ""
COLUMNS_PATH = ""
REWARD_CONFIG_PATH = ""
POLICY_CONFIG_PATH = ""
# Full reward/policy configuration documents from config.json (non-empty dict → use instead of paths).
REWARD_CONFIG_DOCUMENT: dict[str, Any] | None = None
POLICY_CONFIG_DOCUMENT: dict[str, Any] | None = None
SUMO_HOME = ""
TRAIN_SIZE = 0
TEST_SIZE = 0
EPOCHS = 0
STEP_LENGTH = 0.0
DECISION_GAP = 0
SIM_BEGIN = 0
SIM_END = 0
SIMULATION_GUI = False
MIN_GREEN_S = 0
MAX_GREEN_S = 0
OVERSHOOT_COEFF = 0.0
YELLOW_DURATION_S = 0.0
MAX_LANES = 0
MAX_PHASE = 0
MAX_PHASE_TIME = 0.0
MAX_VEHICLES = 0
LR_MIN = 0.0
BUFFER_CAPACITY = 0
SEED = 0
SAVE_EVERY = 0
LOG_EVERY = 0
OUT_DIR = ""
RESULTS_DIR = ""
TOTAL_UPDATES = 0
DECISIONS_PER_EP_DIVISOR = 0.0
FLOW_DEMAND_SUBSLOTS = 5
FLOW_DEMAND_SPREAD = 0.85
FLOW_DEMAND_VARIATION_SEED = 42


def _resolve_path(rel_or_abs: str) -> str:
    if not rel_or_abs:
        return rel_or_abs
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(ROOT, rel_or_abs.replace("/", os.sep)))


def _load_config_data(path: str | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(cfg_path):
        _abort(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_policy_overrides(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Map config.json ``policy`` keys to policy kwargs (``learning_rate`` → ``lr``)."""
    if not raw:
        return {}
    out: dict[str, Any] = dict(raw)
    if "learning_rate" in out:
        out["lr"] = out.pop("learning_rate")
    return {k: v for k, v in out.items() if v is not None}


def merge_reward_kwargs_from_config(file_merged: dict[str, Any]) -> dict[str, Any]:
    """Apply root ``config.json`` ``reward`` overrides on top of reward_configuration merge."""
    return {**file_merged, **REWARD_KWARGS_OVERRIDES}


def merge_policy_kwargs_from_config(file_merged: dict[str, Any]) -> dict[str, Any]:
    """Apply root ``config.json`` ``policy`` overrides on top of policy_configuration merge."""
    return {**file_merged, **POLICY_KWARGS_OVERRIDES}


def _apply_config(cfg: dict[str, Any]) -> None:
    """Set module-level settings from config.json (also when gridsearch imports main)."""
    global CSV_PATH, INTERSECTION_PATH, COLUMNS_PATH, REWARD_CONFIG_PATH, POLICY_CONFIG_PATH
    global REWARD_CONFIG_DOCUMENT, POLICY_CONFIG_DOCUMENT
    global SUMO_HOME, TRAIN_SIZE, TEST_SIZE, EPOCHS, STEP_LENGTH, DECISION_GAP
    global SIM_BEGIN, SIM_END, SIMULATION_GUI, MIN_GREEN_S, MAX_GREEN_S, OVERSHOOT_COEFF, YELLOW_DURATION_S
    global MAX_LANES, MAX_PHASE, MAX_PHASE_TIME, MAX_VEHICLES, LR_MIN, BUFFER_CAPACITY
    global SEED, SAVE_EVERY, LOG_EVERY, OUT_DIR, RESULTS_DIR, TOTAL_UPDATES, SchedulerClass
    global DECISIONS_PER_EP_DIVISOR
    global FLOW_DEMAND_SUBSLOTS, FLOW_DEMAND_SPREAD, FLOW_DEMAND_VARIATION_SEED
    global RewardClass, PolicyClass
    global EnvironmentClass, ObservationClass, ReplayBufferClass, EvalRewardClass
    global POLICY_KWARGS_OVERRIDES, REWARD_KWARGS_OVERRIDES

    _reward_by_name: dict[str, type] = {
        "CompositeReward": CompositeReward,
        "DeltaVehicleCountReward": DeltaVehicleCountReward,
        "DeltaWaitingTimeReward": DeltaWaitingTimeReward,
        "DeltaWaitTimeReward": DeltaWaitingTimeReward,
        "ThroughputQueueReward": ThroughputQueueReward,
        "ThroughputCompositeReward": ThroughputCompositeReward,
        "ThroughputWaitTimeReward": ThroughputWaitTimeReward,
        "ThroughputWaitTime": ThroughputWaitTimeReward,
        "ThroughputReward": ThroughputReward,
        "VehicleCountReward": VehicleCountReward,
        "WaitingTimeReward": WaitingTimeReward,
        "WaitTimeReward": WaitingTimeReward,
    }
    _policy_by_name: dict[str, type] = {
        "DQNPolicy": DQNPolicy,
        "DoubleDQNPolicy": DoubleDQNPolicy,
    }
    _environment_by_name: dict[str, type] = {
        "SumoEnvironment": SumoEnvironment,
    }
    _observation_by_name: dict[str, type] = {
        "QueueObservation": QueueObservation,
    }
    _replay_by_name: dict[str, type] = {
        "UniformReplayBuffer": UniformReplayBuffer,
    }

    comp = cfg.get("components", {})
    _rn = comp.get("reward_class", "ThroughputQueueReward")
    _pn = comp.get("policy_class", "DoubleDQNPolicy")
    if _rn not in _reward_by_name:
        _abort(
            f"Unknown components.reward_class {_rn!r}. "
            f"Use one of: {', '.join(sorted(_reward_by_name))}"
        )
    if _pn not in _policy_by_name:
        _abort(
            f"Unknown components.policy_class {_pn!r}. "
            f"Use one of: {', '.join(sorted(_policy_by_name))}"
        )
    RewardClass = _reward_by_name[_rn]
    PolicyClass = _policy_by_name[_pn]

    _en = comp.get("environment_class", "SumoEnvironment")
    if _en not in _environment_by_name:
        _abort(
            f"Unknown components.environment_class {_en!r}. "
            f"Use one of: {', '.join(sorted(_environment_by_name))}"
        )
    EnvironmentClass = _environment_by_name[_en]

    _on = comp.get("observation_class", "QueueObservation")
    if _on not in _observation_by_name:
        _abort(
            f"Unknown components.observation_class {_on!r}. "
            f"Use one of: {', '.join(sorted(_observation_by_name))}"
        )
    ObservationClass = _observation_by_name[_on]

    _bn = comp.get("replay_buffer_class", "UniformReplayBuffer")
    if _bn not in _replay_by_name:
        _abort(
            f"Unknown components.replay_buffer_class {_bn!r}. "
            f"Use one of: {', '.join(sorted(_replay_by_name))}"
        )
    ReplayBufferClass = _replay_by_name[_bn]

    eval_raw = comp.get("eval_reward_class")
    if eval_raw is None or (
        isinstance(eval_raw, str) and eval_raw.strip().lower() in ("", "none", "null")
    ):
        EvalRewardClass = None
    elif isinstance(eval_raw, str):
        if eval_raw not in _reward_by_name:
            _abort(
                f"Unknown components.eval_reward_class {eval_raw!r}. "
                f"Use one of: {', '.join(sorted(_reward_by_name))}"
            )
        EvalRewardClass = _reward_by_name[eval_raw]
    else:
        _abort(f"components.eval_reward_class must be a string or null, got {eval_raw!r}")

    _policy_raw = cfg.get("policy") or {}
    POLICY_KWARGS_OVERRIDES = _normalize_policy_overrides({
        k: v
        for k, v in _policy_raw.items()
        if not (isinstance(k, str) and k.startswith("_"))
    })
    _reward_raw = cfg.get("reward") or {}
    REWARD_KWARGS_OVERRIDES = {
        k: v
        for k, v in _reward_raw.items()
        if v is not None and not (isinstance(k, str) and k.startswith("_"))
    }

    _rc_doc = cfg.get("reward_configuration")
    REWARD_CONFIG_DOCUMENT = _rc_doc if isinstance(_rc_doc, dict) and _rc_doc else None
    _pc_doc = cfg.get("policy_configuration")
    POLICY_CONFIG_DOCUMENT = _pc_doc if isinstance(_pc_doc, dict) and _pc_doc else None

    paths = cfg.get("paths", {})
    CSV_PATH = _resolve_path(paths.get("csv", "src/data/synthetic_toronto_data.csv"))
    INTERSECTION_PATH = _resolve_path(paths.get("intersection", "src/intersection.json"))
    COLUMNS_PATH = _resolve_path(paths.get("columns", "src/columns.json"))
    REWARD_CONFIG_PATH = _resolve_path(paths.get("reward_configuration", "") or "")
    POLICY_CONFIG_PATH = _resolve_path(paths.get("policy_configuration", "") or "")

    cfg_sumo = (paths.get("sumo_home") or "").strip()
    SUMO_HOME = os.environ.get("SUMO_HOME", "").strip() or cfg_sumo or _default_sumo_home()

    training = cfg.get("training", {})
    EPOCHS = int(training.get("n_epochs", 100))
    TRAIN_SIZE = int(training.get("train_days", training.get("train_size", 25)))
    TEST_SIZE = int(training.get("test_days", 5))
    SAVE_EVERY = int(training.get("save_every", 20))
    LOG_EVERY = int(training.get("log_every", 1))
    SEED = int(training.get("seed", 42))
    LR_MIN = float(training.get("lr_min", 0.0001))
    DECISIONS_PER_EP_DIVISOR = float(training.get("decisions_per_ep_divisor", 2))
    if DECISIONS_PER_EP_DIVISOR < 1e-9:
        DECISIONS_PER_EP_DIVISOR = 2.0

    _fd = cfg.get("flow_demand") or {}
    if not isinstance(_fd, dict):
        _fd = {}
    FLOW_DEMAND_SUBSLOTS = max(1, int(_fd.get("subslots_per_slot", 5)))
    FLOW_DEMAND_SPREAD = float(_fd.get("spread", 0.85))
    _fd_seed = _fd.get("seed")
    FLOW_DEMAND_VARIATION_SEED = int(_fd_seed) if _fd_seed is not None else SEED

    sched_raw = training.get("scheduler", "cosine")
    if sched_raw is None or (isinstance(sched_raw, str) and sched_raw.lower() in ("none", "")):
        SchedulerClass = None
    elif isinstance(sched_raw, str) and sched_raw.lower() == "cosine":
        SchedulerClass = CosineScheduler
    else:
        _abort(f"Unknown training.scheduler value: {sched_raw!r} (use 'none' or 'cosine')")

    sim = cfg.get("simulation", {})
    STEP_LENGTH = float(sim.get("step_length", 1.0))
    DECISION_GAP = int(sim.get("decision_gap", 10))
    SIM_BEGIN = int(sim.get("begin", 28800))
    SIM_END = int(sim.get("end", 64800))
    SIMULATION_GUI = bool(sim.get("gui", False))

    phase = cfg.get("phase_timing", {})
    MIN_GREEN_S = int(phase.get("min_green_s", 15))
    MAX_GREEN_S = int(phase.get("max_green_s", 90))
    OVERSHOOT_COEFF = float(phase.get("overshoot_coeff", 1.0))
    YELLOW_DURATION_S = float(phase.get("yellow_duration_s", 4.0))

    obs = cfg.get("observation", {})
    MAX_LANES = int(obs.get("max_lanes", 16))
    MAX_PHASE = int(obs.get("max_phase", 7))
    MAX_PHASE_TIME = float(obs.get("max_phase_time", 120.0))
    MAX_VEHICLES = int(obs.get("max_vehicles", 20))

    replay = cfg.get("replay_buffer", {})
    BUFFER_CAPACITY = int(replay.get("capacity", 32768))

    out = cfg.get("output", {})
    OUT_DIR = _resolve_path(out.get("out_dir", "src/data"))
    _results = out.get("results_dir") or out.get("logs_dir", "src/data/results")
    RESULTS_DIR = _resolve_path(_results)

    _sim_seconds = SIM_END - SIM_BEGIN
    _steps_per_ep = _sim_seconds / STEP_LENGTH
    _decisions_per_ep = _steps_per_ep / DECISIONS_PER_EP_DIVISOR
    TOTAL_UPDATES = int(EPOCHS * TRAIN_SIZE * _decisions_per_ep)
    if TOTAL_UPDATES < 1:
        TOTAL_UPDATES = 1


if os.path.isfile(DEFAULT_CONFIG_PATH):
    _apply_config(_load_config_data(DEFAULT_CONFIG_PATH))
else:
    _apply_config({})

# ==========================================================================
#  HELPERS
# ==========================================================================

def _load_module(name: str, filepath: str):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_file(path: str, label: str):
    if not os.path.exists(path):
        _abort(f"{label} not found at: {path}")


def _banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def _strip_meta_kwargs_block(block: Any) -> dict[str, Any]:
    """Remove documentation keys from a default or per-class block (``_*``, ``parameter_help``)."""
    if not isinstance(block, dict):
        return {}
    return {
        k: v
        for k, v in block.items()
        if not (isinstance(k, str) and (k.startswith("_") or k == "parameter_help"))
    }


def _load_component_config(path: str, class_name: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge default + class-specific blocks from embedded config (if set) or from JSON file."""
    if label == "Policy":
        all_cfg = POLICY_CONFIG_DOCUMENT
    else:
        all_cfg = REWARD_CONFIG_DOCUMENT  # "Reward", "Eval reward", etc.

    if all_cfg is None:
        if not os.path.exists(path):
            _abort(f"{label} config not found: {path}")
        with open(path, encoding="utf-8") as f:
            all_cfg = json.load(f)

    default_raw = all_cfg.get("default", {}) or {}
    class_raw = all_cfg.get(class_name, {}) or {}
    if not isinstance(default_raw, dict) or not isinstance(class_raw, dict):
        _abort(f"{label} config must map 'default' and class names to objects")
    default_cfg = _strip_meta_kwargs_block(default_raw)
    class_cfg = _strip_meta_kwargs_block(class_raw)
    merged = {**default_cfg, **class_cfg}
    return merged, all_cfg


def _snapshot_component_file_body(doc: dict[str, Any]) -> dict[str, Any]:
    """Strip documentation-only keys for reward_configuration.json / policy_configuration.json copies."""
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if k in ("_about", "parameter_help") or (isinstance(k, str) and k.startswith("_")):
            continue
        if isinstance(v, dict):
            out[k] = _strip_meta_kwargs_block(v)
        else:
            out[k] = v
    return out


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
    """Create a timestamped run folder inside RESULTS_DIR and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{PolicyClass.__name__}_{RewardClass.__name__}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def _write_config_summary(
    run_dir: str,
    use_gui: bool,
    reward_kwargs: dict[str, Any],
    policy_kwargs: dict[str, Any],
) -> None:
    """Write a human-readable config.txt for this run."""
    sched_name = (
        SchedulerClass.__name__ if SchedulerClass is not None else "None (constant LR)"
    )
    reward_cfg_rel = (
        "config.json → reward_configuration"
        if REWARD_CONFIG_DOCUMENT is not None
        else os.path.relpath(REWARD_CONFIG_PATH, ROOT)
    )
    policy_cfg_rel = (
        "config.json → policy_configuration"
        if POLICY_CONFIG_DOCUMENT is not None
        else os.path.relpath(POLICY_CONFIG_PATH, ROOT)
    )
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
        f"Decision gap:  {DECISION_GAP} (SUMO steps)",
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
        f"Total updates: ~{TOTAL_UPDATES:,} (estimated; divisor={DECISIONS_PER_EP_DIVISOR})",
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
    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==========================================================================
#  PIPELINE STEPS
# ==========================================================================

def validate_inputs(sumo_home: str):
    """Make sure all required files and SUMO are present."""
    _check_file(CSV_PATH,          "CSV data file")
    _check_file(INTERSECTION_PATH, "intersection.json")
    _check_file(COLUMNS_PATH,      "columns.json")
    if REWARD_CONFIG_DOCUMENT is None:
        _check_file(REWARD_CONFIG_PATH, "reward_configuration.json")
    if POLICY_CONFIG_DOCUMENT is None:
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
            "paths.sumo_home in config.json."
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
    br.write_sumo_flows(
        df_used,
        active,
        slot_minutes,
        flows_dir,
        int_cfg,
        subslots_per_slot=FLOW_DEMAND_SUBSLOTS,
        spread=FLOW_DEMAND_SPREAD,
        variation_seed=FLOW_DEMAND_VARIATION_SEED,
    )

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
        net_file      = net_file,
        step_length   = STEP_LENGTH,
        decision_gap  = DECISION_GAP,
        gui           = use_gui,
        sumo_home     = SUMO_HOME,
        begin         = SIM_BEGIN,
        end           = SIM_END,
    )

    reward = RewardClass(**reward_kwargs)
    eval_reward = None
    if EvalRewardClass is not None:
        eval_kw, _ = _load_component_config(
            REWARD_CONFIG_PATH, EvalRewardClass.__name__, "Eval reward"
        )
        eval_kw = merge_reward_kwargs_from_config(eval_kw)
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

    sched_name = SchedulerClass.__name__ if SchedulerClass is not None else "None"
    print(f"  Environment   : {EnvironmentClass.__name__} (gui={use_gui})")
    print(f"  Observation   : {ObservationClass.__name__} (size={obs_builder.size()})")
    eval_name = EvalRewardClass.__name__ if EvalRewardClass else "same as train"
    print(f"  Reward        : {RewardClass.__name__} (train) / {eval_name} (eval)")
    print(f"  Policy        : {PolicyClass.__name__} (device={policy.device})")
    print(f"  Scheduler     : {sched_name} (LR {policy_kwargs['lr']} -> {LR_MIN} over ~{TOTAL_UPDATES:,} steps)")
    print(f"  Replay buffer : {ReplayBufferClass.__name__} (capacity={BUFFER_CAPACITY:,})")
    print(
        f"  Phase timing  : min={MIN_GREEN_S}s  max={MAX_GREEN_S}s  "
        f"yellow={YELLOW_DURATION_S}s  (decide every {STEP_LENGTH}s after min)"
    )
    print(f"  Run folder    : {run_dir}")

    agent = Agent(
        environment     = environment,
        observation     = obs_builder,
        reward          = reward,
        policy          = policy,
        replay_buffer   = replay_buffer,
        scheduler       = scheduler,
        eval_reward     = eval_reward,
        step_length       = STEP_LENGTH,
        min_green_s       = MIN_GREEN_S,
        max_green_s       = MAX_GREEN_S,
        overshoot_coeff   = OVERSHOOT_COEFF,
        yellow_duration_s = YELLOW_DURATION_S,
        episode_kpis      = default_episode_kpis(),
    )

    split_path = os.path.join(OUT_DIR, "processed", "split.json")
    flows_dir  = os.path.join(OUT_DIR, "sumo", "flows")
    days_dir   = os.path.join(OUT_DIR, "processed", "days")

    trainer = Trainer(
        agent      = agent,
        split_path = split_path,
        flows_dir  = flows_dir,
        output_dir = run_dir,
        n_epochs   = EPOCHS,
        save_every = SAVE_EVERY,
        log_every  = LOG_EVERY,
        days_dir   = days_dir,
    )

    return trainer


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="RL Traffic Signal Optimiser")
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--gui", action="store_true", help="Launch sumo-gui")
    args = parser.parse_args()

    if args.config:
        _apply_config(_load_config_data(args.config))

    use_gui = bool(args.gui or SIMULATION_GUI)

    _banner("RL Traffic Signal Optimiser")
    cfg_display = os.path.abspath(args.config) if args.config else os.path.abspath(DEFAULT_CONFIG_PATH)
    print(f"  Config file   : {cfg_display}")
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
    reward_kwargs = merge_reward_kwargs_from_config(reward_kwargs)
    policy_kwargs_raw, _ = _load_component_config(
        POLICY_CONFIG_PATH, PolicyClass.__name__, "Policy"
    )
    policy_kwargs_raw = merge_policy_kwargs_from_config(policy_kwargs_raw)
    policy_kwargs = _resolve_policy_params(policy_kwargs_raw)

    run_dir = _create_run_dir()
    _r_dest = os.path.join(run_dir, "reward_configuration.json")
    _p_dest = os.path.join(run_dir, "policy_configuration.json")
    if REWARD_CONFIG_DOCUMENT is not None:
        with open(_r_dest, "w", encoding="utf-8") as f:
            json.dump(_snapshot_component_file_body(REWARD_CONFIG_DOCUMENT), f, indent=2)
            f.write("\n")
    else:
        shutil.copy2(REWARD_CONFIG_PATH, _r_dest)
    if POLICY_CONFIG_DOCUMENT is not None:
        with open(_p_dest, "w", encoding="utf-8") as f:
            json.dump(_snapshot_component_file_body(POLICY_CONFIG_DOCUMENT), f, indent=2)
            f.write("\n")
    else:
        shutil.copy2(POLICY_CONFIG_PATH, _p_dest)
    _write_config_summary(run_dir, use_gui, reward_kwargs, policy_kwargs)

    trainer = build_pipeline(net_file, use_gui, run_dir, reward_kwargs, policy_kwargs)

    _banner("Training")
    results = trainer.run()

    _sim_elapsed = float(SIM_END - SIM_BEGIN)
    _kpi_agg = aggregate_test_kpis(results["test_log"], _sim_elapsed)
    _results_row = {
        "run_name": os.path.basename(run_dir),
        "reward_class": RewardClass.__name__,
        "policy_class": PolicyClass.__name__,
        **_kpi_agg,
    }
    _results_fields = [
        "run_name",
        "reward_class",
        "policy_class",
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
    write_results_csv(
        os.path.join(run_dir, "results.csv"),
        _results_row,
        fieldnames=_results_fields,
    )

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
    try:
        graphs_path = render_run_graphs(run_dir)
        print(f"Graphs saved to: {graphs_path}")
    except Exception as exc:
        print(f"Warning: could not write graphs: {exc}")


if __name__ == "__main__":
    main()
