"""
RL Traffic Signal Optimizer — Streamlit UI
------------------------------------------
Simplest stack: pure Python + Streamlit (https://docs.streamlit.io).

Run from the project root:
    streamlit run streamlit_app.py

Requires: SUMO installed, SUMO_HOME set (or paste path in the sidebar).
"""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from KPIS import aggregate_test_kpis, write_results_csv

from streamlit_config_forms import invalidate_config_widget_sync, render_column_and_intersection_forms
from streamlit_intersection_helpers import (
    build_intersection_dict,
    detect_approaches_from_headers,
    suggest_columns_json_from_dataframe,
    suggest_intersection_name,
)

# Project root = directory containing this file
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = ROOT / "streamlit_runs"
RUNS.mkdir(parents=True, exist_ok=True)

# Example grid-search trial for the "Apply trial hyperparameters" box (report / screenshots).
_EXAMPLE_GRID_TRIAL = (
    "experiments/grid_search/runs/2026-04-02_23-58-59/trial_0003_DoubleDQNPolicy_CompositeReward"
)

# Scratch CSV for the UI only (never `src/data/...`). Removed at the start of each new browser session.
_STREAMLIT_TRAFFIC_CSV = RUNS / "traffic.csv"


def _discard_previous_session_traffic_csv() -> None:
    """First run of a Streamlit session: delete stale `traffic.csv` so nothing loads until user uploads."""
    if "_streamlit_traffic_session_started" in st.session_state:
        return
    st.session_state._streamlit_traffic_session_started = True
    try:
        _STREAMLIT_TRAFFIC_CSV.unlink(missing_ok=True)
    except OSError:
        pass


# UI default for EPOCHS (quick local tests). Change to 60 to match main.py, or set in Advanced.
STREAMLIT_DEFAULT_EPOCHS = 10

def _apply_generated_config_from_dataframe(df: pd.DataFrame) -> None:
    """Fill columns_text + intersection_text from CSV headers and defaults (2/1/1 lanes, 50 km/h)."""
    if df is None or df.empty:
        return

    active = detect_approaches_from_headers(df)
    st.session_state.pop("_show_nswe_warning_once", None)
    if not active:
        active = ["N", "S", "E", "W"]
        st.session_state["_show_nswe_warning_once"] = True

    col_obj = suggest_columns_json_from_dataframe(df, active)
    int_obj = build_intersection_dict(
        suggest_intersection_name(df),
        active,
        lanes_through=2,
        lanes_right=1,
        lanes_left=1,
        speed_kmh=50.0,
    )
    st.session_state.columns_text = json.dumps(col_obj, indent=2)
    st.session_state.intersection_text = json.dumps(int_obj, indent=2)


def _recompute_derived(m) -> None:
    """Recompute TOTAL_UPDATES after the user changes timing/training constants."""
    sim_seconds = m.SIM_END - m.SIM_BEGIN
    steps_per_ep = sim_seconds / m.STEP_LENGTH
    divisor = getattr(m, "DECISIONS_PER_EP_DIVISOR", 2.0) or 2.0
    decisions_per_ep = steps_per_ep / divisor
    m.TOTAL_UPDATES = max(1, int(m.EPOCHS * m.TRAIN_SIZE * decisions_per_ep))


def _sumo_bin_on_path(sumo_home: str) -> None:
    if sumo_home and os.path.isdir(os.path.join(sumo_home, "bin")):
        bin_dir = os.path.join(sumo_home, "bin")
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["SUMO_HOME"] = sumo_home


def _try_eclipse_sumo_home() -> str:
    try:
        import sumo  # noqa: PLC0415

        return sumo.SUMO_HOME
    except Exception:
        return ""


def _write_eval_config(m, data_out: Path, models_dir: Path, path: Path) -> None:
    """Config for evaluate_baseline.py — reward class and all params match main.py."""
    reward_kwargs, _ = m._load_component_config(
        m.REWARD_CONFIG_PATH, m.RewardClass.__name__, "Reward"
    )
    reward_kwargs = m.merge_reward_kwargs_from_config(reward_kwargs)
    policy_kwargs_raw, _ = m._load_component_config(
        m.POLICY_CONFIG_PATH, m.PolicyClass.__name__, "Policy"
    )
    policy_kwargs_raw = m.merge_policy_kwargs_from_config(policy_kwargs_raw)
    policy_kwargs = m._resolve_policy_params(policy_kwargs_raw)

    cfg = {
        "training": {
            "n_epochs": 0,
            "save_every": 10,
            "log_every": 1,
            "test_days": m.TEST_SIZE,
            "seed": m.SEED,
        },
        "simulation": {
            "step_length": float(m.STEP_LENGTH),
            "decision_gap": int(getattr(m, "DECISION_GAP", 10)),
            "begin": int(m.SIM_BEGIN),
            "end": int(m.SIM_END),
            "gui": False,
        },
        "observation": {
            "max_lanes": m.MAX_LANES,
            "max_phase": m.MAX_PHASE,
            "max_phase_time": float(m.MAX_PHASE_TIME),
            "max_vehicles": m.MAX_VEHICLES,
        },
        "reward": {
            "class": m.RewardClass.__name__,
            **reward_kwargs,
        },
        "policy": {
            "learning_rate": float(policy_kwargs["lr"]),
            "gamma": float(policy_kwargs["gamma"]),
            "epsilon_start": float(policy_kwargs["epsilon_start"]),
            "epsilon_end": float(policy_kwargs["epsilon_end"]),
            "epsilon_decay": float(policy_kwargs["epsilon_decay"]),
            "target_update": int(policy_kwargs["target_update"]),
            "batch_size": int(policy_kwargs["batch_size"]),
            "hidden": int(policy_kwargs["hidden"]),
        },
        "replay_buffer": {"capacity": int(m.BUFFER_CAPACITY)},
        "output": {
            "out_dir": str(data_out).replace("\\", "/"),
            "models_dir": str(models_dir).replace("\\", "/"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _reload_main():
    """Fresh `main` module so RewardClass / PolicyClass match repo defaults + config.json."""
    import main

    importlib.reload(main)
    return main


def _apply_grid_search_trial(m, trial_dir: Path) -> str:
    """
    Point reward/policy JSON and component classes at a grid-search trial folder
    (same format as experiments/grid_search/runs/.../trial_*/).
    Matches grid-search training: no LR scheduler (SchedulerClass = None).
    """
    trial_dir = trial_dir.expanduser().resolve()
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {trial_dir}")
    src_r = trial_dir / "reward_configuration.json"
    src_p = trial_dir / "policy_configuration.json"
    if not src_r.is_file() or not src_p.is_file():
        raise FileNotFoundError(
            f"Trial folder must contain reward_configuration.json and policy_configuration.json: {trial_dir}"
        )

    dst_r = RUNS / "overlay_reward_configuration.json"
    dst_p = RUNS / "overlay_policy_configuration.json"
    shutil.copy2(src_r, dst_r)
    shutil.copy2(src_p, dst_p)

    with open(dst_r, encoding="utf-8") as f:
        rj = json.load(f)
    with open(dst_p, encoding="utf-8") as f:
        pj = json.load(f)

    reward_name = next(
        (k for k in rj if k != "default" and isinstance(rj.get(k), dict) and rj[k]),
        None,
    )
    policy_name = next(
        (k for k in pj if k != "default" and isinstance(pj.get(k), dict) and pj[k]),
        None,
    )
    if not reward_name or not policy_name:
        raise ValueError("Could not read reward/policy class names from trial JSON files.")

    reward_map = {
        "CompositeReward": m.CompositeReward,
        "ThroughputQueueReward": m.ThroughputQueueReward,
        "ThroughputReward": m.ThroughputReward,
        "ThroughputCompositeReward": m.ThroughputCompositeReward,
        "ThroughputWaitTimeReward": m.ThroughputWaitTimeReward,
        "DeltaVehicleCountReward": m.DeltaVehicleCountReward,
        "VehicleCountReward": m.VehicleCountReward,
        "DeltaWaitingTimeReward": m.DeltaWaitingTimeReward,
        "WaitingTimeReward": m.WaitingTimeReward,
    }
    policy_map = {
        "DoubleDQNPolicy": m.DoubleDQNPolicy,
        "DQNPolicy": m.DQNPolicy,
    }
    if reward_name not in reward_map:
        raise ValueError(f"Unknown reward class in trial: {reward_name}")
    if policy_name not in policy_map:
        raise ValueError(f"Unknown policy class in trial: {policy_name}")

    m.RewardClass = reward_map[reward_name]
    m.PolicyClass = policy_map[policy_name]
    m.REWARD_CONFIG_PATH = str(dst_r.resolve())
    m.POLICY_CONFIG_PATH = str(dst_p.resolve())
    m.SchedulerClass = None  # align with experiments/grid_search/gridsearch.py
    return f"{reward_name} + {policy_name} (configs copied to streamlit_runs/overlay_*.json)"


def _consistency_errors(intersection: dict, col_map: dict) -> list[str]:
    """Each active approach must appear in column map approaches."""
    err: list[str] = []
    active = intersection.get("active_approaches") or []
    approaches = (col_map or {}).get("approaches") or {}
    for d in active:
        if d not in approaches:
            err.append(
                f"Intersection lists approach {d} but columns.json has no 'approaches.{d}' block."
            )
    return err


def main() -> None:
    st.set_page_config(
        page_title="RL Traffic Optimizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("RL Traffic Signal Optimizer")
    st.caption(
        "Upload a traffic CSV each visit (nothing is loaded from `src/data/`). Then **Generate from CSV**, "
        "adjust the column and intersection fields below, and run training (same code paths as `main.py`)."
    )

    m = _reload_main()
    _discard_previous_session_traffic_csv()

    st.session_state.setdefault("grid_search_trial_dir", None)
    if st.session_state.get("grid_search_trial_dir"):
        try:
            _apply_grid_search_trial(m, Path(st.session_state["grid_search_trial_dir"]))
        except Exception as e:
            st.warning(f"Could not apply grid-search trial overlay (cleared): {e}")
            st.session_state["grid_search_trial_dir"] = None

    if "adv_train" not in st.session_state:
        st.session_state.adv_train = int(m.TRAIN_SIZE)
    if "adv_test" not in st.session_state:
        st.session_state.adv_test = int(m.TEST_SIZE)
    if "adv_epochs" not in st.session_state:
        st.session_state.adv_epochs = STREAMLIT_DEFAULT_EPOCHS

    # Empty until **Generate from CSV** (or paste). Streamlit keeps this across reruns in the same tab.
    st.session_state.setdefault("intersection_text", "")
    st.session_state.setdefault("columns_text", "")
    st.session_state.setdefault("_had_config_once", False)

    with st.expander("Grid-search trial hyperparameters (optional)", expanded=False):
        st.caption(
            "Point training at a **trial_* folder** from `experiments/grid_search/runs/...` "
            "(must contain `reward_configuration.json` and `policy_configuration.json`). "
            "This matches **gridsearch.py**: same reward/policy classes, JSON kwargs, and **no LR scheduler**. "
            "**Apply** also sets TRAIN_SIZE, TEST_SIZE, and EPOCHS from the current `config.json` "
            "(the same source grid search used when you ran the sweep)."
        )
        if "trial_dir_text" not in st.session_state:
            st.session_state.trial_dir_text = _EXAMPLE_GRID_TRIAL
        st.text_input(
            "Trial folder (relative to project root or absolute)",
            key="trial_dir_text",
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Apply trial hyperparameters", type="primary", key="btn_apply_trial"):
                raw = (st.session_state.get("trial_dir_text") or "").strip()
                p = Path(raw)
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                if not p.is_dir():
                    st.error(f"Directory not found: {p}")
                else:
                    st.session_state["grid_search_trial_dir"] = str(p)
                    st.session_state.adv_train = int(m.TRAIN_SIZE)
                    st.session_state.adv_test = int(m.TEST_SIZE)
                    st.session_state.adv_epochs = int(m.EPOCHS)
                    st.rerun()
        with b2:
            if st.button("Clear trial overlay", key="btn_clear_trial"):
                st.session_state["grid_search_trial_dir"] = None
                st.rerun()
        if st.session_state.get("grid_search_trial_dir"):
            st.success(
                f"Trial overlay **active**: `{st.session_state['grid_search_trial_dir']}` — "
                f"reward/policy JSON copied to `streamlit_runs/overlay_*.json`."
            )

    # --- Sidebar: SUMO ---
    st.sidebar.header("Environment")
    sumo_default = os.environ.get("SUMO_HOME", "") or _try_eclipse_sumo_home()
    sumo_home = st.sidebar.text_input(
        "SUMO_HOME",
        value=sumo_default,
        help="Folder that contains bin/sumo (e.g. eclipse-sumo pip install path).",
    )
    st.sidebar.caption(
        f"Default EPOCHS in the UI is **{STREAMLIT_DEFAULT_EPOCHS}** (see `STREAMLIT_DEFAULT_EPOCHS` "
        f"in `streamlit_app.py`). Change there or use Advanced below."
    )

    # --- Main: uploads & JSON ---
    st.subheader("1. Traffic CSV")
    csv_file = st.file_uploader(
        "Upload counts CSV (required — example format: `src/data/synthetic_toronto_data.csv` in the repo)",
        type=["csv"],
    )
    csv_path = _STREAMLIT_TRAFFIC_CSV
    df_preview: pd.DataFrame | None = None
    if csv_file is not None:
        csv_path.write_bytes(csv_file.getvalue())
        df_preview = pd.read_csv(csv_file)
        st.dataframe(df_preview.head(8), use_container_width=True)
    elif csv_path.exists():
        # Same session only: file appears after user uploads; we never load a leftover from a past visit.
        df_preview = pd.read_csv(csv_path)

    g1, g2, g3 = st.columns([1, 2, 1])
    with g1:
        gen_from_csv = st.button(
            "Generate from CSV",
            type="primary",
            disabled=df_preview is None,
            help="Overwrite column map + intersection JSON from the loaded CSV (2/1/1 lanes, 50 km/h). "
            "Does not run until you click this.",
            key="btn_generate_from_csv",
        )
    with g2:
        st.caption(
            "Builds **columns.json** and **intersection.json** from the table above. "
            "Re-click after changing the CSV or to reset from headers."
        )
    with g3:
        if st.button("Clear JSON", help="Empty stored column map + intersection JSON (e.g. stale session data)."):
            st.session_state.columns_text = ""
            st.session_state.intersection_text = ""
            st.session_state["_had_config_once"] = False
            invalidate_config_widget_sync()
            st.rerun()

    if gen_from_csv:
        if df_preview is None:
            st.error("Load a CSV first (upload required each new browser session).")
        else:
            _apply_generated_config_from_dataframe(df_preview)
            st.session_state["_had_config_once"] = True
            st.success("Updated **columns.json** and **intersection.json** from the CSV.")

    if st.session_state.pop("_show_nswe_warning_once", False):
        st.warning(
            "No **n_approaching_***-style column names were found — generated configs use **N, S, E, W**. "
            "Adjust the **column map** fields below if your CSV uses different names."
        )

    render_column_and_intersection_forms()

    with st.expander("Raw JSON (import / backup)", expanded=False):
        st.caption(
            "The forms above are the main editor. Paste complete JSON here and click **Apply** to load from a file, "
            "or copy the preview to save your config."
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**columns.json**")
            st.code(st.session_state.get("columns_text") or "{}", language="json")
            pasted_c = st.text_area("Paste columns.json", height=200, key="paste_columns_buffer", label_visibility="visible")
            if st.button("Apply pasted columns JSON", key="btn_apply_paste_columns"):
                try:
                    json.loads(pasted_c)
                    st.session_state.columns_text = pasted_c
                    st.session_state["_had_config_once"] = True
                    invalidate_config_widget_sync()
                    st.success("Applied column map.")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        with pc2:
            st.markdown("**intersection.json**")
            st.code(st.session_state.get("intersection_text") or "{}", language="json")
            pasted_i = st.text_area("Paste intersection.json", height=200, key="paste_intersection_buffer")
            if st.button("Apply pasted intersection JSON", key="btn_apply_paste_intersection"):
                try:
                    json.loads(pasted_i)
                    st.session_state.intersection_text = pasted_i
                    st.session_state["_had_config_once"] = True
                    invalidate_config_widget_sync()
                    st.success("Applied intersection config.")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

    columns_text = st.session_state.get("columns_text", "")
    intersection_text = st.session_state.get("intersection_text", "")

    with st.expander("Advanced — training & simulation (matches `main.py` constants)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            train_size = st.number_input("TRAIN_SIZE (days)", 1, 100, key="adv_train")
            test_size = st.number_input("TEST_SIZE (days)", 1, 100, key="adv_test")
            epochs = st.number_input(
                "EPOCHS",
                1,
                500,
                key="adv_epochs",
                help=f"Quick tests often use {STREAMLIT_DEFAULT_EPOCHS}; **Apply trial** sets EPOCHS={m.EPOCHS} from config.json. "
                f"CLI `main.py` currently uses n_epochs={m.EPOCHS}.",
            )
        with c2:
            step_length = st.number_input("STEP_LENGTH (s)", 0.5, 60.0, float(m.STEP_LENGTH))
            sim_begin = st.number_input("SIM_BEGIN", 0, 200000, int(m.SIM_BEGIN))
            sim_end = st.number_input("SIM_END", 0, 200000, int(m.SIM_END))
        with c3:
            min_green = st.number_input("MIN_GREEN_S", 1, 300, int(m.MIN_GREEN_S))
            max_green = st.number_input("MAX_GREEN_S", 1, 600, int(m.MAX_GREEN_S))
            save_every = st.number_input("SAVE_EVERY", 1, 10000, int(m.SAVE_EVERY))
            log_every = st.number_input("LOG_EVERY", 1, 100, int(m.LOG_EVERY))

    use_gui = st.checkbox("Run SUMO with GUI (`--gui`)", value=False)

    col_run, col_base = st.columns(2)
    with col_run:
        run_train = st.button("Run training pipeline", type="primary", use_container_width=True)
    with col_base:
        run_baseline = st.button("Evaluate baselines only", use_container_width=True)

    # Persist uploaded config to disk
    data_dir = RUNS / "data"
    int_path = RUNS / "intersection.json"
    col_path = RUNS / "columns.json"

    if csv_file is not None:
        csv_path.write_bytes(csv_file.getvalue())

    if run_train or run_baseline:
        if csv_file is None and not csv_path.exists():
            st.error("Upload a CSV first (required each new browser session).")
            return
        if not str(columns_text).strip() or not str(intersection_text).strip():
            st.error(
                "Column map or intersection config is empty. Click **Generate from CSV** (after loading a CSV), "
                "or paste JSON under **Raw JSON** and click **Apply**."
            )
            return
        try:
            intersection_obj = json.loads(intersection_text)
            col_map_obj = json.loads(columns_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        if not intersection_obj.get("active_approaches"):
            st.error("Intersection JSON has no **active_approaches**. Use **Generate from CSV** or fix the JSON.")
            return
        if not (col_map_obj.get("approaches") or {}):
            st.error("Column map has no **approaches**. Use **Generate from CSV** or fix the JSON.")
            return
        int_path.write_text(intersection_text, encoding="utf-8")
        col_path.write_text(columns_text, encoding="utf-8")

        for msg in _consistency_errors(intersection_obj, col_map_obj):
            st.error(msg)
            return

    if run_train:
        if not sumo_home or not os.path.isdir(sumo_home):
            st.error("Set a valid SUMO_HOME path in the sidebar.")
            return
        _sumo_bin_on_path(sumo_home)

        # Apply settings to main module
        m.CSV_PATH = str(csv_path)
        m.INTERSECTION_PATH = str(int_path)
        m.COLUMNS_PATH = str(col_path)
        m.OUT_DIR = str(data_dir)
        m.TRAIN_SIZE = int(train_size)
        m.TEST_SIZE = int(test_size)
        m.EPOCHS = int(epochs)
        m.STEP_LENGTH = float(step_length)
        m.SIM_BEGIN = int(sim_begin)
        m.SIM_END = int(sim_end)
        m.MIN_GREEN_S = int(min_green)
        m.MAX_GREEN_S = int(max_green)
        m.SAVE_EVERY = int(save_every)
        m.LOG_EVERY = int(log_every)
        m.SUMO_HOME = sumo_home
        _recompute_derived(m)

        try:
            with st.spinner("Validating inputs…"):
                m.validate_inputs(sumo_home)
            with st.spinner("Building routes & flows…"):
                m.run_build_route()
            with st.spinner("Building SUMO network…"):
                net_file = m.run_build_network(sumo_home)
            with st.spinner("Creating run folder…"):
                reward_kwargs, _ = m._load_component_config(
                    m.REWARD_CONFIG_PATH, m.RewardClass.__name__, "Reward"
                )
                reward_kwargs = m.merge_reward_kwargs_from_config(reward_kwargs)
                policy_kwargs_raw, _ = m._load_component_config(
                    m.POLICY_CONFIG_PATH, m.PolicyClass.__name__, "Policy"
                )
                policy_kwargs_raw = m.merge_policy_kwargs_from_config(policy_kwargs_raw)
                policy_kwargs = m._resolve_policy_params(policy_kwargs_raw)
                run_dir = m._create_run_dir()
                m._write_config_summary(run_dir, use_gui, reward_kwargs, policy_kwargs)
        except Exception as e:
            st.exception(e)
            return

        # --- Live training UI ---
        st.subheader("Training")
        prog_bar   = st.progress(0.0, text="Starting…")
        col_log, col_chart = st.columns([1, 1])
        with col_log:
            st.caption("Live log")
            log_box = st.empty()
        with col_chart:
            st.caption("Mean train reward per epoch")
            chart_box = st.empty()
            st.caption("Total reward per episode")
            episode_chart_box = st.empty()

        log_lines: list[str] = []
        reward_series: list[dict] = []
        _ts = int(train_size)

        def _on_log(text: str) -> None:
            log_lines.append(text)
            # keep last 200 lines to avoid giant DOM
            visible = log_lines[-200:]
            log_box.code("\n".join(visible), language=None)

            # parse episode reward lines: "  Train ep=  1  reward=  1234.5  …"
            if "reward=" in text and ("Train" in text or "Test" in text):
                try:
                    r_part = text.split("reward=")[1].split()[0]
                    ep_part = text.split("ep=")[1].split()[0] if "ep=" in text else None
                    ep_i = int(ep_part) if ep_part else len(reward_series) + 1
                    epoch_i = (ep_i - 1) // _ts + 1 if _ts > 0 else ep_i
                    reward_series.append({
                        "episode": ep_i,
                        "epoch": epoch_i,
                        "reward":  float(r_part),
                        "type":    "Test" if "Test" in text else "Train",
                    })
                    chart_df = pd.DataFrame(reward_series)
                    train_chart = chart_df[chart_df["type"] == "Train"]
                    if not train_chart.empty:
                        per_epoch = train_chart.groupby("epoch", sort=True)["reward"].mean()
                        chart_box.line_chart(
                            per_epoch,
                            use_container_width=True,
                        )
                        episode_chart_box.line_chart(
                            train_chart.set_index("episode")["reward"],
                            use_container_width=True,
                        )
                except (IndexError, ValueError):
                    pass

            # update progress bar from epoch lines "--- Epoch N/M ---"
            if "Epoch" in text and "/" in text:
                try:
                    parts = text.strip().strip("-").strip().split()
                    frac_str = [p for p in parts if "/" in p][0]
                    n, total = frac_str.split("/")
                    frac = int(n) / int(total)
                    prog_bar.progress(frac, text=f"Epoch {n} / {total}")
                except (IndexError, ValueError):
                    pass

        try:
            trainer = m.build_pipeline(net_file, use_gui, run_dir, reward_kwargs, policy_kwargs)
            trainer.log_callback = _on_log
            results = trainer.run()
            _elapsed = float(m.SIM_END - m.SIM_BEGIN)
            _kpi_row = {
                "run_name": os.path.basename(run_dir),
                "reward_class": m.RewardClass.__name__,
                "policy_class": m.PolicyClass.__name__,
                **aggregate_test_kpis(results["test_log"], _elapsed),
            }
            _kpi_fields = [
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
                _kpi_row,
                fieldnames=_kpi_fields,
            )
        except Exception as e:
            st.exception(e)
            return

        prog_bar.progress(1.0, text="Done")
        log_box.code("\n".join(log_lines[-200:]), language=None)

        st.session_state["last_train_results"] = results
        st.session_state["last_run_dir"] = run_dir
        st.session_state["last_data_dir"] = str(data_dir)
        st.success("Training finished.")
        st.subheader("Results")

        train_df = pd.DataFrame(results["train_log"])
        pretest_df = pd.DataFrame(results.get("pretest_log") or [])
        test_df = pd.DataFrame(results["test_log"])
        st.write("**Train log (one row per train day; epoch × day)**")
        st.dataframe(train_df, use_container_width=True)
        st.write("**Pre-train test log (held-out days, before learning)**")
        st.dataframe(pretest_df, use_container_width=True)
        st.write("**Test log (per day)**")
        st.dataframe(test_df, use_container_width=True)

        if not train_df.empty and "total_reward" in train_df.columns:
            if "epoch" in train_df.columns:
                st.write("**Mean total reward per epoch (train)**")
                st.line_chart(train_df.groupby("epoch", sort=True)["total_reward"].mean())
                st.write("**Total reward per episode (train)**")
                ep_rewards = train_df["total_reward"].reset_index(drop=True)
                ep_rewards.index.name = "episode"
                st.line_chart(ep_rewards)
            else:
                st.write("**Total reward per episode (train)**")
                st.line_chart(train_df.set_index("episode")["total_reward"])

        mean_test = (
            sum(x["total_reward"] for x in results["test_log"]) / len(results["test_log"])
            if results["test_log"]
            else None
        )
        mean_pretest = (
            sum(x["total_reward"] for x in results.get("pretest_log") or []) / len(results.get("pretest_log") or [])
            if results.get("pretest_log")
            else None
        )
        if mean_pretest is not None:
            st.metric("Mean pre-train test reward", f"{mean_pretest:.2f}")
        if mean_test is not None:
            st.metric("Mean test reward", f"{mean_test:.2f}")

        schedule_path = Path(run_dir) / "schedule.json"
        if schedule_path.exists():
            with open(schedule_path, encoding="utf-8") as _sf:
                schedule_data = json.load(_sf)
            buckets = schedule_data.get("buckets", [])
            if buckets:
                st.write("**Learned Signal Timing Schedule**")
                st.caption(
                    "Median phase durations (seconds) per 15-minute time bucket, "
                    "aggregated from test episodes. This is the RL policy distilled "
                    "into a deployable fixed-time plan."
                )
                sched_df = pd.DataFrame(buckets)
                sched_df["time"] = sched_df["bucket_start_s"].apply(
                    lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}"
                )
                display_cols = ["tls_id", "phase", "time", "median_s", "std_s", "n"]
                st.dataframe(
                    sched_df[[c for c in display_cols if c in sched_df.columns]],
                    use_container_width=True,
                )
                sched_bytes = json.dumps(schedule_data, indent=2).encode("utf-8")
                st.download_button(
                    "Download schedule.json",
                    sched_bytes,
                    "schedule.json",
                    "application/json",
                )

        st.info(f"Artifacts: `{run_dir}`")

    if run_baseline:
        if not sumo_home or not os.path.isdir(sumo_home):
            st.error("Set a valid SUMO_HOME path in the sidebar.")
            return
        _sumo_bin_on_path(sumo_home)

        m.CSV_PATH = str(csv_path)
        m.COLUMNS_PATH = str(col_path)
        m.TRAIN_SIZE = int(train_size)
        m.TEST_SIZE = int(test_size)
        m.EPOCHS = int(epochs)
        m.STEP_LENGTH = float(step_length)
        m.SIM_BEGIN = int(sim_begin)
        m.SIM_END = int(sim_end)
        m.MIN_GREEN_S = int(min_green)
        m.MAX_GREEN_S = int(max_green)
        m.SAVE_EVERY = int(save_every)
        m.LOG_EVERY = int(log_every)
        m.SUMO_HOME = sumo_home
        _recompute_derived(m)

        # Determine the data directory and intersection to use.
        # If a training run exists, reuse its data (split.json, flows, network)
        # to ensure baselines are evaluated on the SAME test days.
        if "last_run_dir" in st.session_state:
            models_dir = Path(st.session_state["last_run_dir"])
        else:
            models_dir = RUNS / "baseline_models"
            models_dir.mkdir(parents=True, exist_ok=True)

        # Use the main project data dir (src/data) so baselines share the
        # same split.json, flow files, and network as the DQN training run.
        # Fall back to streamlit_runs/data only if src/data has no split.
        main_data_dir = ROOT / "src" / "data"
        main_split = main_data_dir / "processed" / "split.json"
        if main_split.exists():
            eval_data_dir = main_data_dir
            st.info(f"Using existing split from `{main_split}` for fair comparison.")
        else:
            eval_data_dir = data_dir
            st.warning(
                "No existing split.json found in src/data. "
                "Building routes (baselines may use different test days than training)."
            )
            m.INTERSECTION_PATH = str(int_path)
            m.OUT_DIR = str(data_dir)
            try:
                with st.spinner("Building routes & network…"):
                    m.validate_inputs(sumo_home)
                    m.run_build_route()
                    m.run_build_network(sumo_home)
            except Exception as e:
                st.exception(e)
                return

        # Use src/intersection.json (same as training) if it exists;
        # otherwise fall back to the Streamlit-generated one.
        main_int_path = ROOT / "src" / "intersection.json"
        baseline_int_path = main_int_path if main_int_path.exists() else int_path

        eval_cfg_path = RUNS / "eval_config.json"
        _write_eval_config(m, eval_data_dir, models_dir, eval_cfg_path)

        cmd = [
            sys.executable,
            str(ROOT / "evaluate_baseline.py"),
            "--config",
            str(eval_cfg_path),
            "--intersection",
            str(baseline_int_path),
            "--sumo-home",
            sumo_home,
        ]
        if use_gui:
            cmd.append("--gui")

        with st.spinner("Running baseline evaluation…"):
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        st.code(proc.stdout + proc.stderr, language="text")
        if proc.returncode != 0:
            st.error("Baseline script exited with an error.")
        else:
            st.success("Baseline evaluation finished.")

        for name in ("baseline_fixed_time_log.json", "baseline_actuated_log.json"):
            p = models_dir / name
            if p.exists():
                st.write(f"**{name}**")
                st.dataframe(pd.read_json(p), use_container_width=True)
        st.session_state["last_baseline_dir"] = str(models_dir)

    # ── Evaluate saved checkpoint ─────────────────────────────────────────────
    st.divider()
    st.subheader("Evaluate Saved Checkpoint")
    st.caption(
        "Load a previously saved checkpoint (.pt) and run the test evaluation "
        "without retraining. Writes test_log.json and pretest_log.json to the run directory."
    )

    with st.expander("Evaluate checkpoint", expanded=False):
        ckpt_path_input = st.text_input(
            "Checkpoint path (.pt)",
            value=str(
                Path(st.session_state.get("last_run_dir", ""))
                / "checkpoints"
                / "checkpoint_ep0999.pt"
            ) if st.session_state.get("last_run_dir") else "",
            placeholder="e.g. src/data/results/2026-.../checkpoints/checkpoint_ep0999.pt",
        )
        ckpt_run_dir = st.text_input(
            "Run directory (where to save test_log.json)",
            value=st.session_state.get("last_run_dir", ""),
            placeholder="e.g. src/data/results/2026-..._DoubleDQNPolicy_ThroughputCompositeReward",
        )

        if st.button("Run test evaluation from checkpoint"):
            ckpt_path_input = ckpt_path_input.strip()
            ckpt_run_dir = ckpt_run_dir.strip()

            if not ckpt_path_input or not os.path.isfile(ckpt_path_input):
                st.error(f"Checkpoint file not found: {ckpt_path_input}")
            elif not ckpt_run_dir:
                st.error("Run directory must be specified.")
            else:
                os.makedirs(ckpt_run_dir, exist_ok=True)
                try:
                    reward_kwargs, _ = m._load_component_config(
                        m.REWARD_CONFIG_PATH, m.RewardClass.__name__, "Reward"
                    )
                    reward_kwargs = m.merge_reward_kwargs_from_config(reward_kwargs)
                    policy_kwargs_ckpt, _ = m._load_component_config(
                        m.POLICY_CONFIG_PATH, m.PolicyClass.__name__, "Policy"
                    )
                    policy_kwargs_ckpt = m.merge_policy_kwargs_from_config(policy_kwargs_ckpt)

                    net_file_ckpt = str(
                        Path(m.SUMO_HOME) / "bin" / ".."
                        if False
                        else Path("src/data/sumo/network").glob("*.net.xml")
                        and next(Path("src/data/sumo/network").glob("*.net.xml"))
                    )
                    # Resolve net file properly
                    net_candidates = list(Path("src/data/sumo/network").glob("*.net.xml"))
                    if not net_candidates:
                        st.error("No .net.xml found in src/data/sumo/network — run preprocessing first.")
                        st.stop()
                    net_file_ckpt = str(net_candidates[0])

                    with st.spinner("Loading checkpoint and running test evaluation…"):
                        ckpt_log_lines = []
                        ckpt_log_box = st.empty()

                        def _ckpt_log(text: str) -> None:
                            ckpt_log_lines.append(text)
                            ckpt_log_box.code("\n".join(ckpt_log_lines[-100:]), language=None)

                        trainer_ckpt = m.build_pipeline(
                            net_file_ckpt, False, ckpt_run_dir,
                            reward_kwargs, policy_kwargs_ckpt,
                        )
                        trainer_ckpt.n_epochs = 0
                        trainer_ckpt.agent.load(ckpt_path_input)
                        trainer_ckpt.log_callback = _ckpt_log
                        ckpt_results = trainer_ckpt.run()

                    test_rewards = [e["total_reward"] for e in ckpt_results["test_log"]]
                    mean_test_ckpt = sum(test_rewards) / len(test_rewards) if test_rewards else None
                    pretest_rewards = [e["total_reward"] for e in ckpt_results.get("pretest_log", [])]
                    mean_pretest_ckpt = sum(pretest_rewards) / len(pretest_rewards) if pretest_rewards else None

                    st.success("Test evaluation complete.")
                    col1, col2 = st.columns(2)
                    if mean_pretest_ckpt is not None:
                        col1.metric("Mean pre-test reward", f"{mean_pretest_ckpt:.2f}")
                    if mean_test_ckpt is not None:
                        col2.metric("Mean test reward (checkpoint)", f"{mean_test_ckpt:.2f}")
                    st.info(f"test_log.json saved to: `{ckpt_run_dir}`")
                    st.session_state["last_run_dir"] = ckpt_run_dir

                except Exception as e:
                    st.exception(e)

    # ── Load existing run results ─────────────────────────────────────────────
    st.divider()
    st.subheader("Load Existing Run Results")
    st.caption("Paste a run directory that already has test_log.json and baseline logs to view results and plots without re-running.")

    with st.expander("Load from directory", expanded=False):
        load_dir_input = st.text_input(
            "Run directory",
            placeholder="e.g. src/data/results/2026-04-05_18-06-20_DoubleDQNPolicy_ThroughputCompositeReward",
        )
        if st.button("Load results"):
            load_dir_input = load_dir_input.strip()
            load_p = Path(load_dir_input)
            dqn_log_p     = load_p / "test_log.json"
            fixed_log_p   = load_p / "baseline_fixed_time_log.json"
            actuated_log_p = load_p / "baseline_actuated_log.json"
            missing = [str(p) for p in [dqn_log_p, fixed_log_p, actuated_log_p] if not p.exists()]
            if missing:
                st.error(f"Missing files:\n" + "\n".join(missing))
            else:
                with open(dqn_log_p) as f:
                    dqn_log = json.load(f)
                with open(fixed_log_p) as f:
                    fixed_log = json.load(f)
                with open(actuated_log_p) as f:
                    actuated_log = json.load(f)

                # Build comparison table
                dqn_map    = {e["day_id"]: e["total_reward"] for e in dqn_log}
                fixed_map  = {e["day_id"]: e["total_reward"] for e in fixed_log}
                act_map    = {e["day_id"]: e["total_reward"] for e in actuated_log}
                all_days   = sorted(set(dqn_map) | set(fixed_map) | set(act_map))

                rows = []
                for d in all_days:
                    rows.append({
                        "Day":        d,
                        "DQN":        round(dqn_map.get(d, float("nan")), 2),
                        "Fixed-Time": round(fixed_map.get(d, float("nan")), 2),
                        "Actuated":   round(act_map.get(d, float("nan")), 2),
                    })
                cmp_df = pd.DataFrame(rows)
                means = {
                    "Day": "Mean",
                    "DQN":        round(sum(dqn_map.values()) / len(dqn_map), 2),
                    "Fixed-Time": round(sum(fixed_map.values()) / len(fixed_map), 2),
                    "Actuated":   round(sum(act_map.values()) / len(act_map), 2),
                }
                cmp_df = pd.concat([cmp_df, pd.DataFrame([means])], ignore_index=True)

                st.write("**Results Comparison — ThroughputCompositeReward**")
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                mean_dqn   = means["DQN"]
                mean_fixed = means["Fixed-Time"]
                mean_act   = means["Actuated"]
                pct_fixed = ((mean_dqn - mean_fixed) / abs(mean_fixed)) * 100
                pct_act   = ((mean_dqn - mean_act)   / abs(mean_act))   * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("DQN mean reward",        f"{mean_dqn:.2f}")
                c2.metric("vs Fixed-Time",          f"{pct_fixed:+.1f}%")
                c3.metric("vs Actuated",            f"{pct_act:+.1f}%")

                st.session_state["last_run_dir"]      = load_dir_input
                st.session_state["last_baseline_dir"] = load_dir_input
                st.success("Results loaded. Scroll down to Generate Plots.")

                # Show schedule.json if it exists in the loaded run
                sched_p = load_p / "schedule.json"
                if sched_p.exists():
                    with open(sched_p, encoding="utf-8") as _sf:
                        sched_data = json.load(_sf)
                    buckets = sched_data.get("buckets", [])
                    if buckets:
                        st.write("**Learned Signal Timing Schedule**")
                        st.caption(
                            "Median phase durations (seconds) per 15-minute time bucket, "
                            "aggregated from test episodes."
                        )
                        sched_df = pd.DataFrame(buckets)
                        sched_df["time"] = sched_df["bucket_start_s"].apply(
                            lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}"
                        )
                        display_cols = ["tls_id", "phase", "time", "median_s", "std_s", "n"]
                        st.dataframe(
                            sched_df[[c for c in display_cols if c in sched_df.columns]],
                            use_container_width=True,
                        )
                        sched_bytes = json.dumps(sched_data, indent=2).encode("utf-8")
                        st.download_button(
                            "Download schedule.json",
                            sched_bytes,
                            "schedule.json",
                            "application/json",
                        )

    # ── Comparison plots ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Comparison Plots")

    run_dir_for_plots = st.session_state.get("last_run_dir")
    baseline_dir_for_plots = st.session_state.get("last_baseline_dir")

    plots_ready = run_dir_for_plots and baseline_dir_for_plots
    if not plots_ready:
        st.info("Run training and baseline evaluation first, then generate plots here.")

    if st.button("Generate comparison plots", disabled=not plots_ready, type="primary"):
        run_dir_p   = Path(run_dir_for_plots)
        base_dir_p  = Path(baseline_dir_for_plots)

        needed = {
            "test_log.json":                 run_dir_p  / "test_log.json",
            "baseline_fixed_time_log.json":  base_dir_p / "baseline_fixed_time_log.json",
            "baseline_actuated_log.json":    base_dir_p / "baseline_actuated_log.json",
        }
        train_log_path = run_dir_p / "train_log.json"
        missing = [k for k, v in needed.items() if not v.exists()]
        if missing:
            st.error(f"Missing log files: {', '.join(missing)}")
        else:
            def _load(p): 
                with open(p) as f: 
                    return json.load(f)

            if train_log_path.exists():
                train_log = _load(train_log_path)
                if not isinstance(train_log, list):
                    train_log = []
            else:
                train_log = []
            dqn_log     = _load(needed["test_log.json"])
            fixed_log   = _load(needed["baseline_fixed_time_log.json"])
            actuated_log = _load(needed["baseline_actuated_log.json"])

            MODELS = ["Fixed-Time", "Actuated", "DQN"]
            COLORS = ["#4C72B0", "#DD8452", "#55A868"]
            LOGS   = [fixed_log, actuated_log, dqn_log]

            def rewards(log):
                return [e["total_reward"] for e in log]

            def day_ids(log):
                return [e["day_id"] for e in log]

            STYLE = {
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 130,
            }
            plt.rcParams.update(STYLE)

            col_a, col_b = st.columns(2)

            # 1. Bar chart
            with col_a:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                means = [np.mean(rewards(log)) for log in LOGS]
                bars = ax.bar(MODELS, means, color=COLORS, width=0.5, zorder=3)
                ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
                ax.set_ylabel("Mean Total Reward")
                ax.set_title("Mean Test Reward by Policy")
                ax.set_ylim(min(means) * 1.15, max(0, max(means) * 1.1))
                fig.tight_layout()
                st.pyplot(fig)
                buf = _fig_to_bytes(fig)
                st.download_button("Download bar chart", buf, "bar_mean_reward.png", "image/png")
                plt.close(fig)

            # 2. Box plot
            with col_b:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                bp = ax.boxplot(
                    [rewards(log) for log in LOGS],
                    tick_labels=MODELS,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    zorder=3,
                )
                for patch, color in zip(bp["boxes"], COLORS):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                ax.set_ylabel("Total Reward")
                ax.set_title("Reward Distribution Across Test Days")
                fig.tight_layout()
                st.pyplot(fig)
                buf = _fig_to_bytes(fig)
                st.download_button("Download box plot", buf, "box_reward_dist.png", "image/png")
                plt.close(fig)

            col_c, col_d = st.columns(2)

            # 3. Per-day line chart
            with col_c:
                shared_ids = sorted(
                    set(day_ids(fixed_log)) & set(day_ids(actuated_log)) & set(day_ids(dqn_log))
                )
                def aligned(log, ids):
                    lookup = {e["day_id"]: e["total_reward"] for e in log}
                    return [lookup[d] for d in ids]

                fig, ax = plt.subplots(figsize=(5, 3.5))
                for label, log, color in zip(MODELS, LOGS, COLORS):
                    ax.plot(shared_ids, aligned(log, shared_ids), marker="o",
                            label=label, color=color, linewidth=2, markersize=6)
                ax.set_xlabel("Test Day ID")
                ax.set_ylabel("Total Reward")
                ax.set_title("Per-Day Reward: All Policies")
                ax.legend(framealpha=0.8)
                fig.tight_layout()
                st.pyplot(fig)
                buf = _fig_to_bytes(fig)
                st.download_button("Download per-day chart", buf, "line_reward_per_day.png", "image/png")
                plt.close(fig)

            # 4. Learning curve (optional — train_log may be missing if run crashed)
            with col_d:
                tr_reward = [e["total_reward"] for e in train_log if "total_reward" in e]
                episodes = [e.get("episode", i + 1) for i, e in enumerate(train_log)]
                if len(tr_reward) < 2:
                    st.info(
                        "No training log (or too few episodes). "
                        "Learning curve skipped — complete a full training run to generate "
                        "`train_log.json`."
                    )
                else:
                    window = min(10, len(tr_reward))
                    if window < 1:
                        window = 1
                    kernel = np.ones(window) / window
                    smoothed = np.convolve(tr_reward, kernel, mode="valid")
                    smoothed_x = episodes[window - 1 :]

                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.plot(episodes, tr_reward, color="#bbbbbb", linewidth=0.8,
                            label="Raw reward", zorder=2)
                    if len(smoothed) > 0:
                        ax.plot(smoothed_x, smoothed, color=COLORS[2], linewidth=2,
                                label=f"{window}-ep rolling mean", zorder=3)
                    ax.set_xlabel("Training Episode")
                    ax.set_ylabel("Total Reward")
                    ax.set_title("DQN Learning Curve (Training)")
                    ax.legend(framealpha=0.8)
                    fig.tight_layout()
                    st.pyplot(fig)
                    buf = _fig_to_bytes(fig)
                    st.download_button("Download learning curve", buf, "learning_curve.png", "image/png")
                    plt.close(fig)

            st.success("Plots generated. Use the download buttons above to save each chart.")


def _fig_to_bytes(fig) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    main()
