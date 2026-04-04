"""
RL Traffic Signal Optimizer — Streamlit UI
------------------------------------------
Simplest stack: pure Python + Streamlit (https://docs.streamlit.io).

Run from the project root:
    streamlit run streamlit_app.py

Requires: SUMO installed, SUMO_HOME set (or paste path in the sidebar).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

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

    import main as m  # noqa: PLC0415 — after sys.path

    _discard_previous_session_traffic_csv()

    # Empty until **Generate from CSV** (or paste). Streamlit keeps this across reruns in the same tab.
    st.session_state.setdefault("intersection_text", "")
    st.session_state.setdefault("columns_text", "")
    st.session_state.setdefault("_had_config_once", False)

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
            train_size = st.number_input("TRAIN_SIZE (days)", 1, 100, int(m.TRAIN_SIZE))
            test_size = st.number_input("TEST_SIZE (days)", 1, 100, int(m.TEST_SIZE))
            epochs = st.number_input(
                "EPOCHS",
                1,
                500,
                STREAMLIT_DEFAULT_EPOCHS,
                help=f"Default {STREAMLIT_DEFAULT_EPOCHS} for quick tests; main.py uses {m.EPOCHS} if you run CLI.",
            )
        with c2:
            step_length = st.number_input("STEP_LENGTH (s)", 0.5, 60.0, float(m.STEP_LENGTH))
            sim_begin = st.number_input("SIM_BEGIN", 0, 200000, int(m.SIM_BEGIN))
            sim_end = st.number_input("SIM_END", 0, 200000, int(m.SIM_END))
        with c3:
            min_green = st.number_input("MIN_GREEN_S", 1, 300, int(m.MIN_GREEN_S))
            max_green = st.number_input("MAX_GREEN_S", 1, 600, int(m.MAX_GREEN_S))
            save_every = st.number_input("SAVE_EVERY", 1, 500, int(m.SAVE_EVERY))
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
            st.caption("Mean train reward per epoch (updates from log)")
            chart_box = st.empty()

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
            else:
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
        st.info(f"Artifacts: `{run_dir}`")

    if run_baseline:
        if not sumo_home or not os.path.isdir(sumo_home):
            st.error("Set a valid SUMO_HOME path in the sidebar.")
            return
        _sumo_bin_on_path(sumo_home)

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
            with st.spinner("Building routes & network (if needed)…"):
                m.validate_inputs(sumo_home)
                m.run_build_route()
                m.run_build_network(sumo_home)
        except Exception as e:
            st.exception(e)
            return

        if "last_run_dir" in st.session_state:
            models_dir = Path(st.session_state["last_run_dir"])
        else:
            models_dir = RUNS / "baseline_models"
            models_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg_path = RUNS / "eval_config.json"
        _write_eval_config(m, data_dir, models_dir, eval_cfg_path)

        cmd = [
            sys.executable,
            str(ROOT / "evaluate_baseline.py"),
            "--config",
            str(eval_cfg_path),
            "--intersection",
            str(int_path),
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


if __name__ == "__main__":
    main()
