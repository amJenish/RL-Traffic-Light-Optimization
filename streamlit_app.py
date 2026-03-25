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

from streamlit_intersection_helpers import (
    DEFAULT_TIMING,
    build_intersection_dict,
    detect_approaches_from_column_map,
    detect_approaches_from_headers,
    suggest_columns_json_from_dataframe,
    suggest_intersection_name,
    validate_csv_minimum,
)

# Project root = directory containing this file
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = ROOT / "streamlit_runs"
RUNS.mkdir(parents=True, exist_ok=True)

# UI default for EPOCHS (quick local tests). Change to 60 to match main.py, or set in Advanced.
STREAMLIT_DEFAULT_EPOCHS = 10


def _recompute_derived(m) -> None:
    """Match main.py: TOTAL_UPDATES and EPSILON_DECAY from timing constants."""
    sim_seconds = m.SIM_END - m.SIM_BEGIN
    steps_per_ep = sim_seconds / m.STEP_LENGTH
    min_gs = max(1, math.ceil(m.MIN_GREEN_S / m.STEP_LENGTH))
    decisions_per_ep = steps_per_ep / (min_gs + 1)
    m.TOTAL_UPDATES = int(m.EPOCHS * m.TRAIN_SIZE * decisions_per_ep)
    tu = max(1, m.TOTAL_UPDATES)
    m.EPSILON_DECAY = (m.EPSILON_END / m.EPSILON_START) ** (1.0 / (0.85 * tu))


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
            "decision_gap": 10,
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
            "normalise": bool(m.REWARD_NORMALISE),
            "scale": float(m.REWARD_SCALE),
            "alpha": float(m.REWARD_ALPHA),
        },
        "policy": {
            "learning_rate": float(m.LEARNING_RATE),
            "gamma": float(m.GAMMA),
            "epsilon_start": float(m.EPSILON_START),
            "epsilon_end": float(m.EPSILON_END),
            "epsilon_decay": float(m.EPSILON_DECAY),
            "target_update": int(m.TARGET_UPDATE),
            "batch_size": int(m.BATCH_SIZE),
            "hidden": int(m.HIDDEN_SIZE),
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
        "Upload traffic data, edit intersection / column maps, tune training, "
        "then run the pipeline (same code paths as `main.py`)."
    )

    import main as m  # noqa: PLC0415 — after sys.path

    default_intersection = (ROOT / "src" / "intersection.json").read_text(encoding="utf-8")
    default_columns = (ROOT / "src" / "columns.json").read_text(encoding="utf-8")
    st.session_state.setdefault("intersection_text", default_intersection)
    st.session_state.setdefault("columns_text", default_columns)

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
        "Upload counts CSV (same shape as `src/data/synthetic_toronto_data.csv`)", type=["csv"]
    )
    csv_path = RUNS / "traffic.csv"
    df_preview: pd.DataFrame | None = None
    if csv_file is not None:
        csv_path.write_bytes(csv_file.getvalue())
        df_preview = pd.read_csv(csv_file)
        st.dataframe(df_preview.head(8), use_container_width=True)
    elif csv_path.exists():
        df_preview = pd.read_csv(csv_path)

    st.subheader("2. Column map (`columns.json`)")
    st.caption("Define the column map first (or paste defaults). The suggestion helper uses this.")
    columns_text = st.text_area(
        "JSON",
        height=240,
        key="columns_text",
        help="Paste or edit columns.json. Use 'Apply auto columns.json' in the expander below if "
        "headers match n_approaching_t style.",
    )

    st.subheader("3. Intersection (`intersection.json`)")
    intersection_text = st.text_area(
        "JSON",
        height=220,
        key="intersection_text",
        help="Paste or edit intersection.json. Use the expander below to generate a draft.",
    )

    with st.expander("Suggest intersection from CSV (lanes + defaults)", expanded=False):
        st.markdown(
            "Detects **active approaches** from your `columns.json` (or from `n_approaching_*` "
            "style headers). You set **lane counts** and **speed**; timing uses standard defaults "
            "(editable below). **Counts alone cannot infer geometry** — adjust as needed."
        )
        if df_preview is None:
            st.info("Upload a CSV above to enable suggestions.")
        else:
            col_parse_err: str | None = None
            col_map_try: dict | None = None
            try:
                col_map_try = json.loads(st.session_state.columns_text)
            except json.JSONDecodeError as e:
                col_parse_err = str(e)

            if col_parse_err:
                st.warning(f"Fix `columns.json` above to valid JSON for map-based detection: {col_parse_err}")
                detected = detect_approaches_from_headers(df_preview)
                st.caption("Using header heuristics only (n_approaching_* column names).")
            else:
                from_map, map_warn = detect_approaches_from_column_map(df_preview, col_map_try)
                detected = from_map if from_map else detect_approaches_from_headers(df_preview)
                for w in map_warn:
                    st.warning(w)

            if not detected:
                st.error(
                    "Could not detect approaches. Add a valid columns.json or rename columns "
                    "to e.g. n_approaching_t, s_approaching_t, …"
                )
            else:
                errs, warns = validate_csv_minimum(df_preview, col_map_try)
                for e in errs:
                    st.error(e)
                for w in warns:
                    st.warning(w)

                name_in = st.text_input(
                    "Intersection name (used in SUMO file names)",
                    value=suggest_intersection_name(df_preview),
                    key="gen_int_name",
                )
                active_sel = st.multiselect(
                    "Active approaches",
                    options=["N", "S", "E", "W"],
                    default=detected,
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    lt = st.number_input("Through lanes", 1, 8, 2, key="gen_lt")
                with c2:
                    lr = st.number_input("Right lanes", 0, 8, 1, key="gen_lr")
                with c3:
                    ll = st.number_input("Left lanes", 0, 8, 1, key="gen_ll")
                with c4:
                    sp = st.number_input("Speed (km/h)", 20, 130, 50, key="gen_sp")

                with st.expander("Timing defaults (usually standard)"):
                    t1, t2, t3 = st.columns(3)
                    with t1:
                        ph = st.text_input("phases", value=str(DEFAULT_TIMING["phases"]), key="gen_ph")
                        mr = st.number_input("min_red_s", 1, 120, int(DEFAULT_TIMING["min_red_s"]), key="gen_mr")
                    with t2:
                        mg = st.number_input("min_green_s", 1, 300, int(DEFAULT_TIMING["min_green_s"]), key="gen_mg")
                        mx = st.number_input("max_green_s", 1, 600, int(DEFAULT_TIMING["max_green_s"]), key="gen_mx")
                    with t3:
                        am = st.number_input("amber_s", 1, 30, int(DEFAULT_TIMING["amber_s"]), key="gen_am")
                        cy = st.number_input("cycle_s", 30, 300, int(DEFAULT_TIMING["cycle_s"]), key="gen_cy")
                        el = st.number_input(
                            "edge_length_m", 10, 500, int(DEFAULT_TIMING["edge_length_m"]), key="gen_el"
                        )

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Apply generated intersection to editor", key="btn_apply_int"):
                        if not active_sel:
                            st.error("Select at least one active approach.")
                        else:
                            timing = {
                                "phases": str(ph),
                                "min_red_s": int(mr),
                                "min_green_s": int(mg),
                                "max_green_s": int(mx),
                                "amber_s": int(am),
                                "cycle_s": int(cy),
                                "edge_length_m": int(el),
                            }
                            obj = build_intersection_dict(
                                name_in,
                                active_sel,
                                int(lt),
                                int(lr),
                                int(ll),
                                float(sp),
                                timing=timing,
                            )
                            st.session_state.intersection_text = json.dumps(obj, indent=2)
                            st.success("Intersection JSON updated — review section 3.")
                            st.rerun()
                with b2:
                    if st.button("Apply auto columns.json from CSV headers", key="btn_apply_col"):
                        if not active_sel:
                            st.error("Select at least one active approach.")
                        else:
                            sug = suggest_columns_json_from_dataframe(df_preview, active_sel)
                            st.session_state.columns_text = json.dumps(sug, indent=2)
                            st.success("Column map updated — review section 2.")
                            st.rerun()

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
            st.error("Upload a CSV first (or place `streamlit_runs/traffic.csv`).")
            return
        try:
            intersection_obj = json.loads(intersection_text)
            col_map_obj = json.loads(columns_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
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
                run_dir = m._create_run_dir()
                m._write_config_summary(run_dir, use_gui)
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
            st.caption("Reward per episode")
            chart_box = st.empty()

        log_lines: list[str] = []
        reward_series: list[dict] = []
        total_episodes = int(epochs) * int(train_size)

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
                    reward_series.append({
                        "episode": int(ep_part) if ep_part else len(reward_series) + 1,
                        "reward":  float(r_part),
                        "type":    "Test" if "Test" in text else "Train",
                    })
                    chart_df = pd.DataFrame(reward_series)
                    train_chart = chart_df[chart_df["type"] == "Train"]
                    if not train_chart.empty:
                        chart_box.line_chart(
                            train_chart.set_index("episode")[["reward"]],
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
            trainer = m.build_pipeline(net_file, use_gui, run_dir)
            trainer.log_callback = _on_log
            results = trainer.run()
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
        test_df = pd.DataFrame(results["test_log"])
        st.write("**Train log (per episode)**")
        st.dataframe(train_df, use_container_width=True)
        st.write("**Test log (per day)**")
        st.dataframe(test_df, use_container_width=True)

        if not train_df.empty and "total_reward" in train_df.columns:
            st.line_chart(train_df.set_index("episode")["total_reward"])

        mean_test = (
            sum(x["total_reward"] for x in results["test_log"]) / len(results["test_log"])
            if results["test_log"]
            else None
        )
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
