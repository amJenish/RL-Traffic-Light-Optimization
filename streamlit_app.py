from __future__ import annotations

import io
import json
import os
import sys
import traceback
import warnings
from datetime import time as dtime
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_system.replay_comparator import ReplayComparator
from eval_system.replay_runner import ReplayRunner
from eval_system.schedule_controller import ScheduleController
from modelling.schedule_builder import ScheduleBuilder
from modelling.webster_schedule_builder import WebsterScheduleBuilder
from streamlit_intersection_helpers import (
    build_intersection_dict_from_ui,
    suggest_columns_json_from_dataframe,
    suggest_intersection_name,
)

RUNS = ROOT / "streamlit_runs"
RUNS.mkdir(parents=True, exist_ok=True)

DIRECTIONS = ["N", "S", "E", "W"]
DIR_LABEL = {"N": "North", "S": "South", "E": "East", "W": "West"}

# Class names must match `modelling.components.reward` and `config.json` → `reward_configuration`.
REWARD_CLASS_OPTIONS = (
    "CompositeReward",
    "DeltaVehicleCountReward",
    "DeltaWaitingTimeReward",
    "ThroughputCompositeReward",
    "ThroughputQueueReward",
    "ThroughputReward",
    "ThroughputWaitTimeReward",
    "VehicleCountReward",
    "WaitingTimeReward",
)


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "columns_text": "",
        "intersection_text": "",
        "csv_bytes": None,
        "csv_df": None,
        "slot_minutes": 15,
        "tab1_valid": False,
        "tab2_valid": False,
        "run_in_progress": False,
        "run_complete": False,
        "last_run_dir": None,
        "last_schedule_path": None,
        "last_schedule_webster_path": None,
        "last_comparison_csv_path": None,
        "last_test_log": None,
        "training_log_lines": [],
        "training_reward_history": [],
        "training_epoch_history": [],
        "last_run_baseline_comparison": False,
        "current_stage": 0,
        "stage_statuses": ["waiting"] * 8,
        "eval_progress_rows": [],
        "pipeline_error_detail": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _new_session_cleanup() -> None:
    if st.session_state.get("_streamlit_new_session_done"):
        return
    st.session_state["_streamlit_new_session_done"] = True
    try:
        (RUNS / "traffic.csv").unlink(missing_ok=True)
    except OSError:
        pass


def _try_eclipse_sumo_home() -> str:
    """Default SUMO path when pip-installed eclipse-sumo is available (matches CLI fallback)."""
    try:
        import sumo  # noqa: PLC0415

        return str(sumo.SUMO_HOME)
    except Exception:
        return ""


def _time_to_seconds(t: dtime) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def _seconds_to_hhmm(sec: int) -> str:
    h = (sec // 3600) % 24
    m = (sec % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _detect_slot_minutes(df: pd.DataFrame, col: str | None) -> int:
    if df is None or df.empty or not col or col not in df.columns:
        return 15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        vals = pd.to_datetime(df[col].astype(str), errors="coerce").dropna()
    if len(vals) < 2:
        return 15
    diffs = vals.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 15
    minutes = int(round(diffs.mode().iloc[0] / 60.0))
    return int(max(1, min(60, minutes)))


def _render_intersection_svg(active: list[str]) -> None:
    def line(x1, y1, x2, y2, on):
        if on:
            return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2c7" stroke-width="8" />'
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#999" stroke-dasharray="6,6" stroke-width="6" />'

    html = f"""
    <svg viewBox="0 0 220 220" width="100%" height="220">
      {line(110,110,110,20,'N' in active)}
      {line(110,110,110,200,'S' in active)}
      {line(110,110,200,110,'E' in active)}
      {line(110,110,20,110,'W' in active)}
      <circle cx="110" cy="110" r="12" fill="#555" />
      <text x="105" y="14" font-size="14">N</text>
      <text x="106" y="214" font-size="14">S</text>
      <text x="206" y="114" font-size="14">E</text>
      <text x="8" y="114" font-size="14">W</text>
    </svg>
    """
    st.markdown(html, unsafe_allow_html=True)


def _estimate_runtime_minutes(n_epochs: int, train_days: int, test_days: int) -> float:
    return float(n_epochs) * float(train_days + test_days) * 0.3


def _render_eta(minutes: float) -> str:
    if minutes < 60:
        return f"Roughly {max(1, int(round(minutes)))} minutes"
    h = int(minutes // 60)
    m = int(round(minutes % 60))
    return f"Roughly {h} hr {m} min"


def _default_policy_learning_rate(base_cfg: dict[str, Any]) -> float:
    pol = base_cfg.get("policy") or {}
    if pol.get("learning_rate") is not None:
        return float(pol["learning_rate"])
    pc = base_cfg.get("policy_configuration") or {}
    default_blk = pc.get("default") or {}
    return float(default_blk.get("lr", 0.01))


def _parse_train_log_for_charts(line: str) -> None:
    """Update session training history from trainer log lines (episode + epoch summaries)."""
    if "mean_reward=" in line and "Train epoch=" in line:
        try:
            epoch = int(line.split("Train epoch=")[1].split()[0])
            mr = float(line.split("mean_reward=")[1].split()[0])
            st.session_state.training_epoch_history.append((epoch, mr))
        except Exception:
            return
        return
    if "Train" not in line or " reward=" not in line or "mean_reward=" in line:
        return
    try:
        rv = float(line.split(" reward=")[1].split()[0])
        ep = int(line.split("ep=")[1].split()[0])
        st.session_state.training_reward_history.append((ep, rv, "Train"))
    except Exception:
        pass


def _render_training_charts(ep_slot: Any, epoch_slot: Any, ui_cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Draw side-by-side matplotlib charts (dark style) into Streamlit placeholders."""
    ep_hist = [(e, r) for e, r, k in st.session_state.training_reward_history if k == "Train"]
    epoch_hist = list(st.session_state.get("training_epoch_history") or [])
    n_ep_total = max(1, int(ui_cfg["n_epochs"]) * int(ui_cfg["train_days"]))

    with plt.style.context("dark_background"):
        if ep_hist:
            fig1, ax1 = plt.subplots(figsize=(5.6, 3.4))
            xs = [t[0] for t in ep_hist]
            ys = [t[1] for t in ep_hist]
            ax1.plot(xs, ys, color="#4fc3f7", linewidth=1.6, alpha=0.95)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Total reward")
            ax1.set_title("Training reward by episode")
            ax1.set_xlim(left=0.5)
            ax1.grid(True, alpha=0.25)
            fig1.tight_layout()
            ep_slot.pyplot(fig1, clear_figure=True)
            plt.close(fig1)
        else:
            ep_slot.markdown("*Episode rewards will appear as training progresses.*")

        if epoch_hist:
            fig2, ax2 = plt.subplots(figsize=(5.6, 3.4))
            xs = [t[0] for t in epoch_hist]
            ys = [t[1] for t in epoch_hist]
            ax2.plot(xs, ys, color="#81c784", linewidth=1.6, marker="o", markersize=4, alpha=0.95)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mean total reward")
            ax2.set_title("Mean training reward by epoch")
            ax2.set_xlim(left=0.5)
            ax2.grid(True, alpha=0.25)
            fig2.tight_layout()
            epoch_slot.pyplot(fig2, clear_figure=True)
            plt.close(fig2)
        else:
            epoch_slot.markdown("*Epoch summaries appear after each training epoch.*")

    latest_ep = max((e for e, _, _ in st.session_state.training_reward_history), default=0)
    latest_epoch = max((e for e, _ in epoch_hist), default=0)
    return latest_ep, latest_epoch, n_ep_total


def _set_stage(i: int, status: str) -> None:
    st.session_state.current_stage = i
    st.session_state.stage_statuses[i] = status


def _build_config_from_ui(base_cfg: dict[str, Any], ui: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("paths", {})
    cfg.setdefault("training", {})
    cfg.setdefault("simulation", {})
    cfg.setdefault("flow_demand", {})
    cfg.setdefault("components", {})
    cfg.setdefault("output", {})

    cfg["paths"]["csv"] = "streamlit_runs/traffic.csv"
    cfg["paths"]["intersection"] = "streamlit_runs/intersection.json"
    cfg["paths"]["columns"] = "streamlit_runs/columns.json"
    cfg["paths"]["sumo_home"] = str(ui["sumo_home"])
    cfg["training"]["n_epochs"] = int(ui["n_epochs"])
    cfg["training"]["train_days"] = int(ui["train_days"])
    cfg["training"]["test_days"] = int(ui["test_days"])
    cfg["training"]["seed"] = int(ui["seed"])
    cfg["training"]["lr_min"] = float(ui["lr_min"])
    cfg["simulation"]["decision_gap"] = int(ui["decision_gap"])
    cfg["simulation"]["step_length"] = float(ui["step_length"])
    cfg["simulation"]["begin"] = int(ui["sim_begin"])
    cfg["simulation"]["end"] = int(ui["sim_end"])
    cfg["flow_demand"]["subslots_per_slot"] = int(ui["subslots_per_slot"])
    cfg["flow_demand"]["spread"] = float(ui["spread"])
    cfg["components"]["reward_class"] = str(ui["reward_class"])
    cfg["reward"] = {"class": str(ui["reward_class"])}
    cfg.setdefault("policy", {})
    cfg["policy"]["learning_rate"] = float(ui["learning_rate"])
    return cfg


def _run_full_pipeline(progress_cb: Callable[[str], None], ui: dict[str, Any]) -> None:
    import main as m

    st.session_state.pipeline_error_detail = ""
    st.session_state.stage_statuses = ["waiting"] * 8
    st.session_state.current_stage = 0
    st.session_state.eval_progress_rows = []
    st.session_state.last_schedule_webster_path = None
    st.session_state.last_comparison_csv_path = None
    st.session_state.last_run_baseline_comparison = bool(ui.get("run_baseline_comparison", False))

    stdout_parts: list[str] = []

    class _StdoutTee:
        __slots__ = ("_real", "_parts")

        def __init__(self, real, parts: list[str]) -> None:
            self._real = real
            self._parts = parts

        def write(self, s: str) -> int:
            if s:
                self._parts.append(s)
            return self._real.write(s)

        def flush(self) -> None:
            self._real.flush()

    _old_stdout = sys.stdout
    sys.stdout = _StdoutTee(sys.__stdout__, stdout_parts)

    def _drain_stdout_to_log() -> None:
        drain = "".join(stdout_parts)
        stdout_parts.clear()
        if drain.strip():
            st.session_state.training_log_lines.append(
                "[stdout]\n" + drain.rstrip()[-20000:]
            )

    def _log(msg: str) -> None:
        _drain_stdout_to_log()
        progress_cb(msg)

    try:
        traffic_path = RUNS / "traffic.csv"
        columns_path = RUNS / "columns.json"
        inter_path = RUNS / "intersection.json"

        traffic_path.write_bytes(st.session_state.csv_bytes or b"")
        columns_path.write_text(st.session_state.columns_text, encoding="utf-8")
        inter_path.write_text(st.session_state.intersection_text, encoding="utf-8")

        base_cfg = m._load_config_data(str(ROOT / "config.json"))
        cfg = _build_config_from_ui(base_cfg, ui)
        m._apply_config(cfg)

        split = None
        net_file = None
        run_dir = None
        results: Any = None

        def fail(idx: int, msg: str) -> None:
            _set_stage(idx, "failed")
            _log(msg)

        try:
            _set_stage(0, "active")
            _log("Preparing traffic flow files...")
            m.validate_inputs(ui["sumo_home"])
            split = m.run_build_route()
            _set_stage(0, "done")
        except Exception:
            st.session_state.pipeline_error_detail = traceback.format_exc()
            fail(0, "Could not prepare traffic flow files.")
            for i in range(1, 8):
                _set_stage(i, "skipped")
            return

        try:
            _set_stage(1, "active")
            _log("Building intersection road network...")
            net_file = m.run_build_network(ui["sumo_home"])
            _set_stage(1, "done")
        except Exception:
            st.session_state.pipeline_error_detail = traceback.format_exc()
            fail(1, "Could not build the intersection road network.")
            for i in range(2, 8):
                _set_stage(i, "skipped")
            return

        try:
            _set_stage(2, "active")
            _log("Training the signal control model...")
            reward_kwargs, _ = m._load_component_config(m.REWARD_CONFIG_PATH, m.RewardClass.__name__, "Reward")
            reward_kwargs = m.merge_reward_kwargs_from_config(reward_kwargs)
            policy_kwargs_raw, _ = m._load_component_config(m.POLICY_CONFIG_PATH, m.PolicyClass.__name__, "Policy")
            policy_kwargs_raw = m.merge_policy_kwargs_from_config(policy_kwargs_raw)
            policy_kwargs = m._resolve_policy_params(policy_kwargs_raw)
            run_dir = m._create_run_dir()
            m._write_config_summary(run_dir, False, reward_kwargs, policy_kwargs)
            trainer = m.build_pipeline(net_file, False, run_dir, reward_kwargs, policy_kwargs)

            def _cb(line: str) -> None:
                _log(line)
                _parse_train_log_for_charts(line)

            trainer.log_callback = _cb
            results = trainer.run()
            _set_stage(2, "done")
        except Exception:
            st.session_state.pipeline_error_detail = traceback.format_exc()
            fail(2, "Training could not complete.")
            for _sk in (3, 4, 5, 6, 7):
                _set_stage(_sk, "skipped")

        sched_path = None
        if st.session_state.stage_statuses[3] == "waiting":
            try:
                _set_stage(3, "active")
                _log("Saving trained model timing plan...")
                inter_cfg = json.loads(inter_path.read_text(encoding="utf-8"))
                sb = ScheduleBuilder()
                schedule = sb.build(list(results.get("phase_duration_records") or []), "J_centre", min_green_s=float(inter_cfg.get("min_green_s", 15)))
                sched_path = os.path.join(run_dir, "schedule.json")
                sb.write(schedule, sched_path)
                st.session_state.last_schedule_path = sched_path
                _set_stage(3, "done")
            except Exception:
                st.session_state.pipeline_error_detail = traceback.format_exc()
                fail(3, "Could not save the trained model timing plan.")
                for _sk in range(4, 8):
                    _set_stage(_sk, "skipped")

        compare = bool(ui.get("run_baseline_comparison", False))
        dqn_results: list = []
        web_results: list = []

        if not compare and st.session_state.stage_statuses[3] == "done":
            for i in range(4, 8):
                _set_stage(i, "skipped")
        elif compare and sched_path and run_dir and net_file:
            web_path = None
            try:
                _set_stage(4, "active")
                _log("Calculating mathematical baseline timing plan...")
                inter_cfg = json.loads(inter_path.read_text(encoding="utf-8"))
                days_dir = os.path.join(m.OUT_DIR, "processed", "days")
                ids = list((split or {}).get("test", []))
                day_paths = [os.path.join(days_dir, f"day_{int(d):02d}.csv") for d in ids]
                day_paths = [p for p in day_paths if os.path.isfile(p)]
                wb = WebsterScheduleBuilder(inter_cfg)
                ws = wb.build_from_day_csvs(day_paths, "J_centre", slot_minutes=int((split or {}).get("slot_minutes", 15)))
                web_path = os.path.join(run_dir, "schedule_webster.json")
                wb.write(ws, web_path)
                st.session_state.last_schedule_webster_path = web_path
                _set_stage(4, "done")
            except Exception:
                st.session_state.pipeline_error_detail = traceback.format_exc()
                fail(4, "Could not calculate the mathematical baseline timing plan.")
                _set_stage(6, "skipped")

            replay_out = os.path.join(run_dir or str(RUNS), "replay")
            os.makedirs(os.path.join(replay_out, "dqn"), exist_ok=True)
            os.makedirs(os.path.join(replay_out, "webster"), exist_ok=True)

            inter_cfg = json.loads(inter_path.read_text(encoding="utf-8"))
            runner = ReplayRunner(
                net_file=net_file,
                step_length=float(cfg["simulation"]["step_length"]),
                decision_gap=int(cfg["simulation"]["decision_gap"]),
                begin=int(cfg["simulation"]["begin"]),
                end=int(cfg["simulation"]["end"]),
                sumo_home=ui["sumo_home"],
                gui=False,
            )

            min_green = float(inter_cfg.get("min_green_s", 15))
            yellow = float(inter_cfg.get("amber_s", 3))
            min_red = float(inter_cfg.get("min_red_s", 1))

            if st.session_state.stage_statuses[5] == "waiting":
                try:
                    _set_stage(5, "active")
                    _log("Running trained model timing plan through simulation...")
                    ctl = ScheduleController(json.loads(Path(sched_path).read_text(encoding="utf-8")), "J_centre", min_green, yellow, float(cfg["simulation"]["step_length"]))
                    for day in list((split or {}).get("test", [])):
                        flow = os.path.join(m.OUT_DIR, "sumo", "flows", f"flows_day_{int(day):02d}.rou.xml")
                        if not os.path.isfile(flow):
                            continue
                        try:
                            res = runner.run(flow, ctl, min_green, yellow, min_red, "J_centre", "Trained model")
                            dqn_results.append((int(day), res))
                            Path(os.path.join(replay_out, "dqn", f"kpi_day_{int(day):02d}.json")).write_text(json.dumps(res.to_json_dict(), indent=2), encoding="utf-8")
                            st.session_state.eval_progress_rows.append({
                                "Day": int(day),
                                "Trained model - vehicles cleared": float(res.kpi_throughput_total),
                                "Mathematical baseline - vehicles cleared": None,
                            })
                        except Exception:
                            _log(f"A day failed during trained-model evaluation (day {day}). Continuing.")
                    _set_stage(5, "done")
                except Exception:
                    st.session_state.pipeline_error_detail = traceback.format_exc()
                    fail(5, "Evaluation failed while running the trained model timing plan.")

            if st.session_state.stage_statuses[6] == "waiting":
                try:
                    _set_stage(6, "active")
                    _log("Running mathematical baseline through simulation...")
                    ctl = ScheduleController(json.loads(Path(web_path).read_text(encoding="utf-8")), "J_centre", min_green, yellow, float(cfg["simulation"]["step_length"]))
                    idx = {int(r["Day"]): r for r in st.session_state.eval_progress_rows}
                    for day in list((split or {}).get("test", [])):
                        flow = os.path.join(m.OUT_DIR, "sumo", "flows", f"flows_day_{int(day):02d}.rou.xml")
                        if not os.path.isfile(flow):
                            continue
                        try:
                            res = runner.run(flow, ctl, min_green, yellow, min_red, "J_centre", "Mathematical baseline")
                            web_results.append((int(day), res))
                            Path(os.path.join(replay_out, "webster", f"kpi_day_{int(day):02d}.json")).write_text(json.dumps(res.to_json_dict(), indent=2), encoding="utf-8")
                            if int(day) in idx:
                                idx[int(day)]["Mathematical baseline - vehicles cleared"] = float(res.kpi_throughput_total)
                            else:
                                st.session_state.eval_progress_rows.append({
                                    "Day": int(day),
                                    "Trained model - vehicles cleared": None,
                                    "Mathematical baseline - vehicles cleared": float(res.kpi_throughput_total),
                                })
                        except Exception:
                            _log(f"A day failed during baseline evaluation (day {day}). Continuing.")
                    _set_stage(6, "done")
                except Exception:
                    st.session_state.pipeline_error_detail = traceback.format_exc()
                    fail(6, "Evaluation failed while running the mathematical baseline.")

            try:
                _set_stage(7, "active")
                _log("Comparing results...")
                bd = {d: r for d, r in dqn_results}
                bw = {d: r for d, r in web_results}
                days = sorted(set(bd) & set(bw))
                if days:
                    ReplayComparator().compare([bd[d] for d in days], [bw[d] for d in days], replay_out)
                    st.session_state.last_comparison_csv_path = os.path.join(replay_out, "comparison.csv")
                    _set_stage(7, "done")
                else:
                    _set_stage(7, "failed")
                    _log("Comparison could not complete because no paired day results were available.")
            except Exception:
                st.session_state.pipeline_error_detail = traceback.format_exc()
                _set_stage(7, "failed")
                _log("Could not finish the comparison step.")
        elif compare:
            for i in range(4, 8):
                _set_stage(i, "skipped")

        st.session_state.last_run_dir = run_dir
        st.session_state.last_test_log = list(results.get("test_log") or []) if results else None
    finally:
        try:
            _drain_stdout_to_log()
        except Exception:
            pass
        sys.stdout = _old_stdout


def _stage_view() -> None:
    names = [
        "Preparing traffic flow files",
        "Building intersection road network",
        "Training the signal control model",
        "Saving trained model timing plan",
        "Calculating mathematical baseline timing plan",
        "Running trained model timing plan through simulation",
        "Running mathematical baseline through simulation",
        "Comparing results",
    ]
    icon = {"waiting": "○", "active": "⏳", "done": "✓", "failed": "✗", "skipped": "–"}
    for i, name in enumerate(names):
        st.write(f"{icon.get(st.session_state.stage_statuses[i], '○')} {i+1:02d}. {name}")


def _render_what_happens_box() -> None:
    st.info(
        "What happens when you start\n\n"
        "Part 1 - Training\n"
        "  1) Prepare traffic flow files\n"
        "  2) Build the intersection road network\n"
        "  3) Train the signal control model\n"
        "  4) Save the trained model timing plan (schedule.json)\n\n"
        "Part 2 - Optional baseline comparison (enable under **Advanced**)\n"
        "  When enabled: build a Webster (mathematical) baseline schedule, replay both schedules "
        "in simulation on your evaluation days, and produce KPI comparison results.\n"
        "  When disabled: the run stops after saving your RL-derived schedule.json."
    )


def _download_file(path: str | None, label: str, key: str, mime: str) -> None:
    if not path or not os.path.isfile(path):
        st.caption("Not available yet.")
        return
    st.download_button(label=label, data=Path(path).read_bytes(), file_name=os.path.basename(path), mime=mime, key=key)


def main() -> None:
    st.set_page_config(
        page_title="RL Traffic Signal Optimizer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    _new_session_cleanup()
    st.session_state.setdefault(
        "ui_sumo_home",
        os.environ.get("SUMO_HOME", "") or _try_eclipse_sumo_home(),
    )

    st.sidebar.header("Environment")
    st.sidebar.text_input(
        "SUMO_HOME",
        key="ui_sumo_home",
        help="Folder that contains bin/netconvert (e.g. Eclipse SUMO install). Required for network build and simulation.",
    )
    st.sidebar.caption(
        "Training runs the same code paths as `main.py`. Leave this tab open until the run finishes."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Quick flow:** 1) Data → 2) Intersection → 3) Advanced (optional) → 4) Run → 5) Results."
    )

    import main as m
    base_cfg = m._load_config_data(str(ROOT / "config.json"))
    policy_lr_default = _default_policy_learning_rate(base_cfg)
    train_cfg = base_cfg.get("training", {})
    sim_cfg = base_cfg.get("simulation", {})
    phase_cfg = base_cfg.get("phase_timing", {})
    flow_cfg = base_cfg.get("flow_demand", {})
    default_reward = base_cfg.get("components", {}).get("reward_class", "ThroughputCompositeReward")
    reward_idx = REWARD_CLASS_OPTIONS.index(default_reward) if default_reward in REWARD_CLASS_OPTIONS else 0

    st.title("RL Traffic Signal Optimizer")
    st.caption(
        "Upload a traffic count file each session (nothing is auto-loaded from `src/data/`). "
        "Configure mapping and the intersection, then run the full pipeline: training, schedules, and evaluation."
    )

    tabs = st.tabs(
        [
            "1 · Data & map",
            "2 · Intersection",
            "3 · Advanced",
            "4 · Run",
            "5 · Results",
        ]
    )

    with tabs[0]:
        st.subheader("Upload your traffic count file")
        up = st.file_uploader("Upload a traffic count file", type=["csv", "xlsx"], key="uploader_traffic", help="Upload your count data as CSV or Excel.")
        if up is None and st.session_state.csv_df is None:
            st.info("Upload a traffic count file to begin.")
        else:
            if up is not None:
                st.session_state.csv_bytes = up.getvalue()
                try:
                    if up.name.lower().endswith(".xlsx"):
                        st.session_state.csv_df = pd.read_excel(io.BytesIO(st.session_state.csv_bytes))
                    else:
                        st.session_state.csv_df = pd.read_csv(io.BytesIO(st.session_state.csv_bytes))
                    st.success("✓ File loaded successfully.")
                except Exception:
                    st.warning("Could not read that file. Please upload a valid CSV or Excel file.")

            df = st.session_state.csv_df
            if df is not None:
                st.dataframe(df.head(5), width="stretch", key="df_preview_5")
                cands = [c for c in df.columns if "time" in str(c).lower()]
                start_guess = cands[0] if cands else None
                detected = _detect_slot_minutes(df, start_guess)
                st.caption(f"Detected count interval: every {detected} minutes")
                st.session_state.slot_minutes = st.number_input(
                    "Count interval (minutes)",
                    min_value=1,
                    max_value=60,
                    value=int(detected),
                    key="slot_minutes_override",
                    help="How many minutes each row of counts represents.",
                )

        st.markdown("---")
        st.subheader("Tell us what each column in your file represents")
        df = st.session_state.csv_df
        active_for_mapping = DIRECTIONS
        if st.session_state.tab2_valid and st.session_state.intersection_text:
            try:
                active_for_mapping = json.loads(st.session_state.intersection_text).get("active_approaches") or DIRECTIONS
            except Exception:
                active_for_mapping = DIRECTIONS

        if df is None:
            st.warning("Upload a file first so we can map your columns.")
        else:
            cols = [str(c) for c in df.columns]
            suggestion = suggest_columns_json_from_dataframe(df, active_for_mapping)
            t_sug = suggestion.get("time", {})
            a_sug = suggestion.get("approaches", {})
            opt_none = ["(none)"] + cols

            st.markdown("**Time columns**")
            c1, c2 = st.columns(2)
            with c1:
                start_col = st.selectbox("Start time column", opt_none, index=opt_none.index(t_sug.get("start_time")) if t_sug.get("start_time") in opt_none else 0, key="map_start_time", help="Column that marks start time for each count row.")
                end_col = st.selectbox("End time column", opt_none, index=opt_none.index(t_sug.get("end_time")) if t_sug.get("end_time") in opt_none else 0, key="map_end_time", help="Optional end-time column.")
            with c2:
                date_col = st.selectbox("Date column", opt_none, index=opt_none.index(t_sug.get("date")) if t_sug.get("date") in opt_none else 0, key="map_date", help="Optional date column.")
                dow_col = st.selectbox("Day of week column", opt_none, index=opt_none.index(t_sug.get("day_of_week")) if t_sug.get("day_of_week") in opt_none else 0, key="map_dow", help="Optional day-of-week column.")

            st.markdown("**Traffic counts**")
            mapped_through = 0
            map_obj = {
                "time": {
                    "start_time": None if start_col == "(none)" else start_col,
                    "end_time": None if end_col == "(none)" else end_col,
                    "date": None if date_col == "(none)" else date_col,
                    "day_of_week": None if dow_col == "(none)" else dow_col,
                },
                "approaches": {},
                "slot_minutes": int(st.session_state.slot_minutes),
            }
            move_opts = ["Not in data"] + cols
            for d in active_for_mapping:
                st.markdown(f"**{DIR_LABEL[d]}bound traffic**")
                sug = a_sug.get(d, {})
                c1, c2, c3 = st.columns(3)
                t_def = sug.get("through") if sug.get("through") in cols else "Not in data"
                r_def = sug.get("right") if sug.get("right") in cols else "Not in data"
                l_def = sug.get("left") if sug.get("left") in cols else "Not in data"
                with c1:
                    v_t = st.selectbox("Vehicles going straight", move_opts, index=move_opts.index(t_def), key=f"map_{d}_t", help="Column with straight-through counts for this direction.")
                    if t_def != "Not in data":
                        st.caption("(auto-detected)")
                with c2:
                    v_r = st.selectbox("Vehicles turning right", move_opts, index=move_opts.index(r_def), key=f"map_{d}_r", help="Column with right-turn counts for this direction.")
                    if r_def != "Not in data":
                        st.caption("(auto-detected)")
                with c3:
                    v_l = st.selectbox("Vehicles turning left", move_opts, index=move_opts.index(l_def), key=f"map_{d}_l", help="Column with left-turn counts for this direction.")
                    if l_def != "Not in data":
                        st.caption("(auto-detected)")

                map_obj["approaches"][d] = {"through": None if v_t == "Not in data" else v_t, "right": None if v_r == "Not in data" else v_r, "left": None if v_l == "Not in data" else v_l, "peds": None}
                if v_t != "Not in data":
                    mapped_through += 1

            if mapped_through > 0:
                st.success("✓ Column mapping complete")
            else:
                st.warning("⚠ Please map at least one straight-movement column to continue.")

            if st.button("Save column mapping", key="btn_save_col_map", help="Save this mapping so Training can use it."):
                if mapped_through == 0:
                    st.warning("Please map at least one straight-movement column before saving.")
                else:
                    st.session_state.columns_text = json.dumps(map_obj, indent=2)
                    st.session_state.tab1_valid = True
                    st.success("✓ Column mapping saved.")

            with st.expander("Edit raw JSON (for advanced users)"):
                txt = st.text_area(
                    "Column mapping JSON",
                    value=st.session_state.columns_text or json.dumps(map_obj, indent=2),
                    height=220,
                    key="col_json_editor",
                    help="Advanced: edit the raw mapping JSON directly.",
                )
                if txt != st.session_state.columns_text:
                    st.session_state.columns_text = txt

        st.markdown("---")
        st.subheader("Which part of the day should the simulation cover?")
        c1, c2 = st.columns(2)
        with c1:
            sim_start = st.time_input("Start of simulation", value=dtime(8, 0), key="ui_sim_start", help="When the simulation day begins.")
        with c2:
            sim_end = st.time_input("End of simulation", value=dtime(18, 0), key="ui_sim_end", help="When the simulation day ends.")
        b = _time_to_seconds(sim_start)
        e = _time_to_seconds(sim_end)
        if e <= b:
            st.warning("End time must be after start time.")
        else:
            st.caption(f"Each simulated day will cover {(e - b) / 3600.0:.1f} hours of traffic data.")

    with tabs[1]:
        st.subheader("Intersection type")
        mode = st.radio("What kind of intersection is this?", ["Four-way (+ shape)", "T-intersection"], horizontal=True, key="ui_intersection_kind", help="Choose whether roads exist in all four directions or one is missing.")
        active = list(DIRECTIONS)
        if mode == "T-intersection":
            miss = st.radio("Which direction has no road?", ["No road to the north", "No road to the south", "No road to the east", "No road to the west"], key="ui_missing_leg", help="Pick the direction that does not exist.")
            dmap = {"No road to the north": "N", "No road to the south": "S", "No road to the east": "E", "No road to the west": "W"}
            active = [d for d in DIRECTIONS if d != dmap[miss]]

        _render_intersection_svg(active)
        st.markdown("---")
        st.subheader("How many lanes does each approach have?")
        lanes_per = {}
        for d in active:
            st.markdown(f"**{DIR_LABEL[d]} approach**")
            c1, c2, c3 = st.columns(3)
            with c1:
                th = st.number_input("Through lanes", 1, 6, 2, key=f"ui_lane_{d}_through", help="Number of through lanes for this approach.")
            with c2:
                rt = st.number_input("Right-turn lane", 0, 2, 1, key=f"ui_lane_{d}_right", help="Number of dedicated right-turn lanes.")
            with c3:
                lf = st.number_input("Left-turn lane", 0, 2, 1, key=f"ui_lane_{d}_left", help="Number of dedicated left-turn lanes.")
            lanes_per[d] = {"through": int(th), "right": int(rt), "left": int(lf)}

        same_speed = st.checkbox("All approaches share the same speed limit", value=True, key="ui_same_speed", help="Use one speed setting for every direction.")
        if same_speed:
            speed_input = float(st.number_input("Speed limit (km/h)", 20.0, 120.0, 50.0, key="ui_speed_all", help="Speed limit used for all approaches."))
        else:
            speed_map = {}
            for d in active:
                speed_map[d] = float(st.number_input(f"{DIR_LABEL[d]} speed limit (km/h)", 20.0, 120.0, 50.0, key=f"ui_speed_{d}", help="Speed limit for this approach."))
            speed_input = speed_map

        st.markdown("---")
        st.subheader("How are left turns handled at this intersection?")
        lt_choice = st.selectbox("Left turn signal type", ["Yield on green - left turns share the main green phase (most common)", "Protected arrow - all left turns get their own dedicated signal", "Mixed - only some directions have a dedicated left-turn signal"], key="ui_lt_mode", help="Choose permissive, protected, or mixed left-turn handling.")
        lt_mode = {
            "Yield on green - left turns share the main green phase (most common)": "permissive",
            "Protected arrow - all left turns get their own dedicated signal": "protected",
            "Mixed - only some directions have a dedicated left-turn signal": "protected_some",
        }[lt_choice]
        protected_some = []
        if lt_mode == "protected_some":
            for d in active:
                if lanes_per[d]["left"] > 0:
                    if st.checkbox(f"{DIR_LABEL[d]} has a dedicated left-turn arrow", key=f"ui_protected_{d}", help="Enable a dedicated left-turn signal for this approach."):
                        protected_some.append(d)
                else:
                    st.checkbox(f"{DIR_LABEL[d]} has a dedicated left-turn arrow", value=False, disabled=True, key=f"ui_protected_{d}", help="This approach has no left-turn lane.")

        st.markdown("---")
        st.subheader("Set the timing rules for this intersection")
        min_green = st.number_input("Minimum green light duration (seconds)", 5, 60, int(phase_cfg.get("min_green_s", 15)), key="ui_min_green_s", help="The signal must stay green at least this long before it can switch.")
        max_green = st.number_input("Maximum green light duration (seconds)", 15, 180, int(phase_cfg.get("max_green_s", 90)), key="ui_max_green_s", help="The signal will not hold green longer than this.")
        amber = st.number_input("Yellow light duration (seconds)", 3, 5, 3, key="ui_amber_s", help="Duration of amber before red.")
        min_red = st.number_input("All-red clearance time (seconds)", 1, 5, 1, key="ui_min_red_s", help="Brief all-red time between phases for safety.")
        cycle = st.number_input("Target signal cycle length (seconds)", 60, 240, 120, key="ui_cycle_s", help="Intended cycle length reference.")
        inter_name = st.text_input("Intersection name", value=suggest_intersection_name(st.session_state.csv_df) if st.session_state.csv_df is not None else "My_Intersection", key="ui_inter_name", help="Used to name outputs. Use letters, numbers, and underscores.")
        edge_len = st.number_input("Approach road length (metres)", 50, 500, 200, key="ui_edge_len", help="How far each approach road extends in simulation.")

        st.markdown("---")
        st.subheader("Confirm and save")
        lt_desc = "Left turns share the main green phase." if lt_mode == "permissive" else ("All approaches use dedicated left-turn arrows." if lt_mode == "protected" else ("Dedicated arrows for: " + ", ".join(DIR_LABEL[d] for d in protected_some) if protected_some else "Mixed mode selected, but no dedicated arrows chosen yet."))
        lane_line = " | ".join(f"{DIR_LABEL[d]}: {lanes_per[d]['through']} straight · {lanes_per[d]['right']} right · {lanes_per[d]['left']} left" for d in active)
        st.info(f"Intersection: {inter_name} · {mode}\nDirections: {', '.join(DIR_LABEL[d] for d in active)}\nLanes: {lane_line}\nLeft turns: {lt_desc}\nTiming: Cycle {cycle}s · Min green {min_green}s · Yellow {amber}s · All-red {min_red}s")

        if st.button("Save intersection configuration", key="btn_save_intersection", help="Save this intersection setup so Training can run."):
            try:
                obj = build_intersection_dict_from_ui(
                    intersection_name=inter_name,
                    active_approaches=active,
                    lanes_per_approach={d: {"lanes": lanes_per[d]} for d in active},
                    speed_kmh=speed_input,
                    left_turn_mode=lt_mode,
                    protected_approaches=protected_some,
                    min_green_s=int(min_green),
                    max_green_s=int(max_green),
                    amber_s=int(amber),
                    min_red_s=int(min_red),
                    cycle_s=int(cycle),
                    edge_length_m=int(edge_len),
                )
                st.session_state.intersection_text = json.dumps(obj, indent=2)
                st.session_state.tab2_valid = True
                st.success("✓ Intersection configuration saved.")
            except Exception:
                st.warning("Could not save intersection settings. Please review your inputs.")

    with tabs[2]:
        st.subheader("Advanced configuration")
        st.caption(
            "Technical controls for reward shaping, optimization, demand generation, "
            "and simulator/runtime settings. Defaults are tuned for stable training."
        )
        with st.expander("Reward function", expanded=False):
            rc = st.selectbox(
                "Reward function",
                list(REWARD_CLASS_OPTIONS),
                index=reward_idx,
                key="ui_reward_class",
                help="Python reward class (see `modelling.components.reward` and `config.json` → `reward_configuration`).",
            )
            st.caption("Hyperparameters merge with `reward_configuration` in config.json for the selected class.")
            if rc == "ThroughputCompositeReward":
                st.slider("alpha (queue penalty)", 0.0, 1.0, 0.3, 0.01, key="rw_alpha", help="Weight applied to queue-related penalty term.")
                st.slider("beta (lane imbalance penalty)", 0.0, 0.5, 0.1, 0.01, key="rw_beta", help="Penalty for inter-lane imbalance.")
                st.slider("gamma (throughput reward)", 0.0, 2.0, 1.0, 0.01, key="rw_gamma", help="Weight applied to departure/throughput gain.")
                st.slider("switch_weight (phase change cost)", 0.0, 1.0, 0.1, 0.01, key="rw_switch_weight", help="Penalty for switching to reduce oscillatory control.")
                st.checkbox("normalize", value=True, key="rw_norm", help="Apply reward normalization.")
            elif rc == "ThroughputQueueReward":
                st.slider("throughput_weight", 0.0, 1.0, 0.5, 0.01, key="rw_throughput_weight", help="Weight for throughput term.")
                st.slider("queue_weight", 0.0, 1.0, 0.25, 0.01, key="rw_queue_weight", help="Weight for queue penalty term.")
                st.slider("switch_weight", 0.0, 1.0, 0.5, 0.01, key="rw_switch_weight", help="Cost applied on phase transitions.")
                st.checkbox("normalize", value=True, key="rw_norm", help="Apply reward normalization.")
            elif rc == "WaitingTimeReward":
                st.slider("switch_weight", 0.0, 1.0, 0.5, 0.01, key="rw_switch_weight", help="Cost applied on phase transitions.")
                st.checkbox("normalize", value=True, key="rw_norm", help="Apply reward normalization.")
            elif rc == "ThroughputWaitTimeReward":
                st.slider("alpha (wait/queue penalty)", 0.0, 1.0, 0.3, 0.01, key="rw_alpha", help="Weight for waiting-time / queue penalties.")
                st.slider("beta (imbalance penalty)", 0.0, 0.5, 0.1, 0.01, key="rw_beta", help="Penalty for directional imbalance.")
                st.slider("gamma (throughput reward)", 0.0, 2.0, 1.0, 0.01, key="rw_gamma", help="Weight for throughput gain.")
                st.checkbox("normalize", value=True, key="rw_norm", help="Apply reward normalization.")
            elif rc == "CompositeReward":
                st.slider("alpha (composite blend)", 0.0, 1.0, 0.65, 0.01, key="rw_alpha", help="Blend coefficient across composite objective terms.")
                st.checkbox("normalize", value=True, key="rw_norm", help="Apply reward normalization.")
            elif rc == "ThroughputReward":
                st.checkbox("normalize", value=False, key="rw_norm", help="Apply reward normalization (see config.json ThroughputReward).")
            elif rc in ("VehicleCountReward", "DeltaVehicleCountReward", "DeltaWaitingTimeReward"):
                st.caption("No extra sliders here; tune under `reward_configuration` in config.json for this class.")
            else:
                st.caption("Optional parameters for this class are in config.json → `reward_configuration`.")

        with st.expander("Training duration", expanded=False):
            st.number_input("n_epochs", 1, 2000, int(train_cfg.get("n_epochs", 200)), key="ui_epochs", help="Number of training epochs/episodes.")
            st.number_input("train_days", 1, 100, int(train_cfg.get("train_days", 5)), key="ui_train_days", help="Number of days used for optimization.")
            st.number_input("test_days", 1, 50, int(train_cfg.get("test_days", 5)), key="ui_test_days", help="Holdout days for post-train evaluation.")
            labels = {1: "Every second (most responsive)", 5: "Every 5 seconds", 10: "Every 10 seconds (recommended)", 15: "Every 15 seconds", 30: "Every 30 seconds (least reactive)"}
            picked = st.selectbox("decision_gap (s)", list(labels.values()), index=2, key="ui_decision_gap_lbl", help="Control interval for policy action selection.")
            st.session_state.ui_decision_gap = [k for k, v in labels.items() if v == picked][0]
            st.number_input("seed", value=int(train_cfg.get("seed", 42)), key="ui_seed", help="Random seed for reproducibility.")
            st.number_input(
                "Initial learning rate (Adam)",
                min_value=0.000001,
                max_value=1.0,
                value=float(policy_lr_default),
                step=0.0001,
                format="%.6f",
                key="ui_learning_rate",
                help="Starting learning rate for the Q-network; merged into config as policy.learning_rate.",
            )
            st.number_input(
                "Minimum LR (scheduler floor)",
                min_value=0.000001,
                max_value=0.01,
                value=float(train_cfg.get("lr_min", 0.0001)),
                step=0.000001,
                format="%.6f",
                key="ui_lr_min",
                help="Lower bound for the learning-rate schedule.",
            )

        with st.expander("Baseline comparison (Webster + replay)", expanded=False):
            st.checkbox(
                "Run Webster baseline and compare in simulation",
                value=False,
                key="ui_run_baseline_comparison",
                help="When enabled: builds schedule_webster.json, runs both schedules on evaluation days, and writes KPI comparison (replay/comparison.csv). When disabled: only the RL schedule.json is produced.",
            )

        with st.expander("Traffic demand variation", expanded=False):
            use_var = st.checkbox("Enable stochastic intra-slot demand shaping", value=True, key="ui_use_variation", help="Enable stochastic demand sampling within each aggregated time slot.")
            if use_var:
                st.slider("spread", 0.0, 1.0, float(flow_cfg.get("spread", 0.85)), 0.01, key="ui_spread", help="Spread parameter for sampled arrivals (higher = more variation).")
                st.selectbox("subslots_per_slot", [1, 2, 3, 5, 10], index=3, key="ui_subslots", help="Temporal sub-discretization factor per demand slot.")
            else:
                st.session_state.ui_spread = 0.0
                st.session_state.ui_subslots = 1

        with st.expander("Simulation engine", expanded=False):
            st.caption("SUMO path is set in the **sidebar** under Environment.")
            step_map = {"0.5 seconds - fine-grained": 0.5, "1 second - recommended": 1.0, "2 seconds - faster, less precise": 2.0}
            step_label = st.selectbox("step_length (s)", list(step_map.keys()), index=1, key="ui_step_label", help="SUMO integration timestep in seconds.")
            st.session_state.ui_step_length = step_map[step_label]
            cap_map = {"Small (4,000)": 4000, "Medium (8,000) - recommended": 8000, "Large (16,000)": 16000, "Extra large (32,768)": 32768}
            cap_label = st.selectbox("replay_buffer_capacity", list(cap_map.keys()), index=1, key="ui_buffer_label", help="Replay buffer capacity (only applicable if policy uses experience replay).")
            st.session_state.ui_capacity = cap_map[cap_label]

    with tabs[3]:
        if not (st.session_state.tab1_valid and st.session_state.tab2_valid):
            st.info("Complete the Data and Intersection tabs before running.")
        else:
            try:
                inter = json.loads(st.session_state.intersection_text)
            except Exception:
                inter = {}

            n_epochs = int(st.session_state.get("ui_epochs", train_cfg.get("n_epochs", 200)))
            tr_days = int(st.session_state.get("ui_train_days", train_cfg.get("train_days", 5)))
            te_days = int(st.session_state.get("ui_test_days", train_cfg.get("test_days", 5)))
            eta = _estimate_runtime_minutes(n_epochs, tr_days, te_days)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Intersection summary**")
                st.write(f"Intersection: {inter.get('intersection_name','(not set)')} ({'T-intersection' if len(inter.get('active_approaches',[]))==3 else 'Four-way'})")
                st.write(f"Directions: {', '.join(DIR_LABEL[d] for d in inter.get('active_approaches', []))}")
                lt = inter.get("left_turn_mode", "permissive")
                lt_txt = {"permissive": "Yield on green", "protected": "Protected arrows", "protected_some": "Mixed"}.get(lt, "Yield on green")
                st.write(f"Left turns: {lt_txt}")
                st.write(f"Simulation: {_seconds_to_hhmm(_time_to_seconds(st.session_state.ui_sim_start))} to {_seconds_to_hhmm(_time_to_seconds(st.session_state.ui_sim_end))}")
            with c2:
                st.markdown("**Run summary**")
                st.write(f"Reward: {st.session_state.get('ui_reward_class', default_reward)}")
                st.write(f"Training days: {tr_days}")
                st.write(f"Evaluation days: {te_days}")
                st.write(f"Training rounds: {n_epochs}")
                st.write(f"Estimated time: {_render_eta(eta)} (estimate - actual time depends on your hardware)")

            _render_what_happens_box()

            ui_cfg = {
                "n_epochs": n_epochs,
                "train_days": tr_days,
                "test_days": te_days,
                "decision_gap": int(st.session_state.get("ui_decision_gap", sim_cfg.get("decision_gap", 10))),
                "seed": int(st.session_state.get("ui_seed", train_cfg.get("seed", 42))),
                "learning_rate": float(st.session_state.get("ui_learning_rate", policy_lr_default)),
                "lr_min": float(st.session_state.get("ui_lr_min", train_cfg.get("lr_min", 0.0001))),
                "step_length": float(st.session_state.get("ui_step_length", sim_cfg.get("step_length", 1.0))),
                "sim_begin": int(_time_to_seconds(st.session_state.ui_sim_start)),
                "sim_end": int(_time_to_seconds(st.session_state.ui_sim_end)),
                "subslots_per_slot": int(st.session_state.get("ui_subslots", flow_cfg.get("subslots_per_slot", 5))),
                "spread": float(st.session_state.get("ui_spread", flow_cfg.get("spread", 0.85))),
                "reward_class": str(st.session_state.get("ui_reward_class", default_reward)),
                "sumo_home": st.session_state.get("ui_sumo_home", os.environ.get("SUMO_HOME", "")),
                "run_baseline_comparison": bool(st.session_state.get("ui_run_baseline_comparison", False)),
            }

            start_disabled = st.session_state.run_in_progress or (ui_cfg["sim_end"] <= ui_cfg["sim_begin"])
            if st.button("▶  Start", type="primary", disabled=start_disabled, key="btn_start_pipeline", help="Runs training and saves schedule.json. Optional Webster/replay comparison is controlled under Advanced."):
                st.session_state.run_in_progress = True
                st.session_state.run_complete = False
                st.session_state.training_log_lines = []
                st.session_state.training_reward_history = []
                st.session_state.training_epoch_history = []
                st.session_state.pipeline_error_detail = ""

                log_box = st.empty()
                chart_row = st.container()
                with chart_row:
                    st.markdown("**Training curves**")
                    c_ep, c_epoch = st.columns(2)
                    with c_ep:
                        ep_chart_ph = st.empty()
                    with c_epoch:
                        epoch_chart_ph = st.empty()
                stats_box = st.empty()
                table_box = st.empty()

                def progress_cb(msg: str) -> None:
                    """Append status; update live widgets without fixed keys (avoids duplicate-widget errors)."""
                    st.session_state.training_log_lines.append(str(msg))
                    text = "\n".join(st.session_state.training_log_lines[-400:])
                    try:
                        log_box.code(text, language=None)
                    except Exception:
                        pass
                    try:
                        latest_ep, latest_epoch, n_ep_total = _render_training_charts(ep_chart_ph, epoch_chart_ph, ui_cfg)
                        if latest_ep or latest_epoch:
                            stats_box.info(
                                f"Training progress: episode **{latest_ep}** / {n_ep_total} · "
                                f"epoch **{latest_epoch}** / {ui_cfg['n_epochs']}"
                            )
                        if st.session_state.eval_progress_rows:
                            table_box.dataframe(pd.DataFrame(st.session_state.eval_progress_rows), width="stretch")
                    except Exception:
                        pass

                try:
                    _run_full_pipeline(progress_cb, ui_cfg)
                    if "failed" in st.session_state.stage_statuses:
                        idx = st.session_state.stage_statuses.index("failed")
                        names = ["Preparing traffic flow files", "Building intersection road network", "Training the signal control model", "Saving trained model timing plan", "Calculating mathematical baseline timing plan", "Running trained model timing plan through simulation", "Running mathematical baseline through simulation", "Comparing results"]
                        st.error(f"Something went wrong during [{names[idx]}].")
                    else:
                        st.balloons()
                        if st.session_state.get("last_run_baseline_comparison"):
                            st.success("✓ Done! Switch to the **Results** tab for KPI comparison and downloads.")
                        else:
                            st.success("✓ Done! Your RL **schedule.json** is ready — open the **Results** tab to download it.")
                        st.session_state.run_complete = True
                except Exception:
                    st.session_state.pipeline_error_detail = traceback.format_exc()
                    st.error("Something went wrong while running the pipelines.")
                finally:
                    st.session_state.run_in_progress = False

            st.markdown("### Live progress")
            _stage_view()
            if st.session_state.eval_progress_rows:
                st.dataframe(pd.DataFrame(st.session_state.eval_progress_rows), width="stretch")
            with st.expander("Show detailed technical log", expanded=False):
                st.caption("Includes status lines, captured [stdout] from the pipeline, and Python tracebacks when available.")
                if st.session_state.get("pipeline_error_detail"):
                    st.code(st.session_state.pipeline_error_detail, language="text")
                st.text_area(
                    "Log output",
                    value="\n".join(st.session_state.training_log_lines[-800:]),
                    height=280,
                    key="ta_logs_view",
                    help="Progress messages and captured stdout from main.py / SUMO steps.",
                )

    with tabs[4]:
        if not st.session_state.run_complete:
            st.info("Run training from the **Run** tab first to unlock results.")
        else:
            run_dir = st.session_state.last_run_dir
            sched_path = st.session_state.last_schedule_path
            had_baseline = bool(st.session_state.get("last_run_baseline_comparison"))
            comp_path = st.session_state.last_comparison_csv_path

            if not run_dir or not sched_path or not os.path.isfile(sched_path):
                st.warning("Results are not available yet.")
            elif had_baseline and comp_path and os.path.isfile(comp_path):
                comp = pd.read_csv(comp_path)
                by = {r["metric"]: r for _, r in comp.iterrows()}

                st.subheader("How did your trained model compare to the mathematical baseline?")
                td = int(st.session_state.get("ui_test_days", 5))
                st.caption(f"Results measured on {td} evaluation days - data the model never saw during training.")

                c1, c2, c3, c4 = st.columns(4)
                m1 = by.get("Throughput total (veh)", {})
                m2 = by.get("Throughput rate (veh/s)", {})
                m3 = by.get("Neg lane wait integral", {})
                m4 = by.get("Phase switches", {})
                c1.metric("Vehicles cleared per day", f"{float(m1.get('dqn_mean',0)):,.0f}", f"{float(m1.get('delta_mean',0)):+,.0f}")
                c2.metric("Clearance rate (vehicles/second)", f"{float(m2.get('dqn_mean',0)):.3f}", f"{float(m2.get('delta_mean',0)):+.3f}")
                c3.metric("Wait time score (higher = less waiting)", f"{float(m3.get('dqn_mean',0)):,.0f}", f"{float(m3.get('delta_mean',0)):+,.0f}")
                c4.metric("Signal phase changes per day", f"{float(m4.get('dqn_mean',0)):.0f}", f"{float(m4.get('delta_mean',0)):+.0f}")
                c4.caption("Informational - neither fewer nor more is necessarily better.")

                wins = int(float(m1.get("delta_mean", 0)) > 0) + int(float(m2.get("delta_mean", 0)) > 0) + int(float(m3.get("delta_mean", 0)) > 0)
                if wins >= 2:
                    st.success(f"Your trained model outperformed the mathematical baseline on {wins} out of 3 performance metrics across {td} evaluation days.")
                else:
                    st.info("The mathematical baseline matched or outperformed your model on some metrics. Consider more training rounds or a different reward class in **Advanced**.")

                st.markdown("---")
                st.subheader("Performance across evaluation days")
                dqn_dir = Path(run_dir) / "replay" / "dqn"
                web_dir = Path(run_dir) / "replay" / "webster"
                drows = []
                wrows = []
                for p in sorted(dqn_dir.glob("kpi_day_*.json")):
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    drows.append({"Day": int(p.stem.split("_")[-1]), **obj})
                for p in sorted(web_dir.glob("kpi_day_*.json")):
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    wrows.append({"Day": int(p.stem.split("_")[-1]), **obj})
                ddf = pd.DataFrame(drows).sort_values("Day") if drows else pd.DataFrame()
                wdf = pd.DataFrame(wrows).sort_values("Day") if wrows else pd.DataFrame()
                t1, t2 = st.tabs(["Trained model", "Mathematical baseline"])
                with t1:
                    if not ddf.empty:
                        st.line_chart(ddf.set_index("Day")["kpi_throughput_total"], key="line_dqn_through")
                        st.line_chart(ddf.set_index("Day")["kpi_neg_lane_waiting_integral"], key="line_dqn_wait")
                        st.dataframe(ddf.rename(columns={"kpi_throughput_total": "Vehicles cleared", "kpi_throughput_rate": "Clearance rate (vehicles/second)", "kpi_neg_lane_waiting_integral": "Wait time score (higher = less waiting)", "phase_switch_count": "Signal phase changes"}), width="stretch", key="df_dqn_days")
                with t2:
                    if not wdf.empty:
                        st.line_chart(wdf.set_index("Day")["kpi_throughput_total"], key="line_web_through")
                        st.line_chart(wdf.set_index("Day")["kpi_neg_lane_waiting_integral"], key="line_web_wait")
                        st.dataframe(wdf.rename(columns={"kpi_throughput_total": "Vehicles cleared", "kpi_throughput_rate": "Clearance rate (vehicles/second)", "kpi_neg_lane_waiting_integral": "Wait time score (higher = less waiting)", "phase_switch_count": "Signal phase changes"}), width="stretch", key="df_web_days")

                with st.expander("View signal timing plans", expanded=False):
                    sp = st.session_state.last_schedule_path
                    wp = st.session_state.last_schedule_webster_path
                    if sp and wp and os.path.isfile(sp) and os.path.isfile(wp):
                        d_sched = json.loads(Path(sp).read_text(encoding="utf-8")).get("buckets", [])
                        w_sched = json.loads(Path(wp).read_text(encoding="utf-8")).get("buckets", [])
                        phases = sorted(set(int(r["phase"]) for r in d_sched) | set(int(r["phase"]) for r in w_sched))
                        ptabs = st.tabs([f"Phase {p+1}" for p in phases])
                        for i, ph in enumerate(phases):
                            with ptabs[i]:
                                dph = [r for r in d_sched if int(r["phase"]) == ph]
                                wph = [r for r in w_sched if int(r["phase"]) == ph]
                                dfp = pd.DataFrame({"Time of day": [_seconds_to_hhmm(int(r["bucket_start_s"])) for r in dph], "Trained model (s)": [float(r["median_s"]) for r in dph], "Trained model variation": [float(r.get("std_s", 0)) for r in dph]}) if dph else pd.DataFrame()
                                if wph:
                                    wf = pd.DataFrame({"Time of day": [_seconds_to_hhmm(int(r["bucket_start_s"])) for r in wph], "Mathematical baseline (s)": [float(r["median_s"]) for r in wph], "Mathematical baseline variation": [float(r.get("std_s", 0)) for r in wph]})
                                    dfp = wf if dfp.empty else dfp.merge(wf, on="Time of day", how="outer")
                                if not dfp.empty:
                                    cols = [c for c in ["Trained model (s)", "Mathematical baseline (s)"] if c in dfp.columns]
                                    st.line_chart(dfp.set_index("Time of day")[cols], key=f"line_phase_{ph}")
                                    st.dataframe(dfp, width="stretch", key=f"df_phase_{ph}")

                st.markdown("---")
                st.subheader("Download your timing plans")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Trained model timing plan**")
                    _download_file(st.session_state.last_schedule_path, "Download (.json)", "dl_dqn_schedule", "application/json")
                    st.caption("Fixed-time signal programme derived from your trained model.")
                with c2:
                    st.markdown("**Mathematical baseline timing plan**")
                    _download_file(st.session_state.last_schedule_webster_path, "Download (.json)", "dl_webster_schedule", "application/json")
                    st.caption("Optimised fixed-time programme calculated from your traffic count data.")
                with c3:
                    st.markdown("**Full comparison report**")
                    _download_file(st.session_state.last_comparison_csv_path, "Download (.csv)", "dl_comparison", "text/csv")
                    st.caption("Side-by-side performance comparison across all evaluation days.")

                st.caption("Want to try a different reward or more training rounds? Go to **Advanced** and **Run** again. Your uploaded data and intersection configuration will be preserved.")
            else:
                st.subheader("Your trained model timing plan")
                if had_baseline:
                    st.warning(
                        "Baseline comparison was enabled but the comparison report is not available. "
                        "You can still download the RL schedule below. Check the Run tab log if the comparison step failed."
                    )
                else:
                    st.success(
                        "Baseline comparison was not run. Enable **Run Webster baseline and compare in simulation** "
                        "under **Advanced → Baseline comparison** to get Webster vs RL KPI metrics and extra downloads."
                    )
                st.markdown("---")
                _download_file(sched_path, "Download trained model schedule (.json)", "dl_dqn_schedule_only", "application/json")
                st.caption("Fixed-time signal programme (`schedule.json`) derived from your trained model.")

                with st.expander("View signal timing plan (trained model)", expanded=False):
                    sp = sched_path
                    if sp and os.path.isfile(sp):
                        d_sched = json.loads(Path(sp).read_text(encoding="utf-8")).get("buckets", [])
                        phases = sorted(set(int(r["phase"]) for r in d_sched))
                        ptabs = st.tabs([f"Phase {p+1}" for p in phases])
                        for i, ph in enumerate(phases):
                            with ptabs[i]:
                                dph = [r for r in d_sched if int(r["phase"]) == ph]
                                if dph:
                                    dfp = pd.DataFrame({
                                        "Time of day": [_seconds_to_hhmm(int(r["bucket_start_s"])) for r in dph],
                                        "Trained model (s)": [float(r["median_s"]) for r in dph],
                                        "Trained model variation": [float(r.get("std_s", 0)) for r in dph],
                                    })
                                    st.line_chart(dfp.set_index("Time of day")["Trained model (s)"], key=f"line_phase_rl_{ph}")
                                    st.dataframe(dfp, width="stretch", key=f"df_phase_rl_{ph}")

                st.caption("Want baseline comparison? Enable it in **Advanced**, then **Run** again.")


if __name__ == "__main__":
    main()
