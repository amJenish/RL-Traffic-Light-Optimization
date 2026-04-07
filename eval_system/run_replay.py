from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval_system.kpi_collector import KPIResult
from eval_system.replay_comparator import ReplayComparator
from eval_system.replay_runner import ReplayRunner
from eval_system.schedule_controller import ScheduleController


def run_replay_pipeline(
    *,
    schedule_path: str,
    schedule_webster_path: str,
    split_path: str,
    flows_dir: str,
    net_file: str,
    config: dict[str, Any],
    intersection_path: str,
    out_dir: str,
    sumo_home: str,
    gui: bool = False,
    tls_id: str = "J_centre",
    log: Callable[[str], None] | None = None,
    before_compare: Callable[[], None] | None = None,
) -> None:
    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg)
    with open(schedule_path, encoding="utf-8") as f:
        sched_dqn = json.load(f)
    with open(schedule_webster_path, encoding="utf-8") as f:
        sched_w = json.load(f)
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    with open(intersection_path, encoding="utf-8") as f:
        int_cfg = json.load(f)

    scfg = config.get("simulation", {})
    step_length = float(scfg.get("step_length", 1.0))
    decision_gap = int(scfg.get("decision_gap", 1))
    begin = int(scfg.get("begin", 0))
    end = int(scfg.get("end", 86400))

    min_green_s = float(int_cfg.get("min_green_s", 15))
    amber_s = float(int_cfg.get("amber_s", 3))
    min_red_s = float(int_cfg.get("min_red_s", 1))
    ptc = config.get("phase_timing", {})
    yellow_duration_s = float(ptc.get("yellow_duration_s", amber_s))

    os.makedirs(out_dir, exist_ok=True)
    dqn_dir = os.path.join(out_dir, "dqn")
    web_dir = os.path.join(out_dir, "webster")
    os.makedirs(dqn_dir, exist_ok=True)
    os.makedirs(web_dir, exist_ok=True)

    runner = ReplayRunner(
        net_file=net_file,
        step_length=step_length,
        decision_gap=decision_gap,
        begin=begin,
        end=end,
        sumo_home=sumo_home,
        gui=gui,
    )

    dqn_list: list[KPIResult] = []
    web_list: list[KPIResult] = []

    test_days = [int(x) for x in split.get("test", [])]
    missing_flows: list[str] = []
    day_failures: list[str] = []
    n_days = len(test_days)
    _emit(f"  [replay] evaluating {n_days} test day(s)")

    for day_idx, day_id in enumerate(test_days, start=1):
        _emit(f"  [replay] day {day_id} ({day_idx}/{n_days}) starting")
        flow_file = os.path.join(flows_dir, f"flows_day_{day_id:02d}.rou.xml")
        if not os.path.isfile(flow_file):
            msg = f"skip day {day_id}: missing {flow_file}"
            _emit(f"  [replay] {msg}")
            missing_flows.append(flow_file)
            continue

        ctrl_d = ScheduleController(
            sched_dqn,
            tls_id,
            min_green_s,
            yellow_duration_s,
            step_length,
        )
        ctrl_w = ScheduleController(
            sched_w,
            tls_id,
            min_green_s,
            yellow_duration_s,
            step_length,
        )

        r_d = None
        r_w = None
        d_err = ""
        w_err = ""
        try:
            _emit(f"  [replay] day {day_id}: running uploaded timing plan")
            r_d = runner.run(
                flow_file,
                ctrl_d,
                min_green_s,
                yellow_duration_s,
                min_red_s,
                tls_id,
                label="DQN",
            )
            p_d = os.path.join(dqn_dir, f"kpi_day_{day_id:02d}.json")
            with open(p_d, "w", encoding="utf-8") as f:
                json.dump(r_d.to_json_dict(), f, indent=2)
            _emit(f"  [replay] day {day_id}: uploaded timing plan complete")
        except Exception as e:
            d_err = str(e)
            _emit(f"  [replay] DQN day {day_id} failed: {e}")

        try:
            _emit(f"  [replay] day {day_id}: running mathematical baseline")
            r_w = runner.run(
                flow_file,
                ctrl_w,
                min_green_s,
                yellow_duration_s,
                min_red_s,
                tls_id,
                label="Webster",
            )
            p_w = os.path.join(web_dir, f"kpi_day_{day_id:02d}.json")
            with open(p_w, "w", encoding="utf-8") as f:
                json.dump(r_w.to_json_dict(), f, indent=2)
            _emit(f"  [replay] day {day_id}: mathematical baseline complete")
        except Exception as e:
            w_err = str(e)
            _emit(f"  [replay] Webster day {day_id} failed: {e}")

        if r_d is not None and r_w is not None:
            dqn_list.append(r_d)
            web_list.append(r_w)
            _emit(f"  [replay] day {day_id} complete ({day_idx}/{n_days})")
        elif r_d is None or r_w is None:
            bits = []
            if r_d is None:
                bits.append(f"DQN ({d_err or 'no result'})")
            if r_w is None:
                bits.append(f"Webster ({w_err or 'no result'})")
            day_failures.append(f"day {day_id}: " + "; ".join(bits))
            _emit(f"  [replay] day {day_id} incomplete ({day_idx}/{n_days})")

    if dqn_list and web_list:
        if before_compare is not None:
            before_compare()
        _emit("  [replay] writing comparison.csv")
        ReplayComparator().compare(dqn_list, web_list, out_dir)
        _emit("  [replay] comparison.csv complete")
        return

    lines = [
        "No evaluation days completed for both timing plans, so comparison.csv was not written.",
        f"test_days from split: {test_days}",
        f"flows_dir: {flows_dir}",
        f"SUMO_HOME: {sumo_home!r}",
    ]
    if not (sumo_home or "").strip():
        lines.append(
            "SUMO_HOME is empty — set it in Advanced or the SUMO_HOME environment variable."
        )
    if missing_flows:
        lines.append(
            f"Missing {len(missing_flows)} flow file(s) (need flows_day_XX.rou.xml for each test day). "
            "Use a run folder that contains sumo/flows/ from a full build, not only schedule.json."
        )
    if day_failures:
        lines.append("Simulation errors: " + " | ".join(day_failures[:5]))
        if len(day_failures) > 5:
            lines.append(f"(+{len(day_failures) - 5} more days failed)")
    dqn_buckets = sum(
        1 for b in (sched_dqn.get("buckets") or []) if b.get("tls_id") == tls_id
    )
    w_buckets = sum(
        1 for b in (sched_w.get("buckets") or []) if b.get("tls_id") == tls_id
    )
    if dqn_buckets == 0 or w_buckets == 0:
        lines.append(
            f"Schedule JSON may not target tls_id={tls_id!r} (uploaded plan: {dqn_buckets} buckets, "
            f"Webster plan: {w_buckets} buckets). Buckets must use that tls_id."
        )
    raise RuntimeError("\n".join(lines))


def main_cli() -> None:
    p = argparse.ArgumentParser(description="Pipeline 2 — schedule replay KPI comparison")
    p.add_argument("--schedule", required=True)
    p.add_argument("--schedule-webster", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--flows-dir", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--intersection", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sumo-home", default=os.environ.get("SUMO_HOME", ""))
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if not args.sumo_home:
        print("ERROR: pass --sumo-home or set SUMO_HOME")
        sys.exit(1)
    run_replay_pipeline(
        schedule_path=args.schedule,
        schedule_webster_path=args.schedule_webster,
        split_path=args.split,
        flows_dir=args.flows_dir,
        net_file=args.net,
        config=cfg,
        intersection_path=args.intersection,
        out_dir=args.out_dir,
        sumo_home=args.sumo_home,
        gui=args.gui,
    )


if __name__ == "__main__":
    main_cli()
