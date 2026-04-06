from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

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
) -> None:
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

    for day_id in test_days:
        flow_file = os.path.join(flows_dir, f"flows_day_{day_id:02d}.rou.xml")
        if not os.path.isfile(flow_file):
            print(f"  [replay] skip day {day_id}: missing {flow_file}")
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
        try:
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
        except Exception as e:
            print(f"  [replay] DQN day {day_id} failed: {e}")

        try:
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
        except Exception as e:
            print(f"  [replay] Webster day {day_id} failed: {e}")

        if r_d is not None and r_w is not None:
            dqn_list.append(r_d)
            web_list.append(r_w)

    if dqn_list and web_list:
        ReplayComparator().compare(dqn_list, web_list, out_dir)
    else:
        print(
            "  [replay] insufficient KPI results for comparison "
            f"(dqn={len(dqn_list)} webster={len(web_list)})."
        )


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
