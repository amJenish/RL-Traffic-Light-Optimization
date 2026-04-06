"""Trainer — runs the training loop over train days, then evaluates on test days."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from modelling.agent import Agent
from modelling.schedule_export import write_test_sequence_episode_json

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _green_phase_records_from_export(
    export: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Green phase ends only (for ScheduleBuilder), from Agent.take_phase_sequence_export."""
    out: list[dict[str, Any]] = []
    for tls_id, events in export.items():
        for ev in events:
            if ev.get("phase_name") == "yellow_clearance":
                continue
            state = ev.get("phase_state") or ""
            if "G" not in state and "g" not in state:
                continue
            out.append(
                {
                    "tls_id": tls_id,
                    "phase": int(ev["phase"]),
                    "end_s": float(ev["sim_time"]),
                    "duration_s": float(ev["duration_s"]),
                }
            )
    return out


class Trainer:

    def __init__(
        self,
        agent:        Agent,
        split_path:   str,
        flows_dir:    str,
        output_dir:   str                       = "src/data/models",
        n_epochs:     int                       = 3,
        save_every:   int                       = 10,
        log_every:    int                       = 5,
        log_callback: Callable[[str], None] | None = None,
        days_dir:     str | None = None,
        net_file:     str | None = None,
    ):
        self.agent        = agent
        self.output_dir   = output_dir
        self.n_epochs     = n_epochs
        self.save_every   = save_every
        self.log_every    = log_every
        self.log_callback = log_callback
        self._days_dir    = days_dir
        self._net_file    = net_file

        os.makedirs(output_dir, exist_ok=True)
        self._checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        with open(split_path) as f:
            self._split = json.load(f)

        self._train_days = self._split["train"]
        self._test_days  = self._split["test"]
        self._flows_dir  = flows_dir
        self._train_log: list[dict] = []
        self._test_log:  list[dict] = []
        self._phase_duration_records: list[dict[str, Any]] = []

    def _emit(self, text: str = "") -> None:
        """Print text to stdout and forward to log_callback if set."""
        print(text)
        if self.log_callback is not None:
            self.log_callback(text)

    def run(self) -> dict[str, Any]:
        """Full training loop, then post-training evaluation on test days."""
        self._emit(f"\nStarting training")
        self._emit(f"  Train days : {len(self._train_days)}")
        self._emit(f"  Test days  : {len(self._test_days)}")
        self._emit(f"  Epochs     : {self.n_epochs}")
        self._emit(f"  Output     : {self.output_dir}\n")

        episode = 0
        self.agent.reset_train_green_phase_stats()
        for epoch in range(1, self.n_epochs + 1):
            self._emit(f"--- Epoch {epoch}/{self.n_epochs} ---")
            epoch_start = len(self._train_log)
            for day_id in self._train_days:
                episode += 1
                metrics = self._run_episode(day_id, train=True)
                metrics["episode"] = episode
                metrics["epoch"]   = epoch
                metrics["day_id"]  = day_id
                self._train_log.append(metrics)

                if episode % self.log_every == 0:
                    self._log(episode, metrics, prefix="Train")

                if episode % self.save_every == 0:
                    path = os.path.join(
                        self._checkpoint_dir, f"checkpoint_ep{episode:04d}.pt"
                    )
                    self.agent.save(path)
                    self._emit(f"  Checkpoint saved -> {path}")

            chunk = self._train_log[epoch_start:]
            if chunk:
                mean_r = sum(m["total_reward"] for m in chunk) / len(chunk)
                self._emit(
                    f"  Train epoch={epoch:4d}  mean_reward={mean_r:8.1f}"
                )

        self._emit(f"\n--- Evaluating on {len(self._test_days)} test days ---")
        self.agent.set_eval_mode()
        self._phase_duration_records.clear()
        for test_episode, day_id in enumerate(self._test_days, start=1):
            metrics = self._run_episode(
                day_id, train=False, test_sequence_idx=test_episode
            )
            metrics["day_id"] = day_id
            self._test_log.append(metrics)
            self._log(test_episode, metrics, prefix="Test ")

        final_path = os.path.join(self.output_dir, "final_model.pt")
        self.agent.save(final_path)
        self._emit(f"\nFinal model saved -> {final_path}")

        self._save_logs()
        self._print_summary()

        return {
            "train_log": self._train_log,
            "pretest_log": [],
            "test_log": self._test_log,
            "phase_duration_records": self._phase_duration_records,
        }

    def _run_episode(
        self,
        day_id: int,
        train: bool,
        test_sequence_idx: int | None = None,
    ) -> dict[str, Any]:
        """Run one episode (one day) in train or eval mode."""
        route_file = os.path.join(
            self._flows_dir, f"flows_day_{day_id:02d}.rou.xml"
        )
        if not os.path.exists(route_file):
            raise FileNotFoundError(
                f"Route file not found: {route_file}\n"
                f"Run BuildRoute.py first to generate flow files."
            )

        if train:
            self.agent.set_train_mode()
        else:
            self.agent.set_eval_mode()

        obs  = self.agent.start_episode(route_file)
        done = False
        while not done:
            obs, rewards, done, loss = self.agent.step(obs)

        metrics = self.agent.end_episode()
        if not train:
            export = self.agent.take_phase_sequence_export()
            if test_sequence_idx is not None:
                self._write_test_sequence_json(test_sequence_idx, export)
                self._phase_duration_records.extend(
                    _green_phase_records_from_export(export)
                )
        return metrics

    def _write_test_sequence_json(
        self, episode_idx: int, tls_sequences: dict[str, list[dict[str, Any]]]
    ) -> None:
        write_test_sequence_episode_json(
            self.output_dir, episode_idx, tls_sequences
        )

    def _log(self, index: int, metrics: dict, prefix: str = "") -> None:
        eps = metrics.get("epsilon")
        eps_str = f"  eps={eps:.3f}" if eps is not None else ""
        loss = metrics.get("mean_loss")
        loss_str = f"  loss={loss:.4f}" if loss is not None else ""
        lr = metrics.get("learning_rate")
        lr_str = f"  lr={lr:.6f}" if lr is not None else ""
        self._emit(
            f"  {prefix} ep={index:4d}"
            f"  reward={metrics['total_reward']:8.1f}"
            f"  steps={metrics['steps']:5d}"
            f"{loss_str}{eps_str}{lr_str}"
        )

    def _save_logs(self) -> None:
        train_path = os.path.join(self.output_dir, "train_log.json")
        test_path = os.path.join(self.output_dir, "test_log.json")
        with open(train_path, "w") as f:
            json.dump(self._train_log, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(self._test_log, f, indent=2)
        self._emit(f"Logs saved -> {train_path}")
        self._emit(f"           -> {test_path}")

    def _print_summary(self) -> None:
        if self._train_log:
            mean_reward = sum(m["total_reward"] for m in self._train_log) / len(self._train_log)
            self._emit(f"\nTrain mean reward : {mean_reward:.2f}")
        if self._test_log:
            mean_reward = sum(m["total_reward"] for m in self._test_log) / len(self._test_log)
            self._emit(f"Test  mean reward : {mean_reward:.2f}")
        self._emit("")
        self._emit("--- Training green-phase hold times (sim seconds) ---")
        for line in self.agent.format_train_green_phase_duration_report():
            self._emit(line)
