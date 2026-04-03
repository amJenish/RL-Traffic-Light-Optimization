"""
Plot training and test metrics from a single RL run folder (train_log.json / test_log.json).

Graphs are written under result_graphs_dir using the same folder name as the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DEFAULT_CONFIG_NAME = "config.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_output_paths(config_path: Path | None) -> tuple[Path, Path]:
    cfg_path = config_path or (_repo_root() / DEFAULT_CONFIG_NAME)
    if not cfg_path.is_file():
        root = _repo_root()
        return root / "src/data/results", root / "src/data/result_graphs"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    out = cfg.get("output", {})
    root = cfg_path.resolve().parent

    def resolve(rel_or_abs: str) -> Path:
        if not rel_or_abs:
            return root
        p = Path(rel_or_abs.replace("/", os.sep))
        if p.is_absolute():
            return p
        return (root / p).resolve()

    _r = out.get("results_dir") or out.get("logs_dir", "src/data/results")
    _g = out.get("result_graphs_dir", "src/data/result_graphs")
    return resolve(_r), resolve(_g)


def resolve_run_dir(run: str | Path, results_dir: Path) -> Path:
    """Find a run directory that contains train_log.json."""
    p = Path(run).expanduser()
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(results_dir / p)
        candidates.append(results_dir / p.name)
    for c in candidates:
        if c.is_dir() and (c / "train_log.json").is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No run folder with train_log.json (tried: {tried})")


STYLE: dict[str, Any] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
}


def _aggregate_train_by_epoch(
    train_log: list[dict[str, Any]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """
    One training row per (epoch, train day). Episode index runs 1..epochs*train_days;
    aggregate to one point per epoch (mean over train days).
    """
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in train_log:
        if "epoch" not in e:
            raise KeyError("train_log entries must include 'epoch' for epoch aggregation")
        by_epoch[int(e["epoch"])].append(e)
    epochs_sorted = sorted(by_epoch.keys())
    rewards: list[float] = []
    losses: list[float] = []
    epsilons: list[float] = []
    for ep in epochs_sorted:
        rows = by_epoch[ep]
        rewards.append(float(np.mean([float(r["total_reward"]) for r in rows])))
        loss_vals = [
            float(r["mean_loss"])
            for r in rows
            if r.get("mean_loss") is not None
        ]
        losses.append(float(np.mean(loss_vals)) if loss_vals else float("nan"))
        epsilons.append(float(np.mean([float(r["epsilon"]) for r in rows])))
    return epochs_sorted, rewards, losses, epsilons


class VisualizeResults:
    """
    Build PNG summaries for one training run.

    Parameters
    ----------
    run_path
        Path to the run folder, or the folder name under ``results_dir`` from config.
    results_dir, result_graphs_dir
        Override paths from ``config.json`` (defaults: ``src/data/results``,
        ``src/data/result_graphs``).
    config_path
        Alternate ``config.json`` location for resolving default directories.
    """

    def __init__(
        self,
        run_path: str | Path,
        *,
        results_dir: str | Path | None = None,
        result_graphs_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        cfg = Path(config_path) if config_path else None
        _rd, _gd = _load_output_paths(cfg)
        self.results_dir = Path(results_dir).resolve() if results_dir else _rd
        self.result_graphs_dir = Path(result_graphs_dir).resolve() if result_graphs_dir else _gd
        self.run_dir = resolve_run_dir(run_path, self.results_dir)
        self.output_dir = self.result_graphs_dir / self.run_dir.name

    def run(self) -> Path:
        """Write figures under ``result_graphs/<run_name>/`` and return that path."""
        train_path = self.run_dir / "train_log.json"
        if not train_path.is_file():
            raise FileNotFoundError(f"Missing {train_path}")

        train_log: list[dict[str, Any]] = _load_json(train_path)
        if not train_log:
            raise ValueError(f"Empty training log: {train_path}")

        test_path = self.run_dir / "test_log.json"
        test_log: list[dict[str, Any]] = _load_json(test_path) if test_path.is_file() else []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(STYLE)
        self._plot_train(train_log)
        self._plot_test(test_log)
        plt.rcdefaults()
        return self.output_dir

    def _plot_train(self, train_log: list[dict[str, Any]]) -> None:
        # Trainer logs one row per (epoch × train day); episode runs 1..epochs*train_days.
        # Plot against epoch (what config calls n_epochs), not per-day episode index.
        if train_log and "epoch" in train_log[0]:
            x, rewards, losses, epsilons = _aggregate_train_by_epoch(train_log)
            x_label = "Epoch"
            reward_label = "Mean total reward (over train days)"
        else:
            x = [int(e["episode"]) for e in train_log]
            rewards = [float(e["total_reward"]) for e in train_log]
            losses = [
                float(e["mean_loss"]) if e.get("mean_loss") is not None else np.nan
                for e in train_log
            ]
            epsilons = [float(e["epsilon"]) for e in train_log]
            x_label = "Episode"
            reward_label = "Total reward"

        window = min(10, max(1, len(rewards) // 5))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        smoothed_x = x[window - 1 :] if len(smoothed) else x

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_r, ax_l, ax_e, ax_s = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax_r.plot(x, rewards, color="#888888", linewidth=0.9, label=reward_label)
        ax_r.set_xlabel(x_label)
        ax_r.set_ylabel("Reward")
        ax_r.set_title("Training reward")
        ax_r.legend(framealpha=0.85, fontsize=8)

        ax_l.plot(x, losses, color="#4C72B0", linewidth=0.9)
        ax_l.set_xlabel(x_label)
        ax_l.set_ylabel("Mean loss")
        ax_l.set_title("Training loss")

        ax_e.plot(x, epsilons, color="#DD8452", linewidth=1.0)
        ax_e.set_xlabel(x_label)
        ax_e.set_ylabel("Epsilon")
        ax_e.set_title("Exploration (epsilon)")

        if len(smoothed):
            ax_s.plot(x, rewards, color="#dddddd", linewidth=0.6, label="Per-epoch mean")
            ax_s.plot(
                smoothed_x,
                smoothed,
                color="#55A868",
                linewidth=2.0,
                label=f"{window}-{x_label.lower()} rolling mean",
            )
        else:
            ax_s.plot(x, rewards, color="#55A868", linewidth=1.0)
        ax_s.set_xlabel(x_label)
        ax_s.set_ylabel("Reward")
        ax_s.set_title("Smoothed reward")
        ax_s.legend(framealpha=0.85, fontsize=8)

        fig.suptitle(self.run_dir.name, fontsize=11, y=1.02)
        fig.tight_layout()
        out = self.output_dir / "train_curves.png"
        fig.savefig(out)
        plt.close(fig)

    def _plot_test(self, test_log: list[dict[str, Any]]) -> None:
        out = self.output_dir / "test_rewards.png"
        if not test_log:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No test_log.json or empty", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            return

        ordered = sorted(test_log, key=lambda e: int(e["day_id"]))
        day_ids = [int(e["day_id"]) for e in ordered]
        rewards = [float(e["total_reward"]) for e in ordered]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(day_ids)), rewards, color="#55A868", alpha=0.85, zorder=3)
        ax.set_xticks(range(len(day_ids)))
        ax.set_xticklabels([str(d) for d in day_ids])
        ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(day_ids))))
        ax.set_xlabel("Test day_id")
        ax.set_ylabel("Total reward")
        ax.set_title("Test evaluation rewards")
        fig.suptitle(self.run_dir.name, fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from a training run folder.")
    parser.add_argument(
        "run",
        help="Path to run folder, or folder name under src/data/results (must contain train_log.json)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (defaults to repo root config.json)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override output.results_dir from config",
    )
    parser.add_argument(
        "--graphs-dir",
        default=None,
        help="Override output.result_graphs_dir from config",
    )
    args = parser.parse_args()
    viz = VisualizeResults(
        args.run,
        config_path=args.config,
        results_dir=args.results_dir,
        result_graphs_dir=args.graphs_dir,
    )
    out = viz.run()
    print(f"Saved graphs to {out}")


if __name__ == "__main__":
    main()
