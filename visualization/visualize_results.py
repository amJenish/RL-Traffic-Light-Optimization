"""
Plot training and test metrics from a single RL run folder (train_log.json / test_log.json).

Graphs are written under result_graphs_dir using the same folder name as the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

_SIDECAR_FILES = (
    ("Policy configuration", "policy_configuration.json"),
    ("Reward configuration", "reward_configuration.json"),
)


def format_training_settings_text(run_dir: Path) -> str:
    """
    Training hyperparameters for the side panel: n_epochs, train_days, test_days,
    scheduler, lr_min. Prefers ``config.txt`` from a ``main.py`` run when present;
    fills gaps from repo root ``config.json`` ``training`` block.
    """
    values: dict[str, str] = {}

    ct = Path(run_dir) / "config.txt"
    if ct.is_file():
        raw = ct.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^Epochs:\s*(.+)$", raw, re.MULTILINE)
        if m:
            values["n_epochs"] = m.group(1).strip()
        m = re.search(r"^Train days:\s*(.+)$", raw, re.MULTILINE)
        if m:
            values["train_days"] = m.group(1).strip()
        m = re.search(r"^Test days:\s*(.+)$", raw, re.MULTILINE)
        if m:
            values["test_days"] = m.group(1).strip()
        m = re.search(r"^Scheduler:\s*(.+)$", raw, re.MULTILINE)
        if m:
            values["scheduler"] = m.group(1).strip()
        m = re.search(r"Learning rate:\s*[\d.]+\s*->\s*([\d.eE+-]+)", raw)
        if m:
            values["lr_min"] = m.group(1).strip()

    cfg_path = _repo_root() / DEFAULT_CONFIG_NAME
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        t = cfg.get("training", {})
        values.setdefault("n_epochs", str(t.get("n_epochs", "?")))
        td = t.get("train_days", t.get("train_size"))
        values.setdefault("train_days", str(td if td is not None else "?"))
        values.setdefault("test_days", str(t.get("test_days", "?")))
        sched = t.get("scheduler", "?")
        values.setdefault("scheduler", str(sched) if sched not in (None, "") else "?")
        lm = t.get("lr_min")
        values.setdefault("lr_min", str(lm) if lm is not None else "?")

    lines = [
        f"n_epochs:   {values.get('n_epochs', '?')}",
        f"train_days: {values.get('train_days', '?')}",
        f"test_days:  {values.get('test_days', '?')}",
        f"scheduler:  {values.get('scheduler', '?')}",
        f"lr_min:     {values.get('lr_min', '?')}",
    ]
    return "\n".join(lines)


def format_sidecar_configs(run_dir: Path) -> str:
    """Pretty-print training settings plus policy/reward JSON (side panel text)."""
    parts: list[str] = []
    rd = Path(run_dir)
    parts.append("Training")
    parts.append("(epochs, train_days, test_days, scheduler, lr_min)")
    parts.append(format_training_settings_text(rd))
    parts.append("")
    for title, fname in _SIDECAR_FILES:
        p = rd / fname
        parts.append(title)
        parts.append(f"({fname})")
        if p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                parts.append(json.dumps(obj, indent=2, ensure_ascii=False))
            except Exception as exc:
                parts.append(f"(invalid JSON: {exc})")
        else:
            parts.append("(file not found)")
        parts.append("")
    text = "\n".join(parts).strip()
    max_chars = 16_000
    if len(text) > max_chars:
        text = text[: max_chars - 40] + "\n\n… (truncated)"
    return text


def _draw_config_side_axes(fig: Any, run_dir: Path, side_spec: Any) -> None:
    ax_side = fig.add_subplot(side_spec)
    ax_side.axis("off")
    body = format_sidecar_configs(run_dir)
    ax_side.text(
        0.02,
        0.98,
        body,
        transform=ax_side.transAxes,
        va="top",
        ha="left",
        fontsize=5.8,
        family="monospace",
        linespacing=1.15,
    )


def figure_single_plot_with_config_side(
    run_dir: Path,
    figsize: tuple[float, float] = (10.5, 4.2),
    *,
    side_ratio: float = 0.52,
) -> tuple[Any, Any]:
    """One main axes on the left, policy/reward JSON panel on the right."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, side_ratio], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    _draw_config_side_axes(fig, run_dir, gs[1])
    return fig, ax


def figure_grid_with_config_side(
    run_dir: Path,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] = (14.5, 8),
    *,
    side_ratio: float = 0.5,
) -> tuple[Any, Any]:
    """``nrows``×``ncols`` grid on the left; right: training + policy/reward. Returns ``(fig, left_gridspec)``."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, side_ratio], wspace=0.08)
    left = gs[0].subgridspec(nrows, ncols, wspace=0.3, hspace=0.35)
    _draw_config_side_axes(fig, run_dir, gs[1])
    return fig, left


def _mean_metrics_from_epoch_buckets(
    by_epoch: dict[int, list[dict[str, Any]]],
) -> tuple[list[int], list[float], list[float], list[float]]:
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


def _bucket_train_rows_by_epoch(
    train_log: list[dict[str, Any]],
    train_days: int | None,
) -> dict[int, list[dict[str, Any]]] | None:
    """
    Group training rows into epochs. Prefer explicit ``epoch`` field; else infer
    ``epoch = (episode - 1) // train_days + 1`` when ``train_days`` is set.
    """
    if not train_log:
        return None
    if all("epoch" in e for e in train_log):
        by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for e in train_log:
            by_epoch[int(e["epoch"])].append(e)
        return dict(by_epoch)
    if train_days is not None and train_days > 0:
        by_epoch = defaultdict(list)
        for e in train_log:
            ep = int(e["episode"])
            epoch = (ep - 1) // train_days + 1
            by_epoch[epoch].append(e)
        return dict(by_epoch)
    return None


def aggregate_train_for_plot(
    train_log: list[dict[str, Any]],
    *,
    train_days: int | None = None,
) -> tuple[list[int], list[float], list[float], list[float], str, str]:
    """
    Series for training plots: **one point per epoch** (mean over train days).

    Uses the ``epoch`` field when present on all rows; otherwise buckets by
    ``episode`` and ``train_days`` (from config). Fails if neither works.

    Returns
    -------
    x, rewards, losses, epsilons, x_label, reward_legend
    """
    if not train_log:
        raise ValueError("train_log is empty")

    buckets = _bucket_train_rows_by_epoch(train_log, train_days)
    if buckets is not None:
        x, rewards, losses, epsilons = _mean_metrics_from_epoch_buckets(buckets)
        return (
            x,
            rewards,
            losses,
            epsilons,
            "Epoch",
            "Mean total reward (over train days)",
        )

    x = [int(e["episode"]) for e in train_log]
    rewards = [float(e["total_reward"]) for e in train_log]
    losses = [
        float(e["mean_loss"]) if e.get("mean_loss") is not None else np.nan
        for e in train_log
    ]
    epsilons = [float(e["epsilon"]) for e in train_log]
    return (
        x,
        rewards,
        losses,
        epsilons,
        "Episode",
        "Total reward",
    )


def _default_train_days_from_config() -> int | None:
    cfg_path = _repo_root() / DEFAULT_CONFIG_NAME
    if not cfg_path.is_file():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    t = cfg.get("training", {})
    v = t.get("train_days", t.get("train_size"))
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def render_run_graphs(
    run_dir: str | Path,
    graphs_dir: str | Path | None = None,
) -> Path:
    """
    Write ``train_curves.png`` and ``test_rewards.png`` under ``graphs_dir``.

    If ``graphs_dir`` is omitted, uses ``<run_dir>/graphs``.
    """
    run_path = Path(run_dir).expanduser().resolve()
    out = (
        Path(graphs_dir).expanduser().resolve()
        if graphs_dir is not None
        else run_path / "graphs"
    )
    return VisualizeResults(run_path, output_dir=out).run()


class VisualizeResults:
    """
    Build PNG summaries for one training run.

    Parameters
    ----------
    run_path
        Path to the run folder, or the folder name under ``results_dir`` from config.
    results_dir, result_graphs_dir
        Override paths from ``config.json`` (defaults: ``src/data/results``,
        ``src/data/result_graphs``). Ignored when ``output_dir`` is set.
    output_dir
        If set, write figures here and treat ``run_path`` as a direct path to the run
        folder (must contain ``train_log.json``). Used e.g. for grid-search trials
        (``.../trial_.../graphs``).
    config_path
        Alternate ``config.json`` location for resolving default directories.
    """

    def __init__(
        self,
        run_path: str | Path,
        *,
        results_dir: str | Path | None = None,
        result_graphs_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if output_dir is not None:
            self.run_dir = Path(run_path).expanduser().resolve()
            if not (self.run_dir / "train_log.json").is_file():
                raise FileNotFoundError(f"Missing train_log.json under {self.run_dir}")
            self.output_dir = Path(output_dir).expanduser().resolve()
            return

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
        td = _default_train_days_from_config()
        x, rewards, losses, epsilons, x_label, reward_label = aggregate_train_for_plot(
            train_log, train_days=td
        )

        window = min(10, max(1, len(rewards) // 5))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        smoothed_x = x[window - 1 :] if len(smoothed) else x

        fig = plt.figure(figsize=(14.5, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.5], wspace=0.08)
        left = gs[0].subgridspec(2, 2, wspace=0.3, hspace=0.35)
        ax_r = fig.add_subplot(left[0, 0])
        ax_l = fig.add_subplot(left[0, 1])
        ax_e = fig.add_subplot(left[1, 0])
        ax_s = fig.add_subplot(left[1, 1])
        _draw_config_side_axes(fig, self.run_dir, gs[1])

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

        roll_lbl = f"{window}-epoch rolling mean" if x_label == "Epoch" else f"{window}-step rolling mean"
        if len(smoothed):
            ax_s.plot(x, rewards, color="#dddddd", linewidth=0.6, label=reward_label)
            ax_s.plot(smoothed_x, smoothed, color="#55A868", linewidth=2.0, label=roll_lbl)
        else:
            ax_s.plot(x, rewards, color="#55A868", linewidth=1.0)
        ax_s.set_xlabel(x_label)
        ax_s.set_ylabel("Reward")
        ax_s.set_title("Smoothed reward")
        ax_s.legend(framealpha=0.85, fontsize=8)

        out = self.output_dir / "train_curves.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    def _plot_test(self, test_log: list[dict[str, Any]]) -> None:
        out = self.output_dir / "test_rewards.png"
        if not test_log:
            fig = plt.figure(figsize=(10, 3.5))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.55], wspace=0.1)
            ax = fig.add_subplot(gs[0])
            ax.text(0.5, 0.5, "No test_log.json or empty", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            _draw_config_side_axes(fig, self.run_dir, gs[1])
            fig.tight_layout()
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            return

        ordered = sorted(test_log, key=lambda e: int(e["day_id"]))
        day_ids = [int(e["day_id"]) for e in ordered]
        rewards = [float(e["total_reward"]) for e in ordered]

        fig = plt.figure(figsize=(11, 4.2), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.52], wspace=0.1)
        ax = fig.add_subplot(gs[0])
        ax.bar(range(len(day_ids)), rewards, color="#55A868", alpha=0.85, zorder=3)
        ax.set_xticks(range(len(day_ids)))
        ax.set_xticklabels([str(d) for d in day_ids])
        ax.xaxis.set_major_locator(mticker.FixedLocator(range(len(day_ids))))
        ax.set_xlabel("Test day_id")
        ax.set_ylabel("Total reward")
        ax.set_title("Test evaluation rewards")
        _draw_config_side_axes(fig, self.run_dir, gs[1])
        fig.savefig(out, bbox_inches="tight")
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
