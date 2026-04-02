"""
Generates comparison plots for Fixed-Time, Actuated, and DQN policies.

Produces four figures saved to models_dir/plots/ (paths from repo config.json):
  1. bar_mean_reward.png — mean total reward per model
  2. box_reward_dist.png — reward distribution across test days
  3. line_reward_per_day.png — per-day reward for all three models overlaid
  4. learning_curve.png — DQN training reward over episodes
"""

from __future__ import annotations

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config_paths import models_dir, plots_dir


def _load(models: Path, name: str) -> list[dict]:
    with open(models / name, encoding="utf-8") as f:
        return json.load(f)


def rewards(log: list[dict]) -> list:
    return [entry["total_reward"] for entry in log]


def day_ids(log: list[dict]) -> list:
    return [entry["day_id"] for entry in log]


def main() -> None:
    mdir = models_dir()
    pdir = plots_dir()
    pdir.mkdir(parents=True, exist_ok=True)

    fixed_log = _load(mdir, "baseline_fixed_time_log.json")
    actuated_log = _load(mdir, "baseline_actuated_log.json")
    dqn_log = _load(mdir, "test_log.json")
    train_log = _load(mdir, "train_log.json")

    models = ["Fixed-Time", "Actuated", "DQN"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    logs = [fixed_log, actuated_log, dqn_log]

    style = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
    }
    plt.rcParams.update(style)

    # 1. Bar chart — mean reward
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [np.mean(rewards(log)) for log in logs]
    bars = ax.bar(models, means, color=colors, width=0.5, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_ylabel("Mean Total Reward")
    ax.set_title("Mean Test Reward by Policy")
    ax.set_ylim(min(means) * 1.15, 0)
    fig.tight_layout()
    fig.savefig(pdir / "bar_mean_reward.png")
    plt.close(fig)
    print("Saved bar_mean_reward.png")

    # 2. Box plot — reward distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        [rewards(log) for log in logs],
        tick_labels=models,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
        flierprops=dict(marker="o", markersize=5, linestyle="none"),
        zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution Across Test Days")
    fig.tight_layout()
    fig.savefig(pdir / "box_reward_dist.png")
    plt.close(fig)
    print("Saved box_reward_dist.png")

    # 3. Line chart — reward per day
    shared_ids = sorted(
        set(day_ids(fixed_log)) & set(day_ids(actuated_log)) & set(day_ids(dqn_log))
    )

    def aligned_rewards(log: list[dict], ids: list) -> list:
        lookup = {entry["day_id"]: entry["total_reward"] for entry in log}
        return [lookup[d] for d in ids]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, log, color in zip(models, logs, colors):
        ax.plot(
            shared_ids,
            aligned_rewards(log, shared_ids),
            marker="o",
            label=label,
            color=color,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Test Day ID")
    ax.set_ylabel("Total Reward")
    ax.set_title("Per-Day Reward: All Policies")
    ax.xaxis.set_major_locator(mticker.FixedLocator(shared_ids))
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(pdir / "line_reward_per_day.png")
    plt.close(fig)
    print("Saved line_reward_per_day.png")

    # 4. Learning curve — DQN training
    episodes = [e["episode"] for e in train_log]
    tr_reward = [e["total_reward"] for e in train_log]

    window = 10
    smoothed = np.convolve(tr_reward, np.ones(window) / window, mode="valid")
    smoothed_x = episodes[window - 1 :]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, tr_reward, color="#bbbbbb", linewidth=0.8, label="Raw reward", zorder=2)
    ax.plot(smoothed_x, smoothed, color=colors[2], linewidth=2, label=f"{window}-ep rolling mean", zorder=3)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Learning Curve (Training)")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(pdir / "learning_curve.png")
    plt.close(fig)
    print("Saved learning_curve.png")

    print(f"\nAll plots saved to {pdir.resolve()}")


if __name__ == "__main__":
    main()
