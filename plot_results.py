"""
Generates comparison plots for Fixed-Time, Actuated, and DQN policies.

Produces four figures saved to src/data/models/plots/:
  1. bar_mean_reward.png — mean total reward per model
  2. box_reward_dist.png — reward distribution across the 6 test days
  3. line_reward_per_day.png — per-day reward for all three models overlaid
  4. learning_curve.png — DQN training reward per epoch (mean over train days)
"""

import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from visualization.visualize_results import (
    aggregate_train_for_plot,
    figure_single_plot_with_config_side,
)

# ── paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = pathlib.Path("src/data/models")
_RUN_CFG_DIR = MODELS_DIR
PLOTS_DIR = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _load(name: str) -> list[dict]:
    with open(MODELS_DIR / name) as f:
        return json.load(f)

fixed_log = _load("baseline_fixed_time_log.json")
actuated_log = _load("baseline_actuated_log.json")
dqn_log = _load("test_log.json")
train_log = _load("train_log.json")

# ── helpers ───────────────────────────────────────────────────────────────────
MODELS = ["Fixed-Time", "Actuated", "DQN"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
LOGS = [fixed_log, actuated_log, dqn_log]

def rewards(log):
    return [entry["total_reward"] for entry in log]

def day_ids(log):
    return [entry["day_id"] for entry in log]

STYLE = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
}
plt.rcParams.update(STYLE)

# ── 1. Bar chart — mean reward ─────────────────────────────────────────────
fig, ax = figure_single_plot_with_config_side(_RUN_CFG_DIR, figsize=(10.5, 4.2))
means = [np.mean(rewards(log)) for log in LOGS]
bars = ax.bar(MODELS, means, color=COLORS, width=0.5, zorder=3)
ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
ax.set_ylabel("Mean Total Reward")
ax.set_title("Mean Test Reward by Policy")
ax.set_ylim(min(means) * 1.15, 0)
fig.savefig(PLOTS_DIR / "bar_mean_reward.png", bbox_inches="tight")
plt.close(fig)
print("Saved bar_mean_reward.png")

# ── 2. Box plot — reward distribution ─────────────────────────────────────
fig, ax = figure_single_plot_with_config_side(_RUN_CFG_DIR, figsize=(10.5, 4.2))
bp = ax.boxplot(
    [rewards(log) for log in LOGS],
    tick_labels=MODELS,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    boxprops=dict(linewidth=1.3),
    whiskerprops=dict(linewidth=1.3),
    capprops=dict(linewidth=1.3),
    flierprops=dict(marker="o", markersize=5, linestyle="none"),
    zorder=3,
)
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_ylabel("Total Reward")
ax.set_title("Reward Distribution Across Test Days")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "box_reward_dist.png", bbox_inches="tight")
plt.close(fig)
print("Saved box_reward_dist.png")

# ── 3. Line chart — reward per day ────────────────────────────────────────
# Align on shared day_ids (same 6 days for all three logs)
shared_ids = sorted(set(day_ids(fixed_log)) & set(day_ids(actuated_log)) & set(day_ids(dqn_log)))

def aligned_rewards(log, ids):
    lookup = {entry["day_id"]: entry["total_reward"] for entry in log}
    return [lookup[d] for d in ids]

fig, ax = figure_single_plot_with_config_side(_RUN_CFG_DIR, figsize=(10.5, 4.2))
for label, log, color in zip(MODELS, LOGS, COLORS):
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
fig.savefig(PLOTS_DIR / "line_reward_per_day.png", bbox_inches="tight")
plt.close(fig)
print("Saved line_reward_per_day.png")

# ── 4. Learning curve — DQN training (per epoch) ─────────────────────────
_cfg_path = pathlib.Path("config.json")
_train_days = None
if _cfg_path.is_file():
    with open(_cfg_path, encoding="utf-8") as _cf:
        _tcfg = json.load(_cf)
        _t = _tcfg.get("training", {})
        _train_days = int(_t.get("train_days", _t.get("train_size", 0)) or 0) or None

epochs, tr_reward, _, _, x_label, reward_lbl = aggregate_train_for_plot(
    train_log, train_days=_train_days
)

window = min(10, max(1, len(tr_reward) // 5))
smoothed = np.convolve(tr_reward, np.ones(window) / window, mode="valid")
smoothed_x = epochs[window - 1 :]

fig, ax = figure_single_plot_with_config_side(_RUN_CFG_DIR, figsize=(10.5, 4.2))
ax.plot(epochs, tr_reward, color="#bbbbbb", linewidth=0.8, label=reward_lbl, zorder=2)
roll = f"{window}-epoch rolling mean" if x_label == "Epoch" else f"{window}-step rolling mean"
ax.plot(smoothed_x, smoothed, color=COLORS[2], linewidth=2, label=roll, zorder=3)
ax.set_xlabel(f"Training {x_label}")
ax.set_ylabel("Total Reward")
ax.set_title("DQN Learning Curve (Training)")
ax.legend(framealpha=0.8)
fig.savefig(PLOTS_DIR / "learning_curve.png", bbox_inches="tight")
plt.close(fig)
print("Saved learning_curve.png")

print(f"\nAll plots saved to {PLOTS_DIR.resolve()}")
