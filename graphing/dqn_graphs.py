"""Training curves from train_log.json — one point per epoch (mean over train days)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.visualize_results import (  # noqa: E402
    aggregate_train_for_plot,
    figure_grid_with_config_side,
)

_CFG = ROOT / "config.json"
_train_days = None
if _CFG.is_file():
    with open(_CFG, encoding="utf-8") as f:
        _cfg = json.load(f)
    _t = _cfg.get("training", {})
    _train_days = int(_t.get("train_days", _t.get("train_size", 0)) or 0) or None

_MODELS = ROOT / "src/data/models"
with open(_MODELS / "train_log.json", encoding="utf-8") as f:
    log = json.load(f)

x, rewards, losses, epsilons, x_label, reward_lbl = aggregate_train_for_plot(
    log, train_days=_train_days
)

window = min(10, max(1, len(rewards) // 5))
smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
smoothed_x = x[window - 1 :] if len(smoothed) else x
roll_lbl = (
    f"{window}-epoch rolling mean"
    if x_label == "Epoch"
    else f"{window}-step rolling mean"
)

fig, left = figure_grid_with_config_side(_MODELS, 2, 2, figsize=(14.5, 8))
ax00 = fig.add_subplot(left[0, 0])
ax01 = fig.add_subplot(left[0, 1])
ax10 = fig.add_subplot(left[1, 0])
ax11 = fig.add_subplot(left[1, 1])

ax00.plot(x, rewards)
ax00.set_title(f"Reward ({x_label})")
ax00.set_xlabel(x_label)

ax01.plot(x, losses)
ax01.set_title(f"Mean loss ({x_label})")
ax01.set_xlabel(x_label)

ax10.plot(x, epsilons)
ax10.set_title(f"Epsilon ({x_label})")
ax10.set_xlabel(x_label)

if len(smoothed):
    ax11.plot(x, rewards, color="#cccccc", linewidth=0.7, label=reward_lbl)
    ax11.plot(smoothed_x, smoothed, color="#2ca02c", linewidth=2.0, label=roll_lbl)
else:
    ax11.plot(x, rewards)
ax11.set_title("Smoothed reward")
ax11.set_xlabel(x_label)
ax11.legend(fontsize=8)

out_dir = ROOT / "src/data/graphs"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
