"""Training curves (reward, loss, epsilon, smoothed reward) from train_log.json.

One point per **epoch** (mean over train days). Saves to out_dir/graphs/training_curves.png.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from config_paths import graphs_dir, load_config, models_dir, repo_root

_REPO = repo_root()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from visualization.visualize_results import aggregate_train_for_plot, figure_grid_with_config_side


def main() -> None:
    mdir = models_dir()
    gdir = graphs_dir()
    gdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    t = cfg.get("training", {})
    train_days = int(t.get("train_days", t.get("train_size", 0)) or 0) or None

    train_log_path = mdir / "train_log.json"
    with open(train_log_path, encoding="utf-8") as f:
        log = json.load(f)

    x, rewards, losses, epsilons, x_label, reward_lbl = aggregate_train_for_plot(
        log, train_days=train_days
    )

    window = min(10, max(1, len(rewards) // 5))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smoothed_x = x[window - 1 :] if len(smoothed) else x
    roll_lbl = (
        f"{window}-epoch rolling mean"
        if x_label == "Epoch"
        else f"{window}-step rolling mean"
    )

    fig, left = figure_grid_with_config_side(mdir, 2, 2, figsize=(14.5, 8))
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

    out_path = gdir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
