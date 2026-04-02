"""Resolve repo root and output paths from the root config.json (no imports from main/modelling)."""

from __future__ import annotations

import json
from pathlib import Path


def repo_root() -> Path:
    """Walk upward from this file until config.json is found."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "config.json"
        if candidate.is_file():
            return parent
    raise FileNotFoundError(
        "Could not find config.json above front-end/scripts; run from the rl_traffic repo."
    )


def load_config() -> dict:
    with open(repo_root() / "config.json", encoding="utf-8") as f:
        return json.load(f)


def out_dir() -> Path:
    return repo_root() / load_config()["output"]["out_dir"]


def models_dir() -> Path:
    return repo_root() / load_config()["output"]["models_dir"]


def plots_dir() -> Path:
    """Comparison plots (plot_results.py)."""
    return models_dir() / "plots"


def graphs_dir() -> Path:
    """Training curve grids (dqn_graphs.py) — under out_dir/graphs."""
    return out_dir() / "graphs"
