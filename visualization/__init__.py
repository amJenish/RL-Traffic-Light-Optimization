from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .visualize_results import (
        VisualizeResults,
        aggregate_train_for_plot,
        format_sidecar_configs,
        format_training_settings_text,
        figure_grid_with_config_side,
        figure_single_plot_with_config_side,
        render_run_graphs,
    )

__all__ = [
    "VisualizeResults",
    "aggregate_train_for_plot",
    "format_sidecar_configs",
    "format_training_settings_text",
    "figure_grid_with_config_side",
    "figure_single_plot_with_config_side",
    "render_run_graphs",
]


def __getattr__(name: str) -> Any:
    if name == "VisualizeResults":
        from .visualize_results import VisualizeResults

        return VisualizeResults
    if name == "render_run_graphs":
        from .visualize_results import render_run_graphs

        return render_run_graphs
    if name == "aggregate_train_for_plot":
        from .visualize_results import aggregate_train_for_plot

        return aggregate_train_for_plot
    if name == "format_sidecar_configs":
        from .visualize_results import format_sidecar_configs

        return format_sidecar_configs
    if name == "format_training_settings_text":
        from .visualize_results import format_training_settings_text

        return format_training_settings_text
    if name == "figure_single_plot_with_config_side":
        from .visualize_results import figure_single_plot_with_config_side

        return figure_single_plot_with_config_side
    if name == "figure_grid_with_config_side":
        from .visualize_results import figure_grid_with_config_side

        return figure_grid_with_config_side
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
