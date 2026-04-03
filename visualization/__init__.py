from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .visualize_results import VisualizeResults

__all__ = ["VisualizeResults"]


def __getattr__(name: str) -> Any:
    if name == "VisualizeResults":
        from .visualize_results import VisualizeResults

        return VisualizeResults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
