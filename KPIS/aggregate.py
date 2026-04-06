"""Filtered aggregates for baseline evaluation (omit crossings keys)."""
from __future__ import annotations

from typing import Any

from .summary import aggregate_test_kpis as _aggregate_full


def aggregate_test_kpis(
    test_log: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, float]:
    full = _aggregate_full(test_log, elapsed_s)
    return {k: v for k, v in full.items() if "crossings" not in k}
