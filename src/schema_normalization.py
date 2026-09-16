"""Helpers for normalizing output schemas across pipeline and UI layers."""

from __future__ import annotations

import pandas as pd


CANONICAL_RECOMMENDED_QUANTITY_COLUMN = "recommended_quantity_adjustment"
LEGACY_RECOMMENDED_QUANTITY_COLUMNS = [
    "recommended_quantity",
    "recommended_qty",
    "quantity",
]


def normalize_output_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy output columns to the project's canonical schema."""
    normalized = frame.copy()

    if CANONICAL_RECOMMENDED_QUANTITY_COLUMN not in normalized.columns:
        for legacy_column in LEGACY_RECOMMENDED_QUANTITY_COLUMNS:
            if legacy_column in normalized.columns:
                normalized = normalized.rename(
                    columns={legacy_column: CANONICAL_RECOMMENDED_QUANTITY_COLUMN}
                )
                break

    return normalized

