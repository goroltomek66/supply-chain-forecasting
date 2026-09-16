"""Input schema mapping and preprocessing utilities for supply-chain datasets."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


LOGICAL_FIELDS = [
    "date",
    "product",
    "demand",
    "location",
    "on_hand_inventory",
    "lead_time_days",
    "expiration_date",
    "shelf_life_days",
    "promo_flag",
    "category",
]

REQUIRED_FIELDS = ["date", "product", "demand"]

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "ds", "forecast_date", "order_date", "sales_date", "transaction_date", "day"],
    "product": ["product", "item", "item_name", "sku", "sku_name", "product_name", "material", "description"],
    "demand": [
        "demand",
        "sales",
        "units_sold",
        "qty",
        "quantity",
        "volume",
        "orders",
        "units",
        "demand_qty",
        "sales_qty",
        "order_qty",
        "forecast_qty",
    ],
    "location": ["location", "store", "warehouse", "site", "plant", "depot", "distribution_center", "dc"],
    "on_hand_inventory": [
        "on_hand_inventory",
        "inventory",
        "inventory_on_hand",
        "stock_on_hand",
        "on_hand",
        "available_inventory",
        "current_inventory",
    ],
    "lead_time_days": ["lead_time_days", "lead_time", "leadtime", "lead_days", "supplier_lead_time", "replenishment_days", "lt"],
    "expiration_date": ["expiration_date", "expiry_date", "expires_on", "use_by_date", "best_by_date", "sell_by_date"],
    "shelf_life_days": ["shelf_life_days", "shelf_life", "shelf_days", "remaining_shelf_life_days"],
    "promo_flag": ["promo_flag", "promotion", "promo", "on_promo", "promotion_flag", "is_promo", "discount_flag"],
    "category": ["category", "product_category", "family", "segment", "department", "class"],
}


def normalize_column_name(column_name: Any) -> str:
    """Normalize an incoming column name to a matching key."""
    normalized = str(column_name).strip()
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def build_schema_mapping(columns: list[str]) -> dict[str, str]:
    """Map dataset columns onto supported logical fields using aliases."""
    normalized_to_original = {normalize_column_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for logical_field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_to_original:
                mapping[logical_field] = normalized_to_original[normalized_alias]
                break
    return mapping


def validate_required_fields(schema_mapping: dict[str, str], columns: list[str]) -> None:
    """Raise a clear error when required fields cannot be mapped."""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in schema_mapping]
    if not missing_fields:
        return

    available_columns = sorted(normalize_column_name(column) for column in columns)
    alias_hints = {field: COLUMN_ALIASES[field] for field in missing_fields}
    if "demand" in missing_fields:
        raise ValueError(
            "No valid demand column was found in the uploaded dataset. "
            f"Available normalized columns: {available_columns}. "
            f"Accepted demand aliases: {COLUMN_ALIASES['demand']}."
        )
    raise ValueError(
        "Input dataset is missing required fields after schema mapping. "
        f"Missing logical fields: {missing_fields}. "
        f"Available columns: {available_columns}. "
        f"Accepted aliases: {alias_hints}."
    )


def standardize_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Map dataset columns to the project's logical schema."""
    schema_mapping = build_schema_mapping(list(data.columns))
    validate_required_fields(schema_mapping, list(data.columns))

    standardized = pd.DataFrame(index=data.index)
    for logical_field in LOGICAL_FIELDS:
        if logical_field in schema_mapping:
            standardized[logical_field] = data[schema_mapping[logical_field]]
        else:
            standardized[logical_field] = pd.Series([pd.NA] * len(data), index=data.index)
    return standardized


def parse_dates_with_fallback(date_series: pd.Series) -> pd.Series:
    """Handle mixed date formats in uploaded CSVs."""
    parsed_dates = pd.to_datetime(date_series, errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        fallback_dates = pd.to_datetime(date_series, errors="coerce", format="mixed", dayfirst=True)
        parsed_dates = parsed_dates.fillna(fallback_dates)
    return parsed_dates


def parse_promo_flag(flag_series: pd.Series) -> pd.Series:
    """Convert common promo encodings to booleans."""
    normalized = flag_series.astype("string").str.strip().str.lower()
    truthy_values = {"1", "true", "yes", "y", "promo", "promoted", "on", "t"}
    falsy_values = {"0", "false", "no", "n", "off", "f"}
    parsed = pd.Series(pd.NA, index=flag_series.index, dtype="boolean")
    parsed.loc[normalized.isin(truthy_values)] = True
    parsed.loc[normalized.isin(falsy_values)] = False
    return parsed


def add_food_features(standardized: pd.DataFrame) -> pd.DataFrame:
    """Create food and dairy oriented perishability features."""
    enriched = standardized.copy()
    if "lead_time_days" not in enriched:
        enriched["lead_time_days"] = pd.Series([pd.NA] * len(enriched), dtype="float64")

    days_from_expiration = (
        enriched["expiration_date"] - enriched["date"]
        if {"expiration_date", "date"}.issubset(enriched.columns)
        else pd.Series([pd.NaT] * len(enriched))
    )
    days_to_expire = days_from_expiration.dt.days if hasattr(days_from_expiration, "dt") else pd.Series([pd.NA] * len(enriched))
    shelf_life_fallback = enriched["shelf_life_days"] if "shelf_life_days" in enriched else pd.Series([pd.NA] * len(enriched))
    enriched["days_to_expire"] = days_to_expire.fillna(shelf_life_fallback)

    enriched["perishable_flag"] = (
        enriched["expiration_date"].notna()
        | enriched["shelf_life_days"].fillna(9999).le(14)
    ).astype("boolean")

    lead_time_threshold = enriched["lead_time_days"].fillna(2)
    inventory_risk = enriched["on_hand_inventory"].fillna(0) > enriched["demand"].fillna(0) * lead_time_threshold.clip(lower=1)
    expiry_risk = enriched["days_to_expire"].fillna(9999) <= (lead_time_threshold + 1)
    expected_days_to_sell = enriched["on_hand_inventory"].fillna(0) / enriched["demand"].replace(0, pd.NA)
    expected_days_to_sell = expected_days_to_sell.fillna(0)
    enriched["expiring_before_expected_sale"] = (
        enriched["perishable_flag"].fillna(False) & (enriched["days_to_expire"].fillna(9999) < expected_days_to_sell)
    ).astype("boolean")
    enriched["spoilage_risk_flag"] = (enriched["perishable_flag"].fillna(False) & (expiry_risk | inventory_risk)).astype("boolean")
    return enriched


def clean_and_preprocess_data(data: pd.DataFrame, default_lead_time_days: int) -> pd.DataFrame:
    """Map input schema, clean types, and add food-specific preprocessing features."""
    cleaned = standardize_schema(data)

    cleaned["date"] = parse_dates_with_fallback(cleaned["date"])
    cleaned["expiration_date"] = parse_dates_with_fallback(cleaned["expiration_date"])
    for text_column in ["product", "location", "category"]:
        cleaned[text_column] = cleaned[text_column].astype("string").str.strip()

    for numeric_column in ["demand", "on_hand_inventory", "lead_time_days", "shelf_life_days"]:
        cleaned[numeric_column] = pd.to_numeric(cleaned[numeric_column], errors="coerce")
    cleaned["promo_flag"] = parse_promo_flag(cleaned["promo_flag"])

    if cleaned["demand"].dropna().empty:
        raise ValueError(
            "A demand column was detected, but it does not contain any valid numeric values after conversion."
        )

    cleaned = cleaned.dropna(subset=["date"])
    if cleaned.empty:
        raise ValueError("No valid rows remain after parsing dates. Check the input date formats.")

    cleaned["product"] = cleaned["product"].replace("", pd.NA)
    cleaned["product"] = cleaned["product"].ffill().bfill()
    if cleaned["product"].dropna().empty:
        raise ValueError("No valid product values remain after cleaning the uploaded CSV.")

    cleaned = cleaned.sort_values(["product", "date"]).reset_index(drop=True)
    cleaned["demand"] = cleaned.groupby("product")["demand"].transform(
        lambda series: series.interpolate(limit_direction="both")
    )
    cleaned["demand"] = cleaned.groupby("product")["demand"].transform(lambda series: series.fillna(series.median()))
    cleaned["demand"] = cleaned["demand"].fillna(0)

    cleaned["location"] = cleaned["location"].replace("", pd.NA).fillna("Unknown")
    cleaned["category"] = cleaned["category"].replace("", pd.NA)
    # Preserve unknown stock. Negative values remain visible for core validation.
    cleaned["inventory_status"] = "known"
    cleaned.loc[cleaned["on_hand_inventory"].isna(), "inventory_status"] = "unknown"
    cleaned.loc[cleaned["on_hand_inventory"] < 0, "inventory_status"] = "invalid"
    cleaned.loc[cleaned["on_hand_inventory"].isin([float("inf"), float("-inf")]), "inventory_status"] = "invalid"
    cleaned["lead_time_source"] = cleaned["lead_time_days"].apply(
        lambda value: "configured default" if pd.isna(value) else "product data"
    )
    cleaned["lead_time_days"] = cleaned["lead_time_days"].fillna(default_lead_time_days)
    cleaned["shelf_life_days"] = cleaned["shelf_life_days"].where(cleaned["shelf_life_days"] > 0, pd.NA)

    cleaned = add_food_features(cleaned)
    column_order = LOGICAL_FIELDS + [
        "inventory_status",
        "lead_time_source",
        "days_to_expire",
        "perishable_flag",
        "expiring_before_expected_sale",
        "spoilage_risk_flag",
    ]
    return cleaned[column_order].copy()
