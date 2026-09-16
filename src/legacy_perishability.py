"""Legacy spoilage diagnostics only; never used to size replenishment orders.

The pre-existing score weights and thresholds are retained pending a separate
perishability redesign. Callers mask unsupported inventory/calendar inputs.
"""
from __future__ import annotations

from statistics import NormalDist
import pandas as pd


def calculate_z_score(service_level: float) -> float:
    return NormalDist().inv_cdf(service_level)


def calculate_promo_metrics(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Measure how strongly promotions lift demand for each product."""
    promo_rows: list[dict[str, object]] = []
    for product, group in cleaned_data.groupby("product", sort=True):
        promo_slice = group[group["promo_flag"].fillna(False)]
        non_promo_slice = group[~group["promo_flag"].fillna(False)]
        promo_mean = float(promo_slice["demand"].mean()) if not promo_slice.empty else 0.0
        baseline_mean = float(non_promo_slice["demand"].mean()) if not non_promo_slice.empty else float(group["demand"].mean())
        promo_lift_pct = ((promo_mean - baseline_mean) / baseline_mean * 100) if baseline_mean else 0.0
        promo_rows.append(
            {
                "product": product,
                "promo_days": int(promo_slice.shape[0]),
                "promo_present": bool(not promo_slice.empty),
                "promo_avg_demand": round(promo_mean, 2),
                "non_promo_avg_demand": round(baseline_mean, 2),
                "promo_lift_pct": round(promo_lift_pct, 2),
                "promo_effect_detected": bool(not promo_slice.empty and promo_lift_pct > 5),
            }
        )
    return pd.DataFrame(promo_rows)


def _clamp_series(values: pd.Series) -> pd.Series:
    """Clamp values into the 0-1 range."""
    return values.clip(lower=0, upper=1)


def calculate_legacy_perishability_metrics(
    cleaned_data: pd.DataFrame,
    future_forecast: pd.DataFrame,
    service_level: float,
    lead_time: int,
) -> pd.DataFrame:
    """Compute inventory planning metrics for each product, including food spoilage signals."""
    if lead_time < 1:
        raise ValueError("lead_time must be at least 1 day")

    z_score = calculate_z_score(service_level)
    latest_snapshot = (
        cleaned_data.sort_values(["product", "date"])
        .groupby("product", as_index=False)
        .tail(1)[
            [
                "product",
                "location",
                "category",
                "on_hand_inventory",
                "lead_time_days",
                "days_to_expire",
                "perishable_flag",
                "expiring_before_expected_sale",
                "spoilage_risk_flag",
            ]
        ]
    )
    forecast_context = (
        future_forecast.groupby("product", as_index=False)
        .agg(
            avg_forecast_demand=("forecast", "mean"),
            total_forecast_demand=("forecast", "sum"),
        )
        .fillna(0)
    )
    promo_metrics = calculate_promo_metrics(cleaned_data)

    inventory = (
        cleaned_data.groupby("product", as_index=False)
        .agg(average_demand=("demand", "mean"), demand_std_dev=("demand", "std"))
        .fillna({"demand_std_dev": 0})
    )
    inventory = inventory.merge(latest_snapshot, on="product", how="left")
    inventory = inventory.merge(forecast_context, on="product", how="left")
    inventory = inventory.merge(promo_metrics, on="product", how="left")
    inventory["lead_time_days"] = inventory["lead_time_days"].fillna(lead_time).clip(lower=1)
    inventory["safety_stock"] = z_score * inventory["demand_std_dev"] * (inventory["lead_time_days"] ** 0.5)

    low_threshold = inventory["demand_std_dev"].quantile(0.33)
    high_threshold = inventory["demand_std_dev"].quantile(0.67)

    def assign_risk_level(std_dev: float) -> str:
        if std_dev >= high_threshold:
            return "High"
        if std_dev <= low_threshold:
            return "Low"
        return "Medium"

    forecast_daily = inventory["avg_forecast_demand"].fillna(inventory["average_demand"]).clip(lower=1)
    inventory_cover_days = inventory["on_hand_inventory"].fillna(0) / forecast_daily
    expiry_days = inventory["days_to_expire"].fillna(9999)
    projected_need = inventory["total_forecast_demand"].fillna(0) + inventory["safety_stock"]
    projected_surplus = inventory["on_hand_inventory"].fillna(0) - projected_need
    expiry_pressure = _clamp_series((inventory["lead_time_days"] + 3 - expiry_days) / (inventory["lead_time_days"] + 3))
    coverage_pressure = _clamp_series((inventory_cover_days - expiry_days) / expiry_days.replace(0, 1))
    surplus_pressure = _clamp_series(projected_surplus / projected_need.replace(0, 1))

    inventory["at_risk_of_expiring_before_sale"] = (
        inventory["expiring_before_expected_sale"].fillna(False)
        | (inventory["perishable_flag"].fillna(False) & (inventory_cover_days > expiry_days))
    ).astype(bool)
    inventory["spoilage_risk_score"] = (
        (expiry_pressure * 45) + (coverage_pressure * 35) + (surplus_pressure * 20)
    ).round(2)

    def assign_spoilage_risk(score: float, perishable_flag: object) -> str:
        if not bool(perishable_flag):
            return "Low"
        if score >= 67:
            return "High"
        if score >= 34:
            return "Medium"
        return "Low"

    inventory["risk_level"] = inventory["demand_std_dev"].apply(assign_risk_level)
    inventory["spoilage_risk"] = inventory.apply(
        lambda row: assign_spoilage_risk(float(row["spoilage_risk_score"]), row["perishable_flag"]),
        axis=1,
    )
    columns = [
        "product", "average_demand", "demand_std_dev", "avg_forecast_demand",
        "total_forecast_demand", "days_to_expire", "perishable_flag",
        "expiring_before_expected_sale", "spoilage_risk_flag", "promo_days",
        "promo_present", "promo_avg_demand", "non_promo_avg_demand", "promo_lift_pct",
        "promo_effect_detected", "at_risk_of_expiring_before_sale",
        "spoilage_risk_score", "risk_level", "spoilage_risk",
    ]
    return inventory[columns]
