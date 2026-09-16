"""Daily periodic-review planning; independent of model selection and spoilage scores.

Inventory position currently equals usable on-hand stock. Outstanding purchase
orders and backorders are not modeled. See docs/inventory_planning.md for calendar
conventions and the limits of the historical-variability approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite, sqrt
from statistics import NormalDist

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Day, Week, MonthBegin, MonthEnd


REVIEW_INTERVAL_DAYS = 1
INVENTORY_ASSUMPTION = (
    "Inventory position equals usable on-hand inventory; outstanding purchase orders "
    "and backorders are not modeled."
)
REDUCE_ACTION = "Reduce replenishment / run down excess"


@dataclass(frozen=True)
class PlanningCalendar:
    frequency: str
    last_label: pd.Timestamp
    as_of: pd.Timestamp
    daily_mean: float
    daily_std: float


def period_bounds(label: pd.Timestamp, frequency: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return [start, end): month labels denote months; other labels end periods."""
    label = pd.Timestamp(label)
    offset = to_offset(frequency)
    if isinstance(offset, (MonthBegin, MonthEnd)) and offset.n == 1:
        start = label.to_period("M").start_time
        return start, start + MonthBegin(1)
    if isinstance(offset, Week):
        days = 7 * offset.n
    elif isinstance(offset, Day):
        days = offset.n
    else:
        raise ValueError("Unsupported calendar frequency; business-day and subdaily data need an explicit calendar.")
    return label + pd.Timedelta(days=1 - days), label + pd.Timedelta(days=1)


def planning_calendar(history: pd.DataFrame) -> PlanningCalendar:
    """Validate regular totals and estimate daily variance without inventing daily observations."""
    history = history.sort_values("date")
    dates = pd.DatetimeIndex(history["date"])
    if len(dates) < 3 or dates.has_duplicates or dates.hasnans:
        raise ValueError("At least three unique, regular dates are required for calendar-day planning.")
    if dates.tz is not None or not (dates == dates.normalize()).all():
        raise ValueError("Subdaily or timezone-dependent observations need an explicit calendar conversion.")
    frequency = pd.infer_freq(dates)
    if frequency is None:
        raise ValueError("Irregular or missing observation periods: calendar-day demand cannot be inferred safely.")
    if "location" in history and history["location"].nunique() > 1:
        raise ValueError("Multiple locations for one product: a single inventory pool must be defined first.")
    bounds = [period_bounds(date, frequency) for date in dates]
    lengths = pd.Series([(end - start).days for start, end in bounds], dtype=float)
    values = pd.to_numeric(history["demand"], errors="coerce").reset_index(drop=True)
    if not values.map(lambda v: pd.notna(v) and isfinite(v) and v >= 0).all():
        raise ValueError("Historical demand must contain finite nonnegative period totals.")
    mean = float(values.sum() / lengths.sum())
    # For fixed k-day periods this is sample_std(period_demand) / sqrt(k).
    # Variable month lengths use a common daily mean and independent daily increments.
    variance = float((((values - mean * lengths) ** 2) / lengths).sum() / (len(values) - 1))
    return PlanningCalendar(frequency, dates[-1], bounds[-1][1], mean, sqrt(max(0, variance)))


def required_planning_steps(history: pd.DataFrame, lead_time: float) -> int:
    """How many native forecast periods cover L + one calendar day?"""
    if not isfinite(lead_time) or lead_time < 1:
        raise ValueError("Lead time must be finite and at least one calendar day.")
    calendar = planning_calendar(history)
    end = calendar.as_of + pd.Timedelta(days=lead_time + REVIEW_INTERVAL_DAYS)
    label = calendar.last_label
    steps = 0
    while True:
        label += to_offset(calendar.frequency)
        steps += 1
        if period_bounds(label, calendar.frequency)[1] >= end:
            return steps


def demand_in_window(calendar: PlanningCalendar, future: pd.DataFrame, days: float) -> float:
    """Integrate uniform within-period demand, rejecting gaps, bad totals and short coverage."""
    end = calendar.as_of + pd.Timedelta(days=days)
    cursor = calendar.as_of
    total = 0.0
    expected_label = calendar.last_label + to_offset(calendar.frequency)
    for _, row in future.sort_values("date").iterrows():
        label = pd.Timestamp(row["date"])
        if label != expected_label:
            raise ValueError("Planning forecast dates do not form consecutive expected calendar periods.")
        start, stop = period_bounds(label, calendar.frequency)
        value = float(row["forecast"])
        if not isfinite(value) or value < 0:
            raise ValueError("Planning forecasts must be finite and nonnegative.")
        if start != cursor:
            raise ValueError("Planning forecasts contain a gap or overlap.")
        overlap_end = min(stop, end)
        total += value * (overlap_end - start) / (stop - start)
        if stop >= end:
            return float(total)
        cursor = stop
        expected_label += to_offset(calendar.frequency)
    raise ValueError(f"Planning forecast does not cover the required {days:g} calendar days.")


def calculate_core_inventory_metrics(
    cleaned_data: pd.DataFrame, planning_forecast: pd.DataFrame,
    service_level: float, default_lead_time: float,
) -> pd.DataFrame:
    """Calculate targets even when stock is unknown; withhold unsupported actions."""
    if not 0.5 <= service_level < 1:
        raise ValueError("Inventory service level must be at least 50% and less than 100%.")
    z = NormalDist().inv_cdf(service_level)
    rows = []
    metric_names = [
        "daily_average_demand", "daily_demand_std", "expected_demand_lead_time",
        "expected_demand_protection_period", "lead_time_safety_stock", "planning_safety_stock",
        "reorder_point", "order_up_to_target", "pre_arrival_shortage", "recommended_quantity_adjustment",
        "excess_quantity", "priority_score",
    ]
    for product, history in cleaned_data.groupby("product", sort=True):
        latest = history.sort_values("date").iloc[-1]
        raw_inventory = latest.get("on_hand_inventory")
        inventory = pd.to_numeric(raw_inventory, errors="coerce")
        inventory_status = "known"
        if pd.isna(inventory):
            inventory_status = "unknown"
        elif not isfinite(float(inventory)) or inventory < 0:
            inventory_status = "invalid"
        usable = float(inventory) if inventory_status == "known" else float("nan")
        # No future-expiry adjustment in this phase. Already expired stock is unusable.
        expiry = latest.get("days_to_expire")
        if inventory_status == "known" and pd.notna(expiry) and expiry <= 0:
            usable = 0.0
        lead = latest.get("lead_time_days", default_lead_time)
        lead = float(default_lead_time if pd.isna(lead) else lead)
        row = dict.fromkeys(metric_names, float("nan"))
        row.update({
            "product": product, "location": latest.get("location", "Unknown"),
            "category": latest.get("category"), "on_hand_inventory": inventory,
            "inventory_status": inventory_status, "usable_on_hand_inventory": usable,
            "inventory_position": usable, "lead_time_days": lead,
            "lead_time_source": latest.get("lead_time_source", "provided or configured default"),
            "review_interval_days": REVIEW_INTERVAL_DAYS, "protection_period_days": lead + 1,
            "service_level": service_level * 100, "z_score": z,
            "inventory_assumption": INVENTORY_ASSUMPTION,
            "forecast_frequency": None, "planning_as_of": None,
            "planning_status": "unavailable", "planning_issue": "", "recommended_action": None,
        })
        try:
            if not isfinite(lead) or lead < 1:
                raise ValueError("Lead time must be finite and at least one calendar day.")
            calendar = planning_calendar(history)
            row.update(forecast_frequency=calendar.frequency, planning_as_of=calendar.as_of,
                       daily_average_demand=calendar.daily_mean, daily_demand_std=calendar.daily_std)
            row["lead_time_safety_stock"] = z * calendar.daily_std * sqrt(lead)
            row["planning_safety_stock"] = z * calendar.daily_std * sqrt(lead + 1)
            future = planning_forecast.loc[planning_forecast["product"] == product]
            row["expected_demand_lead_time"] = demand_in_window(calendar, future, lead)
            row["reorder_point"] = row["expected_demand_lead_time"] + row["lead_time_safety_stock"]
            row["expected_demand_protection_period"] = demand_in_window(calendar, future, lead + 1)
            row["order_up_to_target"] = row["expected_demand_protection_period"] + row["planning_safety_stock"]
            if inventory_status != "known":
                raise ValueError(
                    "On-hand inventory is unknown; provide a current numeric inventory snapshot."
                    if inventory_status == "unknown" else
                    "On-hand inventory is invalid; negative or infinite on-hand values are not supported."
                )
            row["pre_arrival_shortage"] = max(0.0, row["expected_demand_lead_time"] - usable)
            difference = row["order_up_to_target"] - usable
            if (abs(difference) < 1 and row["pre_arrival_shortage"] == 0
                    and (usable > 0 or row["order_up_to_target"] == 0)):
                action, quantity, excess = "Maintain", 0, 0
            elif difference > 0:
                action, quantity, excess = "Increase", ceil(difference), 0
            elif difference < 0:
                action, quantity, excess = REDUCE_ACTION, floor(-difference), floor(-difference)
            else:
                action, quantity, excess = "Maintain", 0, 0
            row.update(recommended_action=action, recommended_quantity_adjustment=quantity,
                       excess_quantity=excess, planning_status="ready",
                       priority_score=3 if row["pre_arrival_shortage"] > 0 else 2 if action == "Increase" else 1)
            row["reasoning"] = (
                f"Usable inventory position is {usable:g} units; the unrounded order-up-to target is "
                f"{row['order_up_to_target']:.6g} units over {lead + 1:g} calendar days "
                f"(lead time {lead:g} + daily review 1). {action}: {quantity} whole units. "
                "Sub-unit gaps are maintained when there is no service gap; zero stock with a positive target still requires replenishment. "
                "Excess means defer replenishment and run down stock, not disposal. "
                f"Potential pre-arrival service gap: {row['pre_arrival_shortage']:.6g} units; "
                "ordinary replenishment may not arrive soon enough to prevent it. "
                + INVENTORY_ASSUMPTION
            )
        except (ValueError, TypeError, OverflowError) as exc:
            row["planning_issue"] = str(exc)
            row["reasoning"] = f"Inventory recommendation withheld: {exc} {INVENTORY_ASSUMPTION}"
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in ["recommended_quantity_adjustment", "excess_quantity", "priority_score"]:
        result[column] = result[column].astype("Int64")
    # Compatibility aliases now have one meaning and never supply competing decisions.
    result["safety_stock"] = result["lead_time_safety_stock"]
    result["lead_time"] = result["lead_time_days"]
    result["inventory_action"] = result["recommended_action"]
    return result
