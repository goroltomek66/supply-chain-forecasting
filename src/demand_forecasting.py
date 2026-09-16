"""Demand forecasting pipeline for practical supply chain analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import clean_and_preprocess_data
from forecasting_models import ProductForecastResult, select_best_forecast_for_product
from schema_normalization import normalize_output_schema
from visualization import create_inventory_charts
from inventory_planning import calculate_core_inventory_metrics, required_planning_steps, INVENTORY_ASSUMPTION
from legacy_perishability import calculate_legacy_perishability_metrics


@dataclass
class PipelineRunResult:
    """Container for the main pipeline outputs and saved file locations."""

    cleaned_data: pd.DataFrame
    run_config: dict[str, float | int]
    planning_forecast: pd.DataFrame
    planning_forecast_path: Path
    actual_vs_forecast: pd.DataFrame
    future_forecast: pd.DataFrame
    method_summary: pd.DataFrame
    top_candidates: pd.DataFrame
    metrics: pd.DataFrame
    inventory_metrics: pd.DataFrame
    summary_text: str
    forecast_csv_path: Path
    cleaned_dataset_path: Path
    metrics_csv_path: Path
    inventory_metrics_path: Path
    method_selection_path: Path
    top_candidates_path: Path
    summary_path: Path
    chart_paths: list[Path]
    inventory_chart_paths: list[Path]


def parse_args() -> argparse.Namespace:
    """Collect command-line settings for the forecasting run."""
    parser = argparse.ArgumentParser(description="Run a demand forecasting pipeline.")
    parser.add_argument("--input", default="data/sample_demand.csv", help="Input CSV path.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--config", default="config/project_config.json", help="JSON config path.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, float | int]:
    """Load runtime configuration from disk."""
    default_config: dict[str, float | int] = {
        "service_level": 95,
        "lead_time": 2,
        "forecast_horizon": 3,
    }
    if not config_path.exists():
        return default_config

    loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "service_level": loaded_config.get("service_level", default_config["service_level"]),
        "lead_time": loaded_config.get("lead_time", default_config["lead_time"]),
        "forecast_horizon": loaded_config.get("forecast_horizon", default_config["forecast_horizon"]),
    }


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load raw demand data from disk."""
    return pd.read_csv(csv_path)


def run_forecast_selection(
    cleaned_data: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select the best supported forecasting method per product."""
    historical_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    planning_frames: list[pd.DataFrame] = []
    method_rows: list[dict[str, object]] = []
    top_candidate_frames: list[pd.DataFrame] = []

    for product, group in cleaned_data.groupby("product", sort=True):
        try:
            lead = float(group.sort_values("date").iloc[-1]["lead_time_days"])
            output_horizon = max(horizon, required_planning_steps(group, lead))
        except (ValueError, TypeError, OverflowError):
            output_horizon = horizon  # Core engine will explain why planning is withheld.
        result: ProductForecastResult = select_best_forecast_for_product(
            group, horizon=horizon, output_horizon=output_horizon,
        )
        historical_frames.append(result.historical)
        future_frames.append(result.future.head(horizon).copy())
        planning_frames.append(result.future)
        top_candidate_frames.append(result.top_candidates)
        method_rows.append(
            {
                "product": product,
                "selected_method": result.model_name,
                "selected_arima_order": result.selected_arima_order,
                "mae": round(result.mae, 2),
                "mape": round(result.mape, 2),
                "validation_points": result.validation_points,
                "candidate_count": result.candidate_count,
                "runner_up_model": result.runner_up_model,
                "runner_up_arima_order": result.runner_up_arima_order,
                "trend_detected": result.trend_detected,
                "seasonality_detected": result.seasonality_detected,
                "selection_reason": result.selection_reason,
            }
        )

    historical = pd.concat(historical_frames, ignore_index=True)
    future = pd.concat(future_frames, ignore_index=True) if future_frames else pd.DataFrame()
    method_summary = pd.DataFrame(method_rows)
    top_candidates = pd.concat(top_candidate_frames, ignore_index=True) if top_candidate_frames else pd.DataFrame()
    planning = pd.concat(planning_frames, ignore_index=True)
    return historical, future, method_summary, top_candidates, planning


def calculate_accuracy_metrics(actual_vs_forecast: pd.DataFrame, method_summary: pd.DataFrame) -> pd.DataFrame:
    """Use selected-model holdout errors everywhere; overall is the product mean."""
    historical_stats = (
        actual_vs_forecast.groupby("product", as_index=False)
        .agg(demand_std=("demand", "std"), mean_demand=("demand", "mean"))
        .fillna({"demand_std": 0})
    )
    per_product = method_summary[
        ["product", "mae", "mape", "selected_method", "trend_detected",
         "seasonality_detected", "selection_reason", "validation_points"]
    ].merge(historical_stats, on="product", how="left")
    per_product["accuracy_basis"] = "Selected-model holdout"
    overall = pd.DataFrame([{
        "product": "Overall",
        "mae": per_product["mae"].mean(),
        "mape": per_product["mape"].mean(),
        "demand_std": actual_vs_forecast["demand"].std(),
        "mean_demand": actual_vs_forecast["demand"].mean(),
        "selected_method": "Mixed by product",
        "trend_detected": pd.NA,
        "seasonality_detected": pd.NA,
        "selection_reason": "Each product uses its own best-performing method.",
        "validation_points": per_product["validation_points"].sum(),
        "accuracy_basis": "Unweighted mean of product holdout errors",
    }])
    metrics = pd.concat([per_product, overall], ignore_index=True)
    for column in ["mae", "mape", "demand_std", "mean_demand"]:
        metrics[column] = metrics[column].round(2)
    return metrics


def calculate_product_insights(cleaned_data: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.Series]:
    """Derive business-oriented product rankings from demand and forecast results."""
    product_totals = cleaned_data.groupby("product", as_index=False)["demand"].sum().sort_values("demand", ascending=False)
    highest_demand = product_totals.iloc[0]
    product_metrics = metrics[metrics["product"] != "Overall"].copy()
    highest_variance = product_metrics.sort_values("demand_std", ascending=False).iloc[0]
    worst_accuracy = product_metrics.sort_values(["mape", "mae"], ascending=False).iloc[0]
    return {
        "highest_demand": highest_demand,
        "highest_variance": highest_variance,
        "worst_accuracy": worst_accuracy,
    }


def normalize_service_level(service_level_input: float) -> float:
    """Convert a service level input like 95 or 0.95 into a decimal."""
    service_level = service_level_input / 100 if service_level_input > 1 else service_level_input
    if not 0 < service_level < 1:
        raise ValueError("service level must be between 0 and 100 percent")
    return service_level


def calculate_inventory_metrics(
    cleaned_data: pd.DataFrame,
    future_forecast: pd.DataFrame,
    service_level: float,
    lead_time: int,
    planning_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine core planning with isolated, explicitly labeled legacy diagnostics."""
    core = calculate_core_inventory_metrics(
        cleaned_data, future_forecast if planning_forecast is None else planning_forecast,
        service_level, lead_time,
    )
    legacy = calculate_legacy_perishability_metrics(
        cleaned_data, future_forecast, service_level, lead_time,
    )
    inventory = core.merge(legacy, on="product", how="left")
    inventory["legacy_perishability_note"] = (
        "Legacy spoilage/promotion diagnostics; not used for replenishment actions or quantities."
    )
    # The legacy score assumes daily observations and known valid stock.
    unavailable = (
        (inventory["inventory_status"] != "known")
        | (inventory["forecast_frequency"] != "D")
        | (inventory["planning_status"] != "ready")
    )
    inventory.loc[unavailable, "spoilage_risk_score"] = float("nan")
    inventory.loc[unavailable, "spoilage_risk"] = "Unknown"
    for column in ["at_risk_of_expiring_before_sale", "expiring_before_expected_sale", "spoilage_risk_flag"]:
        inventory[column] = inventory[column].astype("boolean")
        inventory.loc[unavailable, column] = pd.NA
    return inventory


def build_summary(
    cleaned_data: pd.DataFrame,
    metrics: pd.DataFrame,
    insights: dict[str, pd.Series],
    inventory_metrics: pd.DataFrame,
    method_summary: pd.DataFrame,
    config: dict[str, float | int],
) -> str:
    """Report calculated planning decisions without inferred stockout probabilities."""
    overall = metrics.loc[metrics["product"] == "Overall"].iloc[0]
    lines = [
        "Business Forecast and Inventory Planning Summary",
        f"Cleaned observations: {len(cleaned_data)}; products: {cleaned_data['product'].nunique()}.",
        f"Highest total historical demand: {insights['highest_demand']['product']}.",
        f"Mean product holdout MAE: {overall['mae']:.2f}; MAPE: {overall['mape']:.2f}%.",
        f"Displayed forecast horizon: {config['forecast_horizon']} native periods.",
        "Planning forecasts independently cover each product's lead time plus one calendar day.",
        "Policy: daily periodic review; target = protection-period demand + planning safety stock.",
        "Reorder point = lead-time demand + lead-time safety stock; it does not size the order.",
        INVENTORY_ASSUMPTION,
        "Safety stock uses historical daily variability with independent demand increments and fixed lead time.",
        "The service level is a target, not an estimated actual stockout probability.",
        "Reduce replenishment / run down excess means defer purchasing, not dispose of inventory.",
        "Potential pre-arrival shortages may require earlier supply; ordinary replenishment may arrive too late.",
        "Legacy perishability diagnostics are separate and do not determine core planning actions.",
        "", "Forecast model selection:",
    ]
    for _, row in method_summary.sort_values("product").iterrows():
        lines.append(f"{row['product']}: {row['selected_method']}. {row['selection_reason']}")
    lines.extend(["", "Inventory planning:"])
    for _, row in inventory_metrics.iterrows():
        target = f"{row['order_up_to_target']:.6g}" if pd.notna(row['order_up_to_target']) else "unavailable"
        reorder = f"{row['reorder_point']:.6g}" if pd.notna(row['reorder_point']) else "unavailable"
        lines.append(
            f"{row['product']}: lead time {row['lead_time_days']:g} days ({row['lead_time_source']}), "
            f"protection period {row['protection_period_days']:g} days, "
            f"reorder point {reorder}, target {target}. {row['reasoning']}"
        )
    return "\n".join(lines)


def save_forecast_csv(actual_vs_forecast: pd.DataFrame, future_forecast: pd.DataFrame, output_dir: Path) -> Path:
    """Write a combined forecast file to disk."""
    combined = pd.concat([actual_vs_forecast, future_forecast], ignore_index=True)
    combined = combined.astype(
        {
            "date": "datetime64[ns]",
            "product": "string",
            "demand": "float64",
            "forecast": "float64",
            "type": "string",
            "selected_method": "string",
            "selected_arima_order": "string",
            "trend_detected": "boolean",
            "seasonality_detected": "boolean",
            "selection_reason": "string",
        }
    )
    combined = combined.sort_values(["product", "date", "type"]).reset_index(drop=True)
    combined["demand"] = combined["demand"].round(2)
    combined["forecast"] = combined["forecast"].round(2)
    output_path = output_dir / "forecast_output.csv"
    combined.to_csv(output_path, index=False)
    return output_path


def save_cleaned_dataset(cleaned_data: pd.DataFrame, output_dir: Path) -> Path:
    """Write the cleaned, schema-standardized input dataset to disk."""
    output_path = output_dir / "cleaned_standardized_data.csv"
    cleaned_data.to_csv(output_path, index=False)
    return output_path


def save_metrics_csv(metrics: pd.DataFrame, output_dir: Path) -> Path:
    """Write forecast accuracy metrics to disk."""
    output_path = output_dir / "forecast_accuracy.csv"
    metrics.to_csv(output_path, index=False)
    return output_path


def save_inventory_metrics_csv(inventory_metrics: pd.DataFrame, output_dir: Path) -> Path:
    """Write inventory planning metrics to disk."""
    output_path = output_dir / "inventory_metrics.csv"
    normalize_output_schema(inventory_metrics).to_csv(output_path, index=False)
    return output_path


def save_method_selection_csv(method_summary: pd.DataFrame, output_dir: Path) -> Path:
    """Write per-product model selection results to disk."""
    output_path = output_dir / "forecast_model_selection.csv"
    method_summary.to_csv(output_path, index=False)
    return output_path


def save_top_candidates_csv(top_candidates: pd.DataFrame, output_dir: Path) -> Path:
    """Write the top ranked forecast candidates per product to disk."""
    output_path = output_dir / "forecast_top_candidates.csv"
    top_candidates.to_csv(output_path, index=False)
    return output_path


def save_summary(summary_text: str, output_dir: Path) -> Path:
    """Write the plain-English summary to disk."""
    output_path = output_dir / "forecast_summary.txt"
    output_path.write_text(summary_text + "\n", encoding="utf-8")
    return output_path


def plot_actual_vs_forecast(actual_vs_forecast: pd.DataFrame, future_forecast: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create one clearly labeled chart per product."""
    products = sorted(actual_vs_forecast["product"].dropna().unique())
    chart_paths: list[Path] = []

    for product in products:
        historical = actual_vs_forecast[actual_vs_forecast["product"] == product]
        future = future_forecast[future_forecast["product"] == product]
        method_name = historical["selected_method"].iloc[0]
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.plot(historical["date"], historical["demand"], label="Actual demand", marker="o")
        axis.plot(historical["date"], historical["forecast"], label=f"{method_name} fit", linestyle="--", linewidth=2)
        axis.plot(future["date"], future["forecast"], label="Future forecast", marker="x", linestyle="-.", linewidth=2)
        axis.set_title(f"Demand Forecast for {product} ({method_name})")
        axis.set_xlabel("Date")
        axis.set_ylabel("Demand")
        axis.grid(alpha=0.3)
        axis.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        chart_path = output_dir / f"{product.lower().replace(' ', '_')}_forecast.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        chart_paths.append(chart_path)

    return chart_paths


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    config: dict[str, float | int],
) -> PipelineRunResult:
    """Run the full forecasting workflow and return saved outputs plus in-memory tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    service_level = normalize_service_level(float(config["service_level"]))
    lead_time = int(config["lead_time"])
    forecast_horizon = int(config["forecast_horizon"])

    raw_data = load_data(input_path)
    cleaned_data = clean_and_preprocess_data(raw_data, default_lead_time_days=lead_time)

    actual_vs_forecast, future_forecast, method_summary, top_candidates, planning_forecast = run_forecast_selection(
        cleaned_data,
        horizon=forecast_horizon,
    )
    metrics = calculate_accuracy_metrics(actual_vs_forecast, method_summary)
    insights = calculate_product_insights(cleaned_data, metrics)
    inventory_metrics = calculate_inventory_metrics(
        cleaned_data,
        future_forecast,
        service_level=service_level,
        lead_time=lead_time,
        planning_forecast=planning_forecast,
    )
    inventory_metrics = normalize_output_schema(inventory_metrics)
    summary_text = build_summary(cleaned_data, metrics, insights, inventory_metrics, method_summary, config)

    planning_forecast_path = output_dir / "planning_forecast.csv"
    planning_forecast.to_csv(planning_forecast_path, index=False)
    forecast_csv_path = save_forecast_csv(actual_vs_forecast, future_forecast, output_dir)
    cleaned_dataset_path = save_cleaned_dataset(cleaned_data, output_dir)
    metrics_csv_path = save_metrics_csv(metrics, output_dir)
    inventory_metrics_path = save_inventory_metrics_csv(inventory_metrics, output_dir)
    method_selection_path = save_method_selection_csv(method_summary, output_dir)
    top_candidates_path = save_top_candidates_csv(top_candidates, output_dir)
    summary_path = save_summary(summary_text, output_dir)
    chart_paths = plot_actual_vs_forecast(actual_vs_forecast, future_forecast, output_dir)
    inventory_chart_paths = create_inventory_charts(inventory_metrics, output_dir)

    return PipelineRunResult(
        cleaned_data=cleaned_data,
        run_config={**config, "service_level": service_level * 100},
        planning_forecast=planning_forecast,
        planning_forecast_path=planning_forecast_path,
        actual_vs_forecast=actual_vs_forecast,
        future_forecast=future_forecast,
        method_summary=method_summary,
        top_candidates=top_candidates,
        metrics=metrics,
        inventory_metrics=inventory_metrics,
        summary_text=summary_text,
        forecast_csv_path=forecast_csv_path,
        cleaned_dataset_path=cleaned_dataset_path,
        metrics_csv_path=metrics_csv_path,
        inventory_metrics_path=inventory_metrics_path,
        method_selection_path=method_selection_path,
        top_candidates_path=top_candidates_path,
        summary_path=summary_path,
        chart_paths=chart_paths,
        inventory_chart_paths=inventory_chart_paths,
    )


def main() -> None:
    """Run the full forecasting workflow."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    config = load_config(config_path)
    result = run_pipeline(input_path=input_path, output_dir=output_dir, config=config)

    print(f"Cleaned rows processed: {len(result.cleaned_data)}")
    print(f"Cleaned standardized dataset saved to: {result.cleaned_dataset_path}")
    print(f"Forecast CSV saved to: {result.forecast_csv_path}")
    print(f"Accuracy CSV saved to: {result.metrics_csv_path}")
    print(f"Inventory metrics CSV saved to: {result.inventory_metrics_path}")
    print(f"Model selection CSV saved to: {result.method_selection_path}")
    print(f"Top candidates CSV saved to: {result.top_candidates_path}")
    print(f"Config used: {config_path}")
    print(f"Summary saved to: {result.summary_path}")
    print(f"Forecast charts saved to: {', '.join(str(path) for path in result.chart_paths)}")
    print(f"Inventory charts saved to: {', '.join(str(path) for path in result.inventory_chart_paths)}")


if __name__ == "__main__":
    main()
