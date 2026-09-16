"""Streamlit UI for the supply chain forecasting project."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import clean_and_preprocess_data  # noqa: E402
from ai_assistant import DEFAULT_OLLAMA_MODEL, render_ai_assistant  # noqa: E402
from demand_forecasting import PipelineRunResult, run_pipeline  # noqa: E402
from inventory_planning import INVENTORY_ASSUMPTION  # noqa: E402
from schema_normalization import normalize_output_schema  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOADED_INPUT_PATH = OUTPUT_DIR / "uploaded_input.csv"
# Optional AI assistant toggle. Remove this flag, the import above, and the AI tab below to disable cleanly.
ENABLE_AI_ASSISTANT = True
OLLAMA_MODEL_NAME = DEFAULT_OLLAMA_MODEL


def inject_app_styles() -> None:
    """Inject lightweight CSS for a cleaner dashboard layout."""
    st.markdown(
        """
        <style>
        .hero-card {
            border-radius: 18px;
            padding: 1.25rem 1.25rem 0.5rem 1.25rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #134e4a 100%);
            border: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.28);
            margin-bottom: 1rem;
        }
        .hero-eyebrow {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #93c5fd;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.35rem;
        }
        .hero-copy {
            font-size: 0.98rem;
            color: #cbd5e1;
            margin-bottom: 0.9rem;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1.6rem;
            margin-bottom: 0.75rem;
            color: #f8fafc;
        }
        .summary-card {
            border-radius: 14px;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: #17202c;
            min-height: 108px;
            margin-bottom: 0.8rem;
            color: #f8fafc;
        }
        .summary-label {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #93c5fd;
            margin-bottom: 0.4rem;
        }
        .summary-value {
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }
        .summary-detail {
            font-size: 0.9rem;
            color: #cbd5e1;
        }
        .kpi-card {
            border-radius: 14px;
            padding: 1.1rem 1rem 1rem 1rem;
            border: 1px solid rgba(255, 255, 255, 0.10);
            min-height: 120px;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
            color: #f8fafc;
        }
        .kpi-label {
            font-size: 0.95rem;
            font-weight: 600;
            color: inherit;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.1;
            white-space: normal;
            word-break: break-word;
            color: inherit;
        }
        .kpi-subtle {
            background: #2c2f36;
            color: #ffffff;
        }
        .kpi-good {
            background: #1f7a4d;
            border-color: #2dd47a;
            color: #ffffff;
        }
        .kpi-warn {
            background: #b58900;
            border-color: #ffd24d;
            color: #111111;
        }
        .kpi-bad {
            background: #b02a37;
            border-color: #ff8fa3;
            color: #ffffff;
        }
        .alert-box {
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left-width: 6px;
            font-size: 0.95rem;
            font-weight: 600;
            color: #f8fafc;
        }
        .alert-high {
            background: #3a1a1d;
            border-left-color: #ef4444;
        }
        .alert-medium {
            background: #3a2f1a;
            border-left-color: #facc15;
        }
        .alert-low {
            background: #1a3a2a;
            border-left-color: #22c55e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_download_button(label: str, file_path: Path, mime: str) -> None:
    """Render a download button for an output file if it exists."""
    if not file_path.exists():
        return
    st.download_button(label=label, data=file_path.read_bytes(), file_name=file_path.name, mime=mime)


def render_images(image_paths: list[Path], columns: int = 2) -> None:
    """Display generated charts in a grid layout."""
    existing_images = [path for path in sorted(image_paths) if path.exists()]
    if not existing_images:
        st.info("No charts were generated.")
        return
    image_columns = st.columns(columns)
    for index, image_path in enumerate(existing_images):
        with image_columns[index % columns]:
            st.image(str(image_path), caption=image_path.stem.replace("_", " ").title(), use_container_width=True)


def resolve_first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from the candidate list."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def build_kpi_metrics(result: PipelineRunResult) -> list[dict[str, str]]:
    """Keep service gaps and unavailable plans distinct from legacy spoilage scores."""
    inventory = result.inventory_metrics
    overall = result.metrics.loc[result.metrics["product"] == "Overall"].iloc[0]
    gaps = int(inventory["pre_arrival_shortage"].gt(0).sum())
    unavailable = int(inventory["planning_status"].ne("ready").sum())
    return [
        {"label": "Holdout MAPE", "value": f"{overall['mape']:.2f}%", "tone": "subtle"},
        {"label": "Products Reviewed", "value": str(len(inventory)), "tone": "subtle"},
        {"label": "Pre-arrival Service Gaps", "value": str(gaps), "tone": "bad" if gaps else "good"},
        {"label": "Plans Awaiting Data", "value": str(unavailable), "tone": "warn" if unavailable else "good"},
    ]


def build_executive_summary(result: PipelineRunResult) -> list[str]:
    """Describe the policy and actual core decisions without inferred causes."""
    inventory = result.inventory_metrics
    ready = inventory[inventory["planning_status"] == "ready"]
    return [
        "Inventory is reviewed daily. Each target covers product lead time plus one calendar day, including planning safety stock.",
        INVENTORY_ASSUMPTION,
        f"{len(ready)} of {len(inventory)} products have sufficient data for an inventory recommendation; "
        f"{int(ready['recommended_action'].eq('Increase').sum())} require replenishment.",
        f"{int(inventory['pre_arrival_shortage'].gt(0).sum())} products have a potential pre-arrival service gap. "
        "Ordinary replenishment may not arrive soon enough to prevent it.",
        "Excess means defer replenishment and run down stock through demand, not disposal. "
        "Legacy spoilage and promotion diagnostics do not determine these quantities.",
    ]


def build_top_alerts(result: PipelineRunResult) -> list[dict[str, str]]:
    """Surface service gaps and missing-data issues before routine replenishment."""
    inventory = result.inventory_metrics.copy()
    inventory["_sort"] = inventory["pre_arrival_shortage"].gt(0).astype(int) * 2 + inventory["planning_status"].ne("ready").astype(int)
    alerts = []
    for _, row in inventory.sort_values(["_sort", "pre_arrival_shortage"], ascending=False).head(3).iterrows():
        if row["planning_status"] != "ready":
            tone, message = "medium", f"{row['product']}: recommendation unavailable. {row['planning_issue']}"
        elif row["pre_arrival_shortage"] > 0:
            tone, message = "high", (
                f"{row['product']}: potential pre-arrival service gap of {row['pre_arrival_shortage']:.2f} units. "
                "Ordinary replenishment may arrive too late."
            )
        else:
            tone, message = "low", (
                f"{row['product']}: {row['recommended_action']}; "
                f"{int(row['recommended_quantity_adjustment'])} whole units."
            )
        alerts.append({"tone": tone, "message": message})
    return alerts


def infer_forecast_quality_label(mape_value: float) -> tuple[str, str]:
    """Map overall MAPE to a simple quality label."""
    if mape_value < 10:
        return "High confidence", "Forecast error is low enough to support near-term planning decisions."
    if mape_value < 20:
        return "Moderate confidence", "Forecast accuracy is usable, but planners should monitor high-risk items closely."
    return "Low confidence", "Forecast error is elevated, so planning decisions should be reviewed with added caution."


def estimate_forecast_interval_margin(
    historical_demand: pd.DataFrame,
    historical_forecast: pd.DataFrame,
) -> tuple[float, bool]:
    """Estimate a simple forecast interval width from residuals or demand variability."""
    required_demand_columns = {"date", "demand"}
    required_forecast_columns = {"date", "forecast"}

    if required_demand_columns.issubset(historical_demand.columns) and required_forecast_columns.issubset(historical_forecast.columns):
        residual_frame = historical_demand[["date", "demand"]].merge(
            historical_forecast[["date", "forecast"]],
            on="date",
            how="inner",
        )
        residual_frame["residual"] = residual_frame["demand"] - residual_frame["forecast"]
        residual_std = float(residual_frame["residual"].std(ddof=0)) if not residual_frame.empty else 0.0
        if residual_std > 0:
            return round(1.96 * residual_std, 2), True

    if required_demand_columns.issubset(historical_demand.columns) and not historical_demand.empty:
        rolling_std = historical_demand["demand"].rolling(window=min(4, len(historical_demand)), min_periods=2).std(ddof=0)
        fallback_std = float(rolling_std.dropna().iloc[-1]) if not rolling_std.dropna().empty else float(historical_demand["demand"].std(ddof=0))
        if fallback_std > 0:
            return round(1.96 * fallback_std, 2), True

    return 0.0, True


def build_forecast_chart_dataset(
    chart_frame: pd.DataFrame,
    interval_margin: float,
) -> pd.DataFrame:
    """Prepare a chart dataset with estimated confidence bands."""
    chart_data = chart_frame.reset_index().copy()
    chart_data["Forecast lower bound"] = pd.NA
    chart_data["Forecast upper bound"] = pd.NA

    if "Forecasted demand" in chart_data.columns and interval_margin > 0:
        forecast_values = pd.to_numeric(chart_data["Forecasted demand"], errors="coerce")
        chart_data["Forecast lower bound"] = (forecast_values - interval_margin).clip(lower=0).round(2)
        chart_data["Forecast upper bound"] = (forecast_values + interval_margin).round(2)

    return chart_data


def render_forecast_chart(chart_frame: pd.DataFrame, interval_margin: float, chart_height: int) -> None:
    """Render a clean interactive demand chart with a shaded confidence band."""
    chart_data = build_forecast_chart_dataset(chart_frame, interval_margin)
    if chart_data.empty:
        st.info("No chart data is available for this forecast.")
        return

    figure = go.Figure()
    forecast_hovertemplate = (
        "Date: %{x|%Y-%m-%d}<br>"
        "Forecast: %{y:.2f}<br>"
        "Upper bound: %{customdata[0]:.2f}<br>"
        "Lower bound: %{customdata[1]:.2f}<extra></extra>"
    )

    if {"Forecast lower bound", "Forecast upper bound"}.issubset(chart_data.columns):
        interval_data = chart_data.dropna(subset=["Forecast lower bound", "Forecast upper bound"]).copy()
        if not interval_data.empty:
            figure.add_trace(
                go.Scatter(
                    x=interval_data["date"],
                    y=interval_data["Forecast upper bound"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Upper bound",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=interval_data["date"],
                    y=interval_data["Forecast lower bound"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(96, 165, 250, 0.18)",
                    hoverinfo="skip",
                    showlegend=False,
                    name="Confidence band",
                )
            )

    if "Historical demand" in chart_data.columns:
        historical_data = chart_data.dropna(subset=["Historical demand"])
        if not historical_data.empty:
            figure.add_trace(
                go.Scatter(
                    x=historical_data["date"],
                    y=historical_data["Historical demand"],
                    mode="lines",
                    name="Historical demand",
                    line=dict(color="#f8fafc", width=2.4),
                    hovertemplate=(
                        "Date=%{x|%Y-%m-%d}<br>"
                        "Historical demand=%{y:.2f}<extra></extra>"
                    ),
                )
            )

    if "Forecasted demand" in chart_data.columns:
        forecast_data = chart_data.dropna(subset=["Forecasted demand"])
        if not forecast_data.empty:
            figure.add_trace(
                go.Scatter(
                    x=forecast_data["date"],
                    y=forecast_data["Forecasted demand"],
                    mode="lines",
                    name="Forecasted demand",
                    line=dict(color="#38bdf8", width=2.4, dash="dash"),
                    customdata=forecast_data[["Forecast upper bound", "Forecast lower bound"]].to_numpy(),
                    hovertemplate=forecast_hovertemplate,
                )
            )

    if not figure.data:
        st.info("No chart data is available for this forecast.")
        return

    figure.update_layout(
        height=chart_height,
        margin=dict(l=12, r=12, t=12, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        ),
        xaxis=dict(
            title="Date",
            showgrid=False,
            zeroline=False,
            color="#cbd5e1",
        ),
        yaxis=dict(
            title="Demand",
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            color="#cbd5e1",
        ),
    )
    st.plotly_chart(figure, use_container_width=True, config={"responsive": True})


def build_forecast_summary_metrics(result: PipelineRunResult) -> list[dict[str, str]]:
    """Build top-level metrics that explain the current forecast outlook."""
    overall_metrics = result.metrics.loc[result.metrics["product"] == "Overall"].iloc[0]
    total_by_date = result.cleaned_data.groupby("date", as_index=False)["demand"].sum().sort_values("date")
    first_total = float(total_by_date.iloc[0]["demand"])
    last_total = float(total_by_date.iloc[-1]["demand"])
    trend_pct = ((last_total - first_total) / first_total * 100) if first_total else 0.0
    if trend_pct > 1:
        trend_direction = "Rising"
    elif trend_pct < -1:
        trend_direction = "Softening"
    else:
        trend_direction = "Stable"

    if result.future_forecast.empty:
        forecast_horizon_value = "Not available"
        forecast_horizon_detail = "No future periods were returned by the current pipeline run."
    else:
        horizon_periods = int(result.future_forecast.groupby("product")["date"].nunique().max())
        forecast_horizon_value = f"{horizon_periods} periods"
        forecast_horizon_detail = "Forecast horizon inferred from the future forecast output."

    quality_label, quality_detail = infer_forecast_quality_label(float(overall_metrics["mape"]))

    return [
        {
            "label": "Holdout MAPE",
            "value": f"{overall_metrics['mape']:.2f}%",
            "detail": "Mean of product holdout MAPE values; lower is better.",
        },
        {
            "label": "Forecast Horizon",
            "value": forecast_horizon_value,
            "detail": forecast_horizon_detail,
        },
        {
            "label": "Demand Trend",
            "value": trend_direction,
            "detail": f"Historical demand changed by {trend_pct:.1f}% across the observed period.",
        },
        {
            "label": "Forecast Quality",
            "value": quality_label,
            "detail": quality_detail,
        },
    ]


def build_forecast_chart_frame(result: PipelineRunResult) -> tuple[pd.DataFrame, str]:
    """Create a clean aggregate historical-versus-forecast view for the hero chart."""
    historical = (
        result.cleaned_data.groupby("date", as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "Historical demand"})
    )

    if result.future_forecast.empty:
        fitted = (
            result.actual_vs_forecast.groupby("date", as_index=False)["forecast"]
            .sum()
            .rename(columns={"forecast": "Forecasted demand"})
        )
        chart_frame = historical.merge(fitted, on="date", how="outer").sort_values("date").set_index("date")
        return chart_frame, "Historical demand is shown against the in-sample fitted forecast because no future horizon output is available."

    future = (
        result.future_forecast.groupby("date", as_index=False)["forecast"]
        .sum()
        .rename(columns={"forecast": "Forecasted demand"})
    )
    chart_frame = historical.merge(future, on="date", how="outer").sort_values("date").set_index("date")
    return chart_frame, "Historical demand is aggregated across products and extended with the future forecast horizon."


def build_product_forecast_chart_frame(result: PipelineRunResult, product_name: str) -> tuple[pd.DataFrame | None, str]:
    """Create a product-level historical-versus-forecast view for the dashboard."""
    required_historical_columns = {"date", "product", "demand"}
    if not required_historical_columns.issubset(result.cleaned_data.columns):
        return None, "Product-level historical demand is unavailable because cleaned data is missing required columns."

    historical = (
        result.cleaned_data.loc[result.cleaned_data["product"] == product_name, ["date", "demand"]]
        .groupby("date", as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "Historical demand"})
    )
    if historical.empty:
        return None, f"No historical demand is available for {product_name}."

    required_future_columns = {"date", "product", "forecast"}
    if required_future_columns.issubset(result.future_forecast.columns):
        future = (
            result.future_forecast.loc[result.future_forecast["product"] == product_name, ["date", "forecast"]]
            .groupby("date", as_index=False)["forecast"]
            .sum()
            .rename(columns={"forecast": "Forecasted demand"})
        )
        if not future.empty:
            chart_frame = historical.merge(future, on="date", how="outer").sort_values("date").set_index("date")
            return chart_frame, f"Historical demand for {product_name} is extended with the selected future forecast horizon."

    required_fitted_columns = {"date", "product", "forecast"}
    if required_fitted_columns.issubset(result.actual_vs_forecast.columns):
        fitted = (
            result.actual_vs_forecast.loc[result.actual_vs_forecast["product"] == product_name, ["date", "forecast"]]
            .groupby("date", as_index=False)["forecast"]
            .sum()
            .rename(columns={"forecast": "Forecasted demand"})
        )
        if not fitted.empty:
            chart_frame = historical.merge(fitted, on="date", how="outer").sort_values("date").set_index("date")
            return chart_frame, f"Historical demand for {product_name} is shown against the in-sample fitted forecast because no future forecast rows are available."

    return historical.sort_values("date").set_index("date"), f"Only historical demand is available for {product_name} in the current pipeline output."


def build_aggregate_historical_demand(result: PipelineRunResult) -> pd.DataFrame:
    """Return aggregate historical demand by date."""
    return result.cleaned_data.groupby("date", as_index=False)["demand"].sum()


def build_aggregate_historical_forecast(result: PipelineRunResult) -> pd.DataFrame:
    """Return aggregate fitted forecast by date."""
    return result.actual_vs_forecast.groupby("date", as_index=False)["forecast"].sum()


def build_product_historical_demand(result: PipelineRunResult, product_name: str) -> pd.DataFrame:
    """Return product-level historical demand by date."""
    return (
        result.cleaned_data.loc[result.cleaned_data["product"] == product_name, ["date", "demand"]]
        .groupby("date", as_index=False)["demand"]
        .sum()
    )


def build_product_historical_forecast(result: PipelineRunResult, product_name: str) -> pd.DataFrame:
    """Return product-level fitted forecast by date."""
    return (
        result.actual_vs_forecast.loc[result.actual_vs_forecast["product"] == product_name, ["date", "forecast"]]
        .groupby("date", as_index=False)["forecast"]
        .sum()
    )


def render_forecast_summary(result: PipelineRunResult) -> None:
    """Render the forecast summary metric row."""
    st.markdown('<div class="section-title">Forecast Summary</div>', unsafe_allow_html=True)
    for column, metric in zip(st.columns(4), build_forecast_summary_metrics(result)):
        with column:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-label">{metric['label']}</div>
                    <div class="summary-value">{metric['value']}</div>
                    <div class="summary-detail">{metric['detail']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_forecast_hero(result: PipelineRunResult) -> None:
    """Render the primary forecast-first chart and framing copy."""
    overall_metrics = result.metrics.loc[result.metrics["product"] == "Overall"].iloc[0]
    quality_label, _ = infer_forecast_quality_label(float(overall_metrics["mape"]))
    chart_frame, caption = build_forecast_chart_frame(result)
    interval_margin, interval_is_estimated = estimate_forecast_interval_margin(
        build_aggregate_historical_demand(result),
        build_aggregate_historical_forecast(result),
    )

    st.markdown('<div class="section-title">Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">Forecast Outlook</div>
            <div class="hero-title">Demand signal first, inventory action second</div>
            <div class="hero-copy">
                The current forecast is the lead planning signal for this run. Overall forecast quality is <strong>{quality_label}</strong>,
                and downstream recommendations are derived from where projected demand diverges from current inventory.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_forecast_chart(chart_frame, interval_margin=interval_margin, chart_height=380)
    st.caption(caption)
    if interval_is_estimated and interval_margin > 0:
        st.caption("Shaded band: estimated planning range from historical forecast error.")


def render_product_forecast_detail(result: PipelineRunResult) -> None:
    """Render a product-level demand forecast drill-down in the dashboard."""
    required_columns = {"product", "date", "demand"}
    if not required_columns.issubset(result.cleaned_data.columns):
        st.warning("Product-level forecast view is unavailable because cleaned data is missing product, date, or demand columns.")
        return

    product_options = sorted(result.cleaned_data["product"].dropna().astype(str).unique().tolist())
    if not product_options:
        st.info("No products are available for product-level forecast review.")
        return

    st.markdown('<div class="section-title">Product Forecast Detail</div>', unsafe_allow_html=True)
    selected_product = st.selectbox("Select product", product_options, key="dashboard_selected_product")
    chart_frame, caption = build_product_forecast_chart_frame(result, selected_product)
    if chart_frame is None or chart_frame.empty:
        st.info(caption)
        return

    inventory_metrics = normalize_output_schema(result.inventory_metrics)
    product_inventory = inventory_metrics.loc[inventory_metrics["product"] == selected_product]
    product_lead_time = None
    if not product_inventory.empty:
        lead_time_column = resolve_first_existing_column(product_inventory, ["lead_time_days", "lead_time"])
        if lead_time_column is not None:
            lead_time_value = pd.to_numeric(product_inventory.iloc[0][lead_time_column], errors="coerce")
            if pd.notna(lead_time_value):
                product_lead_time = float(lead_time_value)

    interval_margin, interval_is_estimated = estimate_forecast_interval_margin(
        build_product_historical_demand(result, selected_product),
        build_product_historical_forecast(result, selected_product),
    )
    render_forecast_chart(chart_frame, interval_margin=interval_margin, chart_height=320)
    st.caption(caption)
    if product_lead_time is not None:
        st.caption(f"Resolved lead time for {selected_product}: {product_lead_time:g} calendar days.")
    if interval_is_estimated and interval_margin > 0:
        st.caption("Shaded band: estimated range from historical forecast error.")


def simplify_action_label(action_text: str | None) -> str:
    """Preserve canonical decisions; unknown inventory is not a Maintain action."""
    return str(action_text) if pd.notna(action_text) else "Unavailable — see planning issue"


def render_kpis(result: PipelineRunResult) -> None:
    """Render the dashboard KPI row."""
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    metrics = build_kpi_metrics(result)
    for column, metric in zip(st.columns(len(metrics)), metrics):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card kpi-{metric['tone']}">
                    <div class="kpi-label">{metric['label']}</div>
                    <div class="kpi-value">{metric['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


DISPLAY_LABELS = {
    "on_hand_inventory": "On-Hand Inventory", "lead_time_days": "Lead Time (Days)",
    "promo_flag": "Promotion", "risk_level": "Demand Volatility",
    "order_up_to_target": "Target Inventory", "recommended_action": "Action",
    "recommended_quantity_adjustment": "Recommended Qty", "daily_demand_std": "Daily Demand Std. Dev.",
    "expected_demand_protection_period": "Protection-Period Demand",
    "expected_demand_lead_time": "Lead-Time Demand", "mae": "Holdout MAE", "mape": "Holdout MAPE (%)",
    "selected_arima_order": "ARIMA Order", "runner_up_arima_order": "Runner-Up ARIMA Order",
}


def readable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use readable labels on a display copy, preserving backend schemas."""
    return frame.rename(columns={
        column: DISPLAY_LABELS.get(column, str(column).replace("_", " ").title())
        for column in frame.columns
    })


def inventory_action_frame(inventory: pd.DataFrame) -> pd.DataFrame:
    """Seven business-facing columns; all rounding is display-only."""
    columns = {
        "product": "Product", "on_hand_inventory": "On Hand", "lead_time_days": "Lead Time",
        "reorder_point": "Reorder Point", "order_up_to_target": "Target Inventory",
        "recommended_action": "Action", "recommended_quantity_adjustment": "Recommended Qty",
    }
    frame = inventory[list(columns)].copy()
    frame["recommended_action"] = frame["recommended_action"].fillna("Unavailable").replace(
        {"Reduce replenishment / run down excess": "Run down excess"}
    )
    for column in ["on_hand_inventory", "reorder_point", "order_up_to_target", "recommended_quantity_adjustment"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(0)
    return frame.rename(columns=columns)


def render_inventory_decisions(inventory: pd.DataFrame) -> None:
    """Shared presentation for the overview and business-action tab."""
    st.dataframe(
        inventory_action_frame(inventory), width="stretch", hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(name, format="%.0f")
            for name in ["On Hand", "Reorder Point", "Target Inventory", "Recommended Qty"]
        } | {"Lead Time": st.column_config.NumberColumn("Lead Time", format="%g days")},
    )
    st.caption("Rounded units shown. Run down excess means defer replenishment, not disposal.")
    unavailable = inventory[inventory["planning_status"] != "ready"]
    if not unavailable.empty:
        st.info(f"{len(unavailable)} recommendation(s) unavailable. See calculation details for missing or invalid inputs.")
    with st.expander("Calculation reasoning & details"):
        st.caption(INVENTORY_ASSUMPTION)
        st.caption("Targets below retain calculation precision. A pre-arrival service gap may need earlier supply.")
        st.dataframe(readable_frame(inventory), width="stretch", hide_index=True)


def build_inventory_bar_chart(inventory: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Display existing metrics in the dashboard's dark theme and red accent."""
    data = inventory.dropna(subset=[metric]).sort_values("product")
    figure = go.Figure(go.Bar(
        x=data["product"], y=data[metric], marker_color="#ef4444",
        hovertemplate="%{x}<br>%{y:,.0f} units<extra></extra>",
    ))
    figure.update_layout(
        title=title, template="plotly_dark", height=330,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.55)",
        font=dict(color="#e2e8f0"), margin=dict(l=12, r=12, t=48, b=12),
        xaxis=dict(title=None, tickangle=-35, showgrid=False),
        yaxis=dict(title="Units", gridcolor="rgba(148,163,184,0.18)"), showlegend=False,
    )
    if data.empty:
        figure.add_annotation(text="Not available — see calculation details", x=.5, y=.5,
                              xref="paper", yref="paper", showarrow=False)
    return figure


def render_priority_actions(result: PipelineRunResult) -> None:
    st.markdown('<div class="section-title">Inventory Decisions</div>', unsafe_allow_html=True)
    render_inventory_decisions(normalize_output_schema(result.inventory_metrics))


def render_dashboard_tab(result: PipelineRunResult) -> None:
    """Render the business dashboard tab."""
    render_forecast_hero(result)
    st.write("")
    render_product_forecast_detail(result)
    st.write("")
    render_forecast_summary(result)
    render_kpis(result)
    st.write("")

    st.markdown('<div class="section-title">Planning Narrative</div>', unsafe_allow_html=True)
    for insight in build_executive_summary(result)[2:4]:
        st.markdown(f"- {insight}")

    st.write("")
    st.markdown('<div class="section-title">Alerts & Supporting Insights</div>', unsafe_allow_html=True)
    for alert in build_top_alerts(result):
        st.markdown(
            f'<div class="alert-box alert-{alert["tone"]}">{alert["message"]}</div>',
            unsafe_allow_html=True,
        )
    for column, (metric, title) in zip(st.columns(3), [
        ("planning_safety_stock", "Planning Safety Stock"),
        ("reorder_point", "Reorder Point"),
        ("order_up_to_target", "Target Inventory"),
    ]):
        with column:
            st.plotly_chart(
                build_inventory_bar_chart(result.inventory_metrics, metric, title),
                use_container_width=True, config={"responsive": True},
            )

    st.write("")
    render_priority_actions(result)

    st.write("")
    st.markdown('<div class="section-title">Detailed Narrative & Downloads</div>', unsafe_allow_html=True)
    with st.expander("Detailed Narrative Summary"):
        st.text(result.summary_text)
    with st.expander("Product-Level Forecast Charts"):
        render_images(result.chart_paths, columns=2)

    download_columns = st.columns(3)
    with download_columns[0]:
        render_download_button("Download forecast_output.csv", result.forecast_csv_path, "text/csv")
        render_download_button("Download planning_forecast.csv", result.planning_forecast_path, "text/csv")
    with download_columns[1]:
        render_download_button("Download inventory_metrics.csv", result.inventory_metrics_path, "text/csv")
    with download_columns[2]:
        render_download_button("Download forecast_summary.txt", result.summary_path, "text/plain")


def model_display_name(model: object, order: object = None) -> str:
    if pd.isna(model):
        return "—"
    return f"{model} {order}" if str(model) == "ARIMA" and pd.notna(order) else str(model)


def render_model_tab(result: PipelineRunResult) -> None:
    """Concise model comparison with technical evidence available on demand."""
    st.markdown('<div class="section-title">Forecast Models</div>', unsafe_allow_html=True)
    st.caption("The best model is automatically selected for each product using holdout forecast performance.")
    models = result.method_summary.sort_values("product").copy()
    for column, label, value in zip(st.columns(3),
            ["Products Modeled", "Average Holdout MAPE", "Candidate Models Evaluated"],
            [len(models), f"{models['mape'].mean():.2f}%", int(models['candidate_count'].sum())]):
        column.metric(label, value)
    models["Runner-Up"] = models.apply(
        lambda row: model_display_name(row["runner_up_model"], row["runner_up_arima_order"]), axis=1
    )
    for _, row in models.iterrows():
        st.markdown(
            f"**{row['product']} — {model_display_name(row['selected_method'], row['selected_arima_order'])}**  \n"
            f"Holdout MAPE: **{row['mape']:.2f}%** · Selected over: {row['Runner-Up']}"
        )
    columns = {
        "product": "Product", "selected_method": "Selected Model", "selected_arima_order": "ARIMA Order",
        "mae": "Holdout MAE", "mape": "Holdout MAPE", "Runner-Up": "Runner-Up",
    }
    st.dataframe(models[list(columns)].rename(columns=columns), width="stretch", hide_index=True,
                 column_config={"Holdout MAE": st.column_config.NumberColumn(format="%.2f"),
                                "Holdout MAPE": st.column_config.NumberColumn(format="%.2f%%")})
    with st.expander("Model-selection details"):
        st.caption("Candidates are ranked by holdout MAE, then MAPE, with AIC as the final tie-breaker. Candidate count is summed across products.")
        st.dataframe(readable_frame(result.method_summary), width="stretch", hide_index=True)
        st.dataframe(readable_frame(result.top_candidates), width="stretch", hide_index=True)


def render_inventory_tab(result: PipelineRunResult) -> None:
    """Render the inventory metrics tab with filters."""
    inventory_metrics = normalize_output_schema(result.inventory_metrics)
    st.markdown('<div class="section-title">Planning Outputs</div>', unsafe_allow_html=True)
    st.caption("Inventory recommendations based on forecast demand, current stock, product lead time, and safety stock.")
    filter_columns = st.columns(2)
    with filter_columns[0]:
        category_options = ["All"] + sorted(inventory_metrics["category"].dropna().astype(str).unique().tolist())
        selected_category = st.selectbox("Filter by category", category_options)
    with filter_columns[1]:
        risk_options = ["All"] + sorted(inventory_metrics["risk_level"].dropna().astype(str).unique().tolist())
        selected_risk = st.selectbox("Filter by demand volatility", risk_options)

    filtered = inventory_metrics
    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]
    if selected_risk != "All":
        filtered = filtered[filtered["risk_level"] == selected_risk]
    render_inventory_decisions(filtered)


def render_raw_tab(raw_data: pd.DataFrame, cleaned_preview: pd.DataFrame | None, preview_error: str | None) -> None:
    """Render raw and standardized data previews."""
    st.caption("Compare the uploaded dataset with the standardized data used by the forecasting pipeline.")
    st.markdown('<div class="section-title">Raw Uploaded Data</div>', unsafe_allow_html=True)
    st.dataframe(readable_frame(raw_data.head(50)), width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Standardized / Cleaned Preview</div>', unsafe_allow_html=True)
    if cleaned_preview is not None:
        st.dataframe(readable_frame(cleaned_preview.head(50)), width="stretch", hide_index=True)
    else:
        st.error(f"Could not standardize uploaded data: {preview_error}")


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="Demand Forecasting & Planning Tool", layout="wide")
    inject_app_styles()
    st.title("Demand Forecasting & Planning Tool")
    st.write("Upload demand data, generate a forecast-led planning view, and connect projected demand to the inventory actions that matter next.")

    with st.sidebar:
        st.header("Configuration")
        service_level = st.number_input("Service level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5)
        lead_time = st.number_input("Lead time (days)", min_value=1, max_value=30, value=2, step=1)
        forecast_horizon = st.number_input("Forecast horizon", min_value=1, max_value=30, value=14, step=1)

    uploaded_file = st.file_uploader("Upload demand CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to preview the data and run the forecast planning analysis.")
        return

    raw_data = pd.read_csv(uploaded_file)

    cleaned_preview: pd.DataFrame | None = None
    preview_error: str | None = None
    try:
        cleaned_preview = clean_and_preprocess_data(raw_data, default_lead_time_days=int(lead_time))
    except Exception as exc:  # noqa: BLE001
        preview_error = str(exc)

    if st.button("Run Analysis", type="primary", disabled=cleaned_preview is None):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(UPLOADED_INPUT_PATH, index=False)
        config = {
            "service_level": float(service_level),
            "lead_time": int(lead_time),
            "forecast_horizon": int(forecast_horizon),
        }
        with st.spinner("Running demand forecasting and planning analysis..."):
            try:
                st.session_state["pipeline_result"] = run_pipeline(
                    input_path=UPLOADED_INPUT_PATH,
                    output_dir=OUTPUT_DIR,
                    config=config,
                )
                st.session_state.pop("ai_assistant_messages", None)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")
                return

    result: PipelineRunResult | None = st.session_state.get("pipeline_result")
    if result is not None:
        st.caption(
            f"Results use saved run settings: {result.run_config['service_level']}% service level; "
            f"{result.run_config['lead_time']} calendar-day default lead time; "
            f"{result.run_config['forecast_horizon']} displayed periods. Rerun analysis to apply changes."
        )
    tab_labels = ["Forecast Overview", "Forecast Models", "Planning Outputs", "Raw / Cleaned Data"]
    if ENABLE_AI_ASSISTANT:
        tab_labels.append("AI Assistant")
    if "active_tab" not in st.session_state or st.session_state["active_tab"] not in tab_labels:
        st.session_state["active_tab"] = tab_labels[0]

    st.session_state["active_tab"] = st.segmented_control(
        "Navigation",
        options=tab_labels,
        default=st.session_state["active_tab"],
        key="active_tab_selector",
    )
    active_tab = st.session_state["active_tab"]

    if active_tab == "Forecast Overview":
        if result is None:
            st.info("Run the analysis to populate the dashboard.")
        else:
            render_dashboard_tab(result)
    elif active_tab == "Forecast Models":
        if result is None:
            st.info("Run the analysis to review model selection.")
        else:
            render_model_tab(result)
    elif active_tab == "Planning Outputs":
        if result is None:
            st.info("Run the analysis to inspect inventory metrics.")
        else:
            render_inventory_tab(result)
    elif active_tab == "Raw / Cleaned Data":
        render_raw_tab(raw_data, cleaned_preview, preview_error)
    elif ENABLE_AI_ASSISTANT and active_tab == "AI Assistant":
        render_ai_assistant(
            result=result,
            service_level=float(result.run_config["service_level"]) if result else float(service_level),
            lead_time=int(result.run_config["lead_time"]) if result else int(lead_time),
            forecast_horizon=int(result.run_config["forecast_horizon"]) if result else int(forecast_horizon),
            model_name=OLLAMA_MODEL_NAME,
        )


if __name__ == "__main__":
    main()
