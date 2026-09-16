"""Forecasting model selection helpers built on exponential smoothing methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ProductForecastResult:
    """Container for the winning forecast model output for one product."""

    historical: pd.DataFrame
    future: pd.DataFrame
    model_name: str
    mae: float
    mape: float
    trend_detected: bool
    seasonality_detected: bool
    selection_reason: str
    validation_points: int
    candidate_count: int
    runner_up_model: str | None
    selected_arima_order: str | None
    runner_up_arima_order: str | None
    top_candidates: pd.DataFrame


@dataclass(frozen=True)
class ModelCandidateSpec:
    """Definition for one model candidate in the selection pipeline."""

    model_name: str
    fit_function: Any
    metadata: dict[str, Any]


def calculate_error_metrics(actual: pd.Series, forecast: pd.Series) -> tuple[float, float]:
    """Return MAE and MAPE for an in-sample forecast."""
    absolute_error = (actual - forecast).abs()
    mae = float(absolute_error.mean())

    non_zero_actual = actual.replace(0, pd.NA)
    percentage_error = (absolute_error / non_zero_actual) * 100
    mape = float(percentage_error.mean(skipna=True) if not percentage_error.dropna().empty else 0.0)
    return mae, mape


def detect_trend(demand_series: pd.Series) -> bool:
    """Use the overall slope to detect a simple upward or downward trend."""
    x_values = np.arange(len(demand_series))
    slope = np.polyfit(x_values, demand_series.to_numpy(dtype=float), 1)[0]
    threshold = max(float(demand_series.std(ddof=0)) * 0.05, 0.1)
    return abs(float(slope)) > threshold


def detect_seasonality(demand_series: pd.Series, seasonal_periods: int = 4) -> bool:
    """Use lag correlation as a lightweight seasonality signal."""
    if len(demand_series) < seasonal_periods * 2:
        return False
    autocorrelation = demand_series.autocorr(lag=seasonal_periods)
    return bool(pd.notna(autocorrelation) and autocorrelation > 0.3)


def infer_step(product_history: pd.DataFrame) -> str:
    """Infer date frequency and fall back to daily."""
    inferred_frequency = pd.infer_freq(product_history["date"])
    return inferred_frequency or "D"


def build_reason(model_name: str, mae: float, mape: float, trend_detected: bool, seasonality_detected: bool) -> str:
    """Explain in plain English why a model was selected."""
    drivers: list[str] = [f"it produced the lowest error score with MAE {mae:.2f} and MAPE {mape:.2f}%"]
    if model_name == "Holt-Winters Seasonal":
        drivers.append("the series had enough history to support seasonal smoothing")
    if model_name == "ARIMA":
        drivers.append("autoregressive structure improved the holdout forecast")
    if trend_detected and model_name in {"Holt Trend", "Holt-Winters Seasonal"}:
        drivers.append("a trend signal was detected")
    if seasonality_detected and model_name == "Holt-Winters Seasonal":
        drivers.append("a seasonal pattern was detected")
    if model_name == "Simple Exponential Smoothing":
        drivers.append("the demand pattern looked stable enough for a level-only model")
    return "Selected because " + ", ".join(drivers) + "."


def build_selection_reason(
    model_name: str,
    mae: float,
    mape: float,
    aic: float,
    trend_detected: bool,
    seasonality_detected: bool,
    validation_points: int,
    candidate_count: int,
    runner_up_model: str | None,
    selected_arima_order: str | None,
    runner_up_arima_order: str | None,
) -> str:
    """Explain the model selection using the holdout comparison context."""
    base_reason = build_reason(model_name, mae, mape, trend_detected, seasonality_detected).rstrip(".")
    comparison_parts = [f"based on a {validation_points}-period holdout comparison across {candidate_count} candidates"]
    if model_name == "ARIMA" and selected_arima_order:
        comparison_parts.append(f"ARIMA order {selected_arima_order} delivered the best holdout accuracy")
    if runner_up_model:
        runner_up_label = runner_up_model
        if runner_up_model == "ARIMA" and runner_up_arima_order:
            runner_up_label = f"{runner_up_model} {runner_up_arima_order}"
        comparison_parts.append(f"it outperformed {runner_up_label} as the closest alternative")
    if np.isfinite(aic):
        comparison_parts.append(f"its AIC was {aic:.2f} after tie-breaking")
    return f"{base_reason}, {' and '.join(comparison_parts)}."


def _fit_simple_exponential_smoothing(demand_series: pd.Series) -> tuple[pd.Series, Any]:
    fitted_model = SimpleExpSmoothing(demand_series, initialization_method="estimated").fit(optimized=True)
    fitted_values = fitted_model.fittedvalues.astype(float)
    return fitted_values, fitted_model


def _fit_holt_trend(demand_series: pd.Series) -> tuple[pd.Series, Any]:
    fitted_model = Holt(demand_series, initialization_method="estimated").fit(optimized=True)
    fitted_values = fitted_model.fittedvalues.astype(float)
    return fitted_values, fitted_model


def _fit_holt_winters(demand_series: pd.Series, seasonal_periods: int = 4) -> tuple[pd.Series, Any]:
    fitted_model = ExponentialSmoothing(
        demand_series,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)
    fitted_values = fitted_model.fittedvalues.astype(float)
    return fitted_values, fitted_model


def _fit_arima(demand_series: pd.Series, order: tuple[int, int, int]) -> tuple[pd.Series, Any]:
    trend = "n" if order[1] > 0 else "c"
    fitted_model = ARIMA(demand_series, order=order, trend=trend).fit()
    fitted_values = fitted_model.predict(start=demand_series.index[0], end=demand_series.index[-1], typ="levels")
    return fitted_values.astype(float), fitted_model


def determine_validation_window(series_length: int, horizon: int, seasonal_periods: int) -> int:
    """Choose a reasonable holdout size while keeping enough training history."""
    minimum_train_size = max(seasonal_periods * 2, 6)
    max_validation = max(series_length - minimum_train_size, 1)
    proposed_window = max(horizon, min(seasonal_periods, 3), int(round(series_length * 0.25)))
    return max(1, min(proposed_window, max_validation))


def _score_model_on_holdout(
    demand_series: pd.Series,
    fit_function: Any,
    validation_points: int,
) -> tuple[pd.Series, float, float, float]:
    """Train on the earlier history and score forecasts on the holdout tail."""
    training_series = demand_series.iloc[:-validation_points]
    holdout_series = demand_series.iloc[-validation_points:]
    _, trained_model = fit_function(training_series)
    holdout_forecast = trained_model.forecast(validation_points)
    holdout_forecast = pd.Series(np.asarray(holdout_forecast, dtype=float), index=holdout_series.index)
    mae, mape = calculate_error_metrics(holdout_series, holdout_forecast)
    aic = float(getattr(trained_model, "aic", np.inf))
    return holdout_forecast, mae, mape, aic


def format_arima_order(order: tuple[int, int, int] | None) -> str | None:
    """Render an ARIMA order for CSV and summary outputs."""
    if order is None:
        return None
    return f"({order[0]},{order[1]},{order[2]})"


def build_candidate_specs(demand_series: pd.Series, seasonal_periods: int) -> list[ModelCandidateSpec]:
    """Create model candidates in a way that can be extended to SARIMA or Prophet later."""
    candidate_specs: list[ModelCandidateSpec] = [
        ModelCandidateSpec(
            model_name="Simple Exponential Smoothing",
            fit_function=_fit_simple_exponential_smoothing,
            metadata={},
        ),
        ModelCandidateSpec(
            model_name="Holt Trend",
            fit_function=_fit_holt_trend,
            metadata={},
        ),
    ]
    if len(demand_series) >= seasonal_periods * 2:
        candidate_specs.append(
            ModelCandidateSpec(
                model_name="Holt-Winters Seasonal",
                fit_function=lambda series: _fit_holt_winters(series, seasonal_periods),
                metadata={},
            )
        )
    if len(demand_series) >= 6:
        for p_value in range(3):
            for d_value in range(2):
                for q_value in range(3):
                    order = (p_value, d_value, q_value)
                    candidate_specs.append(
                        ModelCandidateSpec(
                            model_name="ARIMA",
                            fit_function=lambda series, order=order: _fit_arima(series, order),
                            metadata={"arima_order": order},
                        )
                    )
    return candidate_specs


def select_best_forecast_for_product(
    product_history: pd.DataFrame,
    horizon: int,
    seasonal_periods: int = 4,
    output_horizon: int | None = None,
) -> ProductForecastResult:
    """Fit supported models, score them, and return the best one for a product."""
    product_history = product_history.sort_values("date").copy()
    step = infer_step(product_history)
    # Month-start/end offsets need a monthly PeriodIndex; scoring is unchanged.
    offset = pd.tseries.frequencies.to_offset(step)
    period_step = "M" if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)) else step
    date_index = pd.DatetimeIndex(product_history["date"]).to_period(period_step)
    demand_series = pd.Series(product_history["demand"].to_numpy(dtype=float), index=date_index)
    product_name = str(product_history["product"].iloc[0])
    trend_detected = detect_trend(demand_series)
    seasonality_detected = detect_seasonality(demand_series, seasonal_periods=seasonal_periods)
    validation_points = determine_validation_window(len(demand_series), horizon, seasonal_periods)

    candidate_specs = build_candidate_specs(demand_series, seasonal_periods)

    candidates: list[dict[str, Any]] = []
    for candidate_spec in candidate_specs:
        try:
            _, validation_mae, validation_mape, validation_aic = _score_model_on_holdout(
                demand_series,
                candidate_spec.fit_function,
                validation_points,
            )
            candidates.append(
                {
                    "model_name": candidate_spec.model_name,
                    "fit_function": candidate_spec.fit_function,
                    "mae": validation_mae,
                    "mape": validation_mape,
                    "aic": validation_aic,
                    "arima_order": format_arima_order(candidate_spec.metadata.get("arima_order")),
                }
            )
        except Exception:
            continue

    if not candidates:
        raise ValueError(f"No supported model could be fit for product {product_name}")

    ranked_candidates = sorted(candidates, key=lambda item: (item["mae"], item["mape"], item["aic"]))
    # Diagnostics describe successful holdout evaluations. Losing candidates are
    # not fitted on full history merely to establish diagnostic eligibility.
    for selected_index, best_candidate in enumerate(ranked_candidates):
        try:
            fitted_values, fitted_model = best_candidate["fit_function"](demand_series)
        except Exception:
            continue
        best_candidate["fitted_values"] = fitted_values
        best_candidate["model"] = fitted_model
        break
    else:
        raise ValueError(f"No supported model could be fit for product {product_name}")

    # The next holdout-ranked alternative follows the successful winner; earlier
    # candidates that failed full fitting remain visible in the diagnostic table.
    runner_up = ranked_candidates[selected_index + 1] if selected_index + 1 < len(ranked_candidates) else None
    top_candidates = pd.DataFrame(ranked_candidates[:3]).copy()
    top_candidates.insert(0, "rank", range(1, len(top_candidates) + 1))
    top_candidates = top_candidates.rename(columns={"model_name": "model_family", "arima_order": "arima_order"})
    top_candidates = top_candidates[
        ["rank", "model_family", "arima_order", "mae", "mape", "aic"]
    ].copy()
    top_candidates.insert(0, "product", product_name)
    selection_reason = build_selection_reason(
        best_candidate["model_name"],
        best_candidate["mae"],
        best_candidate["mape"],
        best_candidate["aic"],
        trend_detected,
        seasonality_detected,
        validation_points,
        len(ranked_candidates),
        runner_up["model_name"] if runner_up else None,
        best_candidate["arima_order"],
        runner_up["arima_order"] if runner_up else None,
    )
    historical = product_history.copy()
    historical["forecast"] = best_candidate["fitted_values"].to_numpy(dtype=float)
    historical["type"] = "historical"
    historical["selected_method"] = best_candidate["model_name"]
    historical["selected_arima_order"] = best_candidate["arima_order"]
    historical["trend_detected"] = trend_detected
    historical["seasonality_detected"] = seasonality_detected
    historical["selection_reason"] = selection_reason

    # Extra planning coverage never changes the selection/validation horizon above.
    forecast_steps = max(horizon, output_horizon) if output_horizon is not None else horizon
    future_dates = pd.date_range(start=product_history["date"].max(), periods=forecast_steps + 1, freq=step)[1:]
    future_forecast = best_candidate["model"].forecast(forecast_steps) if forecast_steps > 0 else pd.Series(dtype=float)
    future = pd.DataFrame(
        {
            "date": future_dates,
            "product": product_name,
            "demand": pd.Series([float("nan")] * forecast_steps, dtype="float64"),
            "forecast": pd.Series(np.asarray(future_forecast, dtype=float)).round(2),
            "type": pd.Series(["future"] * forecast_steps, dtype="string"),
            "selected_method": pd.Series([best_candidate["model_name"]] * forecast_steps, dtype="string"),
            "selected_arima_order": pd.Series([best_candidate["arima_order"]] * forecast_steps, dtype="string"),
            "trend_detected": pd.Series([trend_detected] * forecast_steps, dtype="bool"),
            "seasonality_detected": pd.Series([seasonality_detected] * forecast_steps, dtype="bool"),
            "selection_reason": pd.Series([selection_reason] * forecast_steps, dtype="string"),
        }
    )

    return ProductForecastResult(
        historical=historical[
            [
                "date",
                "product",
                "demand",
                "forecast",
                "type",
                "selected_method",
                "selected_arima_order",
                "trend_detected",
                "seasonality_detected",
                "selection_reason",
            ]
        ],
        future=future,
        model_name=best_candidate["model_name"],
        mae=float(best_candidate["mae"]),
        mape=float(best_candidate["mape"]),
        trend_detected=trend_detected,
        seasonality_detected=seasonality_detected,
        selection_reason=selection_reason,
        validation_points=validation_points,
        candidate_count=len(ranked_candidates),
        runner_up_model=runner_up["model_name"] if runner_up else None,
        selected_arima_order=best_candidate["arima_order"],
        runner_up_arima_order=runner_up["arima_order"] if runner_up else None,
        top_candidates=top_candidates,
    )
