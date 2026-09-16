"""Behavioral regressions for daily review and downstream planning outputs.

Run: MPLCONFIGDIR=/tmp/supply-chain-mpl .venv/bin/python -B -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing import clean_and_preprocess_data
from inventory_planning import (
    REDUCE_ACTION, calculate_core_inventory_metrics,
)
from demand_forecasting import calculate_accuracy_metrics, calculate_inventory_metrics, run_pipeline
from forecasting_models import select_best_forecast_for_product
from ai_assistant import (
    build_assistant_context, request_ollama_chat, _build_product_context_row,
    build_product_alias_map, resolve_product_mentions,
)
from app import (
    build_executive_summary, build_kpi_metrics, build_top_alerts,
    inventory_action_frame, readable_frame, build_inventory_bar_chart,
)


def history(stock=50, lead=10, product="Widget", demand=10, frequency="D", periods=16):
    raw = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=periods, freq=frequency),
        "product": product, "demand": demand, "on_hand_inventory": stock,
        "lead_time_days": lead,
    })
    return clean_and_preprocess_data(raw, default_lead_time_days=2)


def forecast(cleaned, demand=10, periods=20, frequency="D"):
    return pd.DataFrame({
        "product": cleaned["product"].iloc[0], "forecast": demand,
        "date": pd.date_range(cleaned["date"].max(), periods=periods + 1, freq=frequency)[1:],
    })


def plan(stock=50, lead=10):
    cleaned = history(stock=stock, lead=lead)
    return calculate_core_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]


class CorePlanningTests(unittest.TestCase):
    def test_previous_bad_case(self):
        row = plan()
        self.assertEqual(row.reorder_point, 100)
        self.assertEqual(row.order_up_to_target, 110)
        self.assertEqual(row.recommended_action, "Increase")
        self.assertEqual(row.recommended_quantity_adjustment, 60)
        self.assertEqual(row.pre_arrival_shortage, 50)
        self.assertEqual(row.planning_safety_stock, 0)

    def test_exact_target(self):
        row = plan(110)
        self.assertEqual(row.recommended_action, "Maintain")
        self.assertEqual(row.recommended_quantity_adjustment, 0)
        self.assertEqual(row.pre_arrival_shortage, 0)

    def test_excess_is_rundown_not_disposal(self):
        row = plan(150)
        self.assertEqual(row.recommended_action, REDUCE_ACTION)
        self.assertEqual(row.recommended_quantity_adjustment, 40)
        self.assertEqual(row.excess_quantity, 40)
        self.assertIn("not disposal", row.reasoning)

    def test_confirmed_zero(self):
        row = plan(0)
        self.assertEqual(row.inventory_status, "known")
        self.assertEqual(row.recommended_quantity_adjustment, 110)
        self.assertEqual(row.pre_arrival_shortage, 100)

    def test_unknown_inventory_preserves_targets_but_withholds_actions(self):
        for value in [None, "", "not a number"]:
            with self.subTest(value=value):
                row = plan(value)
                self.assertEqual(row.inventory_status, "unknown")
                self.assertTrue(pd.isna(row.recommended_action))
                self.assertTrue(pd.isna(row.recommended_quantity_adjustment))
                self.assertTrue(pd.isna(row.pre_arrival_shortage))
                self.assertEqual(row.reorder_point, 100)
                self.assertEqual(row.order_up_to_target, 110)

    def test_absent_inventory_column_and_missing_latest_snapshot(self):
        for raw in [history().drop(columns="on_hand_inventory"), history()]:
            if "on_hand_inventory" in raw:
                raw.loc[raw.index[-1], "on_hand_inventory"] = float("nan")
            cleaned = clean_and_preprocess_data(raw, 2)
            row = calculate_core_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]
            self.assertEqual(row.inventory_status, "unknown")
            self.assertTrue(pd.isna(row.recommended_action))

    def test_invalid_inventory(self):
        for value in [-1, float("inf")]:
            row = plan(value)
            self.assertEqual(row.inventory_status, "invalid")
            self.assertTrue(pd.isna(row.recommended_quantity_adjustment))
            self.assertEqual(row.order_up_to_target, 110)

    def test_different_product_lead_times(self):
        a, b = history(50, 2, "A"), history(50, 10, "B")
        rows = calculate_core_inventory_metrics(pd.concat([a, b]), pd.concat([forecast(a), forecast(b)]), .95, 2).set_index("product")
        self.assertEqual(rows.loc["A", "order_up_to_target"], 30)
        self.assertEqual(rows.loc["A", "excess_quantity"], 20)
        self.assertEqual(rows.loc["B", "order_up_to_target"], 110)
        self.assertEqual(rows.loc["B", "recommended_quantity_adjustment"], 60)

    def test_fractional_target_rounds_quantity_not_target(self):
        cleaned = history(110)
        future = forecast(cleaned, 10.05)
        row = calculate_core_inventory_metrics(cleaned, future, .95, 2).iloc[0]
        self.assertAlmostEqual(row.order_up_to_target, 110.55)
        self.assertEqual(row.recommended_quantity_adjustment, 0)
        self.assertEqual(row.recommended_action, "Maintain")
        cleaned["on_hand_inventory"] = 111
        row = calculate_core_inventory_metrics(cleaned, future, .95, 2).iloc[0]
        self.assertEqual(row.recommended_action, "Maintain")
        self.assertEqual(row.excess_quantity, 0)

    def test_weekly_conversion_and_variability(self):
        cleaned = history(stock=30, lead=3, demand=[63, 77] * 8, frequency="W-SUN")
        row = calculate_core_inventory_metrics(cleaned, forecast(cleaned, 70, 2, "W-SUN"), .95, 2).iloc[0]
        self.assertEqual(row.planning_status, "ready")
        self.assertAlmostEqual(row.daily_demand_std, cleaned.demand.std() / 7 ** .5)
        self.assertEqual(row.expected_demand_lead_time, 30)
        self.assertEqual(row.expected_demand_protection_period, 40)
        self.assertEqual(row.recommended_quantity_adjustment, 19)

    def test_monthly_actual_calendar_lengths(self):
        for freq in ["MS", "ME"]:
            cleaned = history(stock=30, lead=3, demand=310, frequency=freq, periods=12)
            # A constant daily rate despite varying monthly totals implies zero variance.
            cleaned["demand"] = cleaned.date.dt.days_in_month * 10
            future = forecast(cleaned, 310, 2, freq)  # January and February 2027
            future["forecast"] = future.date.dt.days_in_month * 10
            row = calculate_core_inventory_metrics(cleaned, future, .95, 2).iloc[0]
            self.assertEqual(row.planning_status, "ready")
            self.assertEqual(row.expected_demand_protection_period, 40)
            self.assertEqual(row.daily_demand_std, 0)
            self.assertEqual(row.recommended_quantity_adjustment, 10)

    def test_unsupported_calendars_withhold_quantities(self):
        for cleaned in [history(frequency="B"), history().drop(index=4), pd.concat([history(), history().iloc[[0]]])]:
            row = calculate_core_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]
            self.assertEqual(row.planning_status, "unavailable")
            self.assertTrue(pd.isna(row.recommended_quantity_adjustment))
            self.assertTrue(row.planning_issue)

    def test_missing_coverage_and_negative_forecasts(self):
        cleaned = history()
        for future in [forecast(cleaned, periods=3), forecast(cleaned, demand=-1)]:
            row = calculate_core_inventory_metrics(cleaned, future, .95, 2).iloc[0]
            self.assertEqual(row.planning_status, "unavailable")
            self.assertTrue(pd.isna(row.recommended_action))

    def test_fractional_lead_time_and_source(self):
        row = plan(0, 2.5)
        self.assertEqual(row.expected_demand_lead_time, 25)
        self.assertEqual(row.order_up_to_target, 35)
        self.assertEqual(row.lead_time_days, 2.5)
        self.assertEqual(row.lead_time_source, "product data")

    def test_nonzero_daily_safety_stock_formulas(self):
        cleaned = history(50, demand=[5, 15] * 8)
        row = calculate_core_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]
        sigma = cleaned.demand.std()
        self.assertAlmostEqual(row.lead_time_safety_stock, row.z_score * sigma * 10 ** .5)
        self.assertEqual(row.safety_stock, row.lead_time_safety_stock)
        self.assertAlmostEqual(row.planning_safety_stock, row.z_score * sigma * 11 ** .5)
        self.assertAlmostEqual(row.reorder_point, 100 + row.lead_time_safety_stock)
        self.assertAlmostEqual(row.order_up_to_target, 110 + row.planning_safety_stock)

    def test_forecast_shape_and_display_length_do_not_replace_protection_window(self):
        cleaned = history(0, lead=2)
        future = forecast(cleaned)
        future.loc[:2, "forecast"] = [10, 20, 30]
        future.loc[3:, "forecast"] = 1000
        row = calculate_core_inventory_metrics(cleaned, future, .95, 2).iloc[0]
        self.assertEqual(row.expected_demand_lead_time, 30)
        self.assertEqual(row.order_up_to_target, 60)
        truncated = calculate_core_inventory_metrics(cleaned, future.head(3), .95, 2).iloc[0]
        self.assertEqual(row.recommended_quantity_adjustment, truncated.recommended_quantity_adjustment)

    def test_invalid_lead_time_does_not_crash_legacy_diagnostics(self):
        for lead in [-1, float("inf")]:
            cleaned = history(50, lead)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = calculate_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]
            self.assertEqual(row.planning_status, "unavailable")
            self.assertTrue(pd.isna(row.recommended_quantity_adjustment))

    def test_default_lead_time_is_explicit(self):
        cleaned = history(50, None)
        row = calculate_core_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]
        self.assertEqual(row.lead_time_days, 2)
        self.assertEqual(row.lead_time_source, "configured default")
        self.assertEqual(row.order_up_to_target, 30)

    def test_expired_stock_is_not_usable(self):
        cleaned = history(100)
        cleaned["days_to_expire"] = -1
        row = calculate_core_inventory_metrics(cleaned, forecast(cleaned), .95, 2).iloc[0]
        self.assertEqual(row.usable_on_hand_inventory, 0)
        self.assertEqual(row.recommended_quantity_adjustment, 110)

    def test_legacy_spoilage_does_not_override_core(self):
        cleaned = history(150)
        cleaned["days_to_expire"] = 1
        cleaned["perishable_flag"] = True
        metrics = calculate_inventory_metrics(cleaned, forecast(cleaned), .95, 2)
        self.assertEqual(metrics.iloc[0].recommended_action, REDUCE_ACTION)
        self.assertEqual(metrics.iloc[0].recommended_quantity_adjustment, 40)
        self.assertIn("not used", metrics.iloc[0].legacy_perishability_note)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cleaned = history()
        cls.input = cls.root / "input.csv"
        cleaned.to_csv(cls.input, index=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.result = run_pipeline(cls.input, cls.root / "output", {"service_level": 95, "lead_time": 2, "forecast_horizon": 3})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_coverage_and_exports(self):
        result = self.result
        self.assertEqual(len(result.future_forecast), 3)
        self.assertEqual(len(result.planning_forecast), 11)
        row = result.inventory_metrics.iloc[0]
        self.assertEqual(row.order_up_to_target, 110)
        self.assertEqual(row.recommended_quantity_adjustment, 60)
        saved = pd.read_csv(result.inventory_metrics_path)
        self.assertEqual(saved.iloc[0].pre_arrival_shortage, 50)
        self.assertTrue(result.planning_forecast_path.exists())
        self.assertIn("backorders are not modeled", result.summary_text)

    def test_output_extension_does_not_change_selection(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            short = select_best_forecast_for_product(history(), horizon=3)
            extended = select_best_forecast_for_product(history(), horizon=3, output_horizon=11)
        self.assertEqual(short.model_name, extended.model_name)
        self.assertEqual(short.selected_arima_order, extended.selected_arima_order)
        self.assertEqual(short.validation_points, extended.validation_points)
        self.assertEqual(short.mae, extended.mae)
        pd.testing.assert_frame_equal(short.top_candidates, extended.top_candidates)
        pd.testing.assert_frame_equal(short.future, extended.future.head(3))

    def test_ai_uses_saved_settings_and_core_values(self):
        context = build_assistant_context(self.result, "Widget", 99, 30, 30)
        self.assertEqual(context["overall_summary"]["service_level_pct"], 95)
        self.assertEqual(context["overall_summary"]["forecast_horizon_periods"], 3)
        product = context["selected_product_summary"]
        self.assertEqual(product["lead_time_days"], 10)
        self.assertEqual(product["order_up_to_target"], 110)
        self.assertEqual(product["pre_arrival_shortage"], 50)
        self.assertEqual(product["recommended_quantity_adjustment"], 60)
        self.assertEqual(product["product_mape"], self.result.method_summary.iloc[0].mape)
        self.assertIn("holdout", context["overall_summary"]["accuracy_basis"])
        captured = {}
        def send(req, timeout):
            captured.update(json.loads(req.data))
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = b'{"message":{"content":"ok"}}'
            return response
        with patch("ai_assistant.request.urlopen", side_effect=send):
            request_ollama_chat("http://example.invalid", "test", context, [])
        self.assertIn("never disposal", captured["messages"][0]["content"])
        self.assertNotIn("primary driver", captured["messages"][0]["content"])
        self.assertIn("directly from order_up_to_target", captured["messages"][0]["content"])
        self.assertIn("SAME excess", captured["messages"][0]["content"])
        self.assertIn("do not invent a monthly optimal target", captured["messages"][0]["content"])
        instructions = captured["messages"][0]["content"]
        self.assertIn("reorder_point = expected_demand_lead_time + lead_time_safety_stock", instructions)
        self.assertIn("each named product's own record in products_by_name", instructions)
        self.assertIn("Quote each product's supplied reorder point and its two supplied components", instructions)
        self.assertIn("If the necessary inputs are missing", instructions)
        self.assertIn("Do not speculate about product popularity, outside market demand, hypothetical lead times, hypothetical volatility", instructions)

    def test_product_comparison_context_is_aligned_by_name(self):
        yogurt = history(200, 5, "Greek Yogurt", demand=[100, 150] * 8)
        milk = history(500, 2, "Whole Milk", demand=[200, 202] * 8)
        cleaned = pd.concat([yogurt, milk], ignore_index=True)
        future = pd.concat([forecast(milk, 201), forecast(yogurt, 125)], ignore_index=True)
        inventory = calculate_inventory_metrics(cleaned, future, .95, 2)
        metrics = pd.concat([
            self.result.metrics.loc[self.result.metrics["product"] == "Overall"],
            pd.DataFrame([
                {"product": "Whole Milk", "mape": 2.25, "selected_method": "ARIMA"},
                {"product": "Greek Yogurt", "mape": 7.75, "selected_method": "Holt Trend"},
            ]),
        ], ignore_index=True)
        # Independently shuffled frames and arbitrary indexes must not change product alignment.
        result = SimpleNamespace(**vars(self.result))
        result.cleaned_data, result.future_forecast, result.metrics = cleaned, future, metrics
        result.method_summary = metrics[metrics["product"] != "Overall"].copy()
        result.inventory_metrics = inventory.iloc[::-1].set_axis([91, 14])
        context = build_assistant_context(result, "Whole Milk", 95, 2, 3)
        columns = [
            "daily_average_demand", "daily_demand_std", "lead_time_days",
            "expected_demand_lead_time", "lead_time_safety_stock", "reorder_point",
            "protection_period_days", "order_up_to_target", "usable_on_hand_inventory",
            "recommended_action", "recommended_quantity_adjustment", "pre_arrival_shortage",
        ]
        for name in ["Greek Yogurt", "Whole Milk"]:
            actual = context["products_by_name"][name]
            expected = inventory.set_index("product").loc[name]
            self.assertEqual(actual["product"], name)
            for column in columns:
                self.assertEqual(actual[column], expected[column], f"{name}: {column}")
            self.assertEqual(actual["product_mape"], metrics.set_index("product").loc[name, "mape"])
        self.assertEqual(context["selected_product_summary"], context["products_by_name"]["Whole Milk"])
        names = {name for _, name in resolve_product_mentions(
            "Why is Greek Yogurt's reorder point higher than Whole Milk's?", context["product_aliases"]
        )}
        self.assertEqual(names, {"Greek Yogurt", "Whole Milk"})

    def test_duplicate_product_records_are_rejected(self):
        result = SimpleNamespace(**vars(self.result))
        result.inventory_metrics = pd.concat([result.inventory_metrics, result.inventory_metrics])
        with self.assertRaises(ValueError):
            build_assistant_context(result, "Widget", 95, 2, 3)

    def test_ambiguous_alias_does_not_choose_a_product(self):
        aliases = build_product_alias_map(["Whole Milk", "Skim Milk", "Greek Yogurt"])
        self.assertNotIn("milk", aliases)
        self.assertEqual(aliases["whole milk"], "Whole Milk")
        self.assertEqual(aliases["skim milk"], "Skim Milk")

    def test_plotly_calls_use_supported_parameters_and_config(self):
        tree = ast.parse((ROOT / "app.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "plotly_chart"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertEqual(set(keywords), {"use_container_width", "config"})
            self.assertEqual(ast.literal_eval(keywords["config"]), {"responsive": True})
            self.assertIs(ast.literal_eval(keywords["use_container_width"]), True)

    def test_sour_cream_ai_context_has_one_adjustment_and_direct_target(self):
        inventory = pd.Series({
            "on_hand_inventory": 1800, "inventory_position": 1800,
            "usable_on_hand_inventory": 1800, "order_up_to_target": 1087.4,
            "recommended_quantity_adjustment": 712, "excess_quantity": 712,
            "recommended_action": REDUCE_ACTION, "lead_time_days": 2,
            "protection_period_days": 3,
        })
        context = _build_product_context_row("Sour Cream", pd.Series(dtype=object), inventory, pd.DataFrame(), 2)
        self.assertEqual(context["order_up_to_target"], 1087.4)
        self.assertEqual(context["recommended_quantity_adjustment"], 712)
        self.assertNotIn("excess_quantity", context)
        self.assertIn("Single excess amount", context["adjustment_meaning"])

    def test_inventory_presentation_preserves_calculations(self):
        inventory = self.result.inventory_metrics.copy()
        inventory.loc[0, "order_up_to_target"] = 110.49
        before = inventory.copy(deep=True)
        display = inventory_action_frame(inventory)
        self.assertEqual(list(display.columns), [
            "Product", "On Hand", "Lead Time", "Reorder Point", "Target Inventory", "Action", "Recommended Qty",
        ])
        self.assertEqual(display.iloc[0]["Target Inventory"], 110)
        pd.testing.assert_frame_equal(inventory, before)
        renamed = readable_frame(inventory)
        self.assertIn("On-Hand Inventory", renamed)
        self.assertIn("Lead Time (Days)", renamed)
        self.assertEqual(readable_frame(pd.DataFrame({"promo_flag": [True]})).columns[0], "Promotion")
        for metric in ["planning_safety_stock", "reorder_point", "order_up_to_target"]:
            figure = build_inventory_bar_chart(inventory, metric, "Test")
            self.assertEqual(figure.data[0].type, "bar")
            self.assertEqual(figure.data[0].marker.color, "#ef4444")
            self.assertEqual(figure.data[0].y[0], inventory.iloc[0][metric])

    def test_accuracy_uses_holdout_not_historical_fit(self):
        historical = self.result.actual_vs_forecast.copy()
        historical["forecast"] = historical["demand"]  # Perfect fit must not replace holdout scores.
        methods = self.result.method_summary.copy()
        methods["mae"], methods["mape"] = 12.5, 25.0
        metrics = calculate_accuracy_metrics(historical, methods)
        self.assertTrue(metrics["mape"].eq(25).all())
        self.assertTrue(metrics["mae"].eq(12.5).all())

    def test_unknown_inventory_survives_csv_ui_ai(self):
        cleaned = history(None)
        metrics = calculate_inventory_metrics(cleaned, forecast(cleaned), .95, 2)
        result = SimpleNamespace(**vars(self.result))
        result.inventory_metrics = metrics
        for builder in [build_executive_summary, build_kpi_metrics, build_top_alerts]:
            self.assertTrue(builder(result))
        context = build_assistant_context(result, "Widget", 95, 2, 3)
        self.assertIsNone(context["selected_product_summary"]["recommended_quantity_adjustment"])
        self.assertIsNone(context["selected_product_summary"]["recommended_action"])
        self.assertIsNone(context["selected_product_summary"]["inventory_position"])
        path = self.root / "unknown.csv"
        metrics.to_csv(path, index=False)
        self.assertTrue(pd.isna(pd.read_csv(path).iloc[0].recommended_quantity_adjustment))

    def test_non_daily_pipeline(self):
        for frequency, demand in [("W-SUN", 70), ("MS", 300), ("ME", 300)]:
            with self.subTest(frequency=frequency), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = history(30, 3, demand=demand, frequency=frequency)
                path = self.root / f"{frequency}.csv"
                raw.to_csv(path, index=False)
                result = run_pipeline(path, self.root / frequency, {"service_level": 95, "lead_time": 2, "forecast_horizon": 1})
                self.assertEqual(result.inventory_metrics.iloc[0].planning_status, "ready")
                self.assertEqual(len(result.future_forecast), 1)
                self.assertLess(result.inventory_metrics.iloc[0].expected_demand_protection_period, demand)

    def test_streamlit_views_render_known_and_unknown_quantities(self):
        from streamlit.testing.v1 import AppTest
        from streamlit.elements import plotly_chart as plotly_element
        script = '''
import streamlit as st
from app import render_dashboard_tab, render_inventory_tab, render_model_tab, render_raw_tab
result = st.session_state["result"]
render_dashboard_tab(result)
render_inventory_tab(result)
render_model_tab(result)
render_raw_tab(result.cleaned_data, result.cleaned_data, None)
'''
        for stock in [50, None]:
            with self.subTest(stock=stock):
                result = SimpleNamespace(**vars(self.result))
                cleaned = history(stock)
                result.inventory_metrics = calculate_inventory_metrics(cleaned, forecast(cleaned), .95, 2)
                app = AppTest.from_string(script)
                app.session_state["result"] = result
                with patch("streamlit.elements.plotly_chart.show_deprecation_warning",
                           wraps=plotly_element.show_deprecation_warning) as plotly_warning:
                    app.run(timeout=20)
                    plotly_warning.assert_not_called()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertIn("Filter by demand volatility", [widget.label for widget in app.selectbox])
                self.assertFalse(any("because Selected because" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
