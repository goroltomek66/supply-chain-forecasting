"""Deferred full-history fitting: output parity, ranking, and fit budgets."""
from contextlib import ExitStack
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecasting_models as models
import demand_forecasting as pipeline
from data_preprocessing import clean_and_preprocess_data


def eager_reference(history, **kwargs):
    """Reproduce the former eager fit lifecycle, including full-fit eligibility.

    Every successful holdout evaluation immediately fits full history. Cache that
    result so the deferred selector consumes the same fit without fitting twice.
    Ranking is separately verified against explicit expectations below.
    """
    original_score = models._score_model_on_holdout
    original_build = models.build_candidate_specs
    cache = {}

    def build(series, seasonal_periods):
        specs = []
        for index, spec in enumerate(original_build(series, seasonal_periods)):
            def fit(values, fn=spec.fit_function, key=index):
                if len(values) == len(history) and key in cache:
                    return cache[key]
                return fn(values)
            specs.append(models.ModelCandidateSpec(spec.model_name, fit, spec.metadata))
        return specs

    def score(series, fn, points):
        result = original_score(series, fn, points)
        # Each wrapped callable's default key uniquely identifies its candidate.
        cache[fn.__defaults__[1]] = fn(series)
        return result

    with patch.object(models, "build_candidate_specs", side_effect=build), patch.object(
        models, "_score_model_on_holdout", side_effect=score
    ):
        return models.select_best_forecast_for_product(history, **kwargs)


class OutputParityTests(unittest.TestCase):
    def assert_result_equal(self, actual, expected):
        for field, value in vars(expected).items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(getattr(actual, field), value, check_exact=True)
            else:
                self.assertEqual(getattr(actual, field), value, field)

    def test_demo_pipeline_exact_parity_and_fit_budget(self):
        counts = Counter()
        fit_names = ["_fit_arima", "_fit_simple_exponential_smoothing", "_fit_holt_trend", "_fit_holt_winters"]
        with warnings.catch_warnings(), tempfile.TemporaryDirectory() as directory:
            warnings.simplefilter("ignore")
            config = {"service_level": 95, "lead_time": 2, "forecast_horizon": 14}
            with patch.object(pipeline, "select_best_forecast_for_product", side_effect=eager_reference):
                expected = pipeline.run_pipeline(ROOT / "data/demo_dairy_demand.csv", Path(directory) / "old", config)
            with ExitStack() as stack:
                for name in fit_names:
                    original = getattr(models, name)
                    def counted(series, *args, fn=original, family=name, **kwargs):
                        counts[(family, len(series))] += 1
                        return fn(series, *args, **kwargs)
                    stack.enter_context(patch.object(models, name, side_effect=counted))
                actual = pipeline.run_pipeline(ROOT / "data/demo_dairy_demand.csv", Path(directory) / "new", config)
            for field, value in vars(expected).items():
                if isinstance(value, pd.DataFrame):
                    pd.testing.assert_frame_equal(getattr(actual, field), value, check_exact=True)
            self.assertEqual(actual.summary_text, expected.summary_text)
            self.assertEqual(sum(counts.values()), 132)
            self.assertEqual(sum(n for (family, _), n in counts.items() if family == "_fit_arima"), 112)
            self.assertEqual(sum(n for (_, length), n in counts.items() if length == 70), 6)
            self.assertEqual(actual.method_summary.selected_arima_order.tolist(), ["(2,1,2)", "(0,0,1)", "(2,0,2)", "(2,0,2)", None, None])
            self.assertEqual(actual.inventory_metrics.recommended_quantity_adjustment.tolist(), [74, 833, 257, 413, 712, 668])

    def test_unrounded_demo_scores_and_extended_forecasts(self):
        cleaned = clean_and_preprocess_data(pd.read_csv(ROOT / "data/demo_dairy_demand.csv"), default_lead_time_days=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for product, group in cleaned.groupby("product"):
                with self.subTest(product=product):
                    kwargs = {"horizon": 3, "output_horizon": 14}
                    self.assert_result_equal(models.select_best_forecast_for_product(group, **kwargs), eager_reference(group, **kwargs))

    def test_sample_short_constant_and_zero_series(self):
        sample = clean_and_preprocess_data(pd.read_csv(ROOT / "data/sample_demand.csv"), default_lead_time_days=2)
        histories = [group for _, group in sample.groupby("product")]
        for length, demand in [(4, 10), (16, 10), (16, 0)]:
            histories.append(pd.DataFrame({"date": pd.date_range("2026-01-01", periods=length), "product": "Test", "demand": demand}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for history in histories:
                with self.subTest(product=history["product"].iloc[0], length=len(history), demand=history.demand.iloc[0]):
                    self.assert_result_equal(models.select_best_forecast_for_product(history, horizon=3), eager_reference(history, horizon=3))


class SelectionBehaviorTests(unittest.TestCase):
    def run_selection(self, scores, full_failures=(), holdout_failures=()):
        history = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=16), "product": "Test", "demand": 10})
        fits = []
        specs = []
        for index in range(len(scores)):
            fn = Mock(side_effect=ValueError("full fit failed")) if index in full_failures else Mock(return_value=(pd.Series(np.full(16, index)), SimpleNamespace(forecast=lambda n, value=index: np.full(n, value))))
            fits.append(fn)
            specs.append(models.ModelCandidateSpec(str(index), fn, {}))
        def score(series, fn, points):
            index = fits.index(fn)
            if index in holdout_failures:
                raise ValueError("holdout failed")
            return None, *scores[index]
        with patch.object(models, "build_candidate_specs", return_value=specs), patch.object(models, "_score_model_on_holdout", side_effect=score):
            result = models.select_best_forecast_for_product(history, horizon=3)
        return result, fits

    def test_selection_precedence_and_stable_ties(self):
        for scores, winner in [([(1, 99, 99), (2, 0, 0)], "0"), ([(1, 2, 0), (1, 1, 99)], "1"), ([(1, 1, 2), (1, 1, 1)], "1"), ([(1, 1, 1), (1, 1, 1)], "0")]:
            with self.subTest(scores=scores):
                result, fits = self.run_selection(scores)
                self.assertEqual(result.model_name, winner)
                self.assertEqual(sum(f.call_count for f in fits), 1)

    def test_losers_are_not_full_fitted(self):
        result, fits = self.run_selection([(1, 1, 1), (2, 2, 2), (3, 3, 3)], full_failures=(1, 2))
        self.assertEqual([f.call_count for f in fits], [1, 0, 0])
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.runner_up_model, "1")
        self.assertEqual(result.top_candidates.model_family.tolist(), ["0", "1", "2"])

    def test_full_fit_failure_uses_next_ranked_candidate(self):
        result, fits = self.run_selection([(1, 1, 1), (2, 2, 2), (3, 3, 3)], full_failures=(0,))
        self.assertEqual(result.model_name, "1")
        self.assertEqual(result.mae, 2)
        self.assertEqual(result.future.forecast.tolist(), [1, 1, 1])
        self.assertEqual([f.call_count for f in fits], [1, 1, 0])
        self.assertEqual(result.runner_up_model, "2")
        self.assertEqual(result.top_candidates.model_family.tolist(), ["0", "1", "2"])

    def test_holdout_failure_excluded(self):
        result, fits = self.run_selection([(1, 1, 1), (2, 2, 2)], holdout_failures=(0,))
        self.assertEqual(result.model_name, "1")
        self.assertEqual(result.candidate_count, 1)
        fits[0].assert_not_called()

    def test_all_full_fits_fail(self):
        with self.assertRaisesRegex(ValueError, "No supported model could be fit for product Test"):
            self.run_selection([(1, 1, 1), (2, 2, 2)], full_failures=(0, 1))

    def test_all_holdout_fits_fail(self):
        with self.assertRaisesRegex(ValueError, "No supported model could be fit for product Test"):
            self.run_selection([(1, 1, 1)], holdout_failures=(0,))


if __name__ == "__main__":
    unittest.main()
