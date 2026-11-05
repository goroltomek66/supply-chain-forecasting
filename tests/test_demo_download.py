"""The public demo is bundled, downloadable, and never analyzed automatically."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app

CARGO = ROOT / "data" / "Cargo_Data.csv"


class DemoDownloadTests(unittest.TestCase):
    def test_bundled_cargo_schema_and_daily_coverage(self):
        self.assertTrue(CARGO.is_file())
        data = pd.read_csv(CARGO)
        self.assertEqual(data.shape, (720, 11))
        self.assertEqual(data.columns.tolist(), pd.read_csv(ROOT / "data/demo_dairy_demand.csv", nrows=0).columns.tolist())
        self.assertEqual(data["product"].nunique(), 6)
        self.assertFalse(data.duplicated(["product", "date"]).any())
        expected = pd.date_range("2026-07-01", periods=120)
        for _, group in data.groupby("product"):
            dates = pd.DatetimeIndex(pd.to_datetime(group["date"], format="%m/%d/%y")).sort_values()
            pd.testing.assert_index_equal(dates, expected, check_names=False)

    def test_download_uses_cargo_bytes_without_automatic_analysis(self):
        with patch.object(st, "download_button", wraps=st.download_button) as download, \
             patch.object(app, "run_pipeline") as pipeline:
            page = AppTest.from_string("import app\napp.main()", default_timeout=30).run()
        self.assertEqual(len(page.exception), 0)
        download.assert_called_once()
        self.assertEqual(download.call_args.args, ("Download Demo Dataset",))
        self.assertEqual(download.call_args.kwargs["file_name"], "Cargo_Data.csv")
        self.assertEqual(download.call_args.kwargs["data"], CARGO.read_bytes())
        self.assertEqual(download.call_args.kwargs["mime"], "text/csv")
        self.assertEqual(download.call_args.kwargs["on_click"], "ignore")
        pipeline.assert_not_called()

    def test_missing_demo_keeps_uploader_available(self):
        original_read = Path.read_bytes

        def read(path):
            if path == CARGO:
                raise FileNotFoundError("Demo unavailable")
            return original_read(path)

        with patch.object(Path, "read_bytes", read), patch.object(app, "run_pipeline") as pipeline:
            page = AppTest.from_string("import app\napp.main()", default_timeout=30).run()
        self.assertEqual(len(page.exception), 0)
        self.assertTrue(any("demo dataset is currently unavailable" in item.value for item in page.info))
        self.assertEqual(len(page.get("file_uploader")), 1)
        pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
