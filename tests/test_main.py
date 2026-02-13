"""Tests for src.main — uses synthetic deterministic data (no network)."""

import sys
import os
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import (
    compute_sma,
    compute_rsi,
    compute_macd,
    build_features,
    train_model,
    backtest,
    FEATURE_COLS,
)


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a slight upward trend."""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5 + 0.02)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(1_000_000, 10_000_000, n)
    dates = pd.bdate_range(end="2025-01-31", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestIndicators(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    def test_sma_length(self):
        sma = compute_sma(self.series, 3)
        self.assertEqual(len(sma), len(self.series))
        self.assertTrue(sma.iloc[:2].isna().all())

    def test_sma_value(self):
        sma = compute_sma(self.series, 3)
        self.assertAlmostEqual(sma.iloc[2], 2.0)  # mean(1,2,3)
        self.assertAlmostEqual(sma.iloc[3], 3.0)  # mean(2,3,4)

    def test_rsi_range(self):
        rsi = compute_rsi(self.series, window=5)
        non_nan = rsi.dropna()
        self.assertTrue((non_nan >= 0).all())
        self.assertTrue((non_nan <= 100).all())

    def test_rsi_monotonic_up(self):
        """A monotonically increasing series should have RSI = 100."""
        rsi = compute_rsi(self.series, window=3)
        # After warm-up, avg_loss=0 → RSI should be 100
        self.assertAlmostEqual(rsi.iloc[-1], 100.0)

    def test_macd_columns(self):
        macd = compute_macd(self.series, fast=3, slow=5, signal=3)
        self.assertListEqual(
            sorted(macd.columns.tolist()),
            ["macd", "macd_hist", "macd_signal"],
        )

    def test_macd_length(self):
        macd = compute_macd(self.series, fast=3, slow=5, signal=3)
        self.assertEqual(len(macd), len(self.series))


class TestBuildFeatures(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(300)

    def test_output_columns(self):
        feat = build_features(self.df)
        for col in FEATURE_COLS + ["target"]:
            self.assertIn(col, feat.columns, f"Missing column: {col}")

    def test_no_nans(self):
        feat = build_features(self.df)
        self.assertFalse(feat[FEATURE_COLS].isna().any().any())

    def test_target_binary(self):
        feat = build_features(self.df)
        self.assertTrue(set(feat["target"].unique()).issubset({0, 1}))

    def test_fewer_rows_than_input(self):
        feat = build_features(self.df)
        self.assertLess(len(feat), len(self.df))


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.feat = build_features(_make_ohlcv(300))

    def test_result_keys(self):
        result = train_model(self.feat)
        for key in ("model", "X_train", "X_test", "y_train", "y_test",
                     "y_pred", "report", "accuracy"):
            self.assertIn(key, result)

    def test_accuracy_range(self):
        result = train_model(self.feat)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_temporal_split_no_leakage(self):
        """Train set dates must all precede test set dates."""
        result = train_model(self.feat)
        train_end = result["X_train"].index[-1]
        test_start = result["X_test"].index[0]
        self.assertLess(train_end, test_start)

    def test_predictions_length(self):
        result = train_model(self.feat)
        self.assertEqual(len(result["y_pred"]), len(result["y_test"]))


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.feat = build_features(_make_ohlcv(300))
        self.result = train_model(self.feat)

    def test_backtest_columns(self):
        bt = backtest(self.feat, self.result["model"])
        for col in ("signal", "strategy_return", "cumulative_return"):
            self.assertIn(col, bt.columns)

    def test_signal_values(self):
        bt = backtest(self.feat, self.result["model"])
        self.assertTrue(set(bt["signal"].unique()).issubset({0, 1}))

    def test_cumulative_return_type(self):
        bt = backtest(self.feat, self.result["model"])
        self.assertIsInstance(bt["cumulative_return"].iloc[-1], (float, np.floating))

    def test_no_nans_in_returns(self):
        bt = backtest(self.feat, self.result["model"])
        self.assertFalse(bt["strategy_return"].isna().any())


if __name__ == "__main__":
    unittest.main()
