import unittest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import TradingAnalyzer

class TestTradingAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = TradingAnalyzer()

    def test_load_data(self):
        self.analyzer.load_data()
        self.assertIsNotNone(self.analyzer.data)
        self.assertIsInstance(self.analyzer.data, pd.DataFrame)
        self.assertGreater(len(self.analyzer.data), 0)

    def test_train_model(self):
        self.analyzer.load_data()
        self.analyzer.train_model()
        self.assertIsNotNone(self.analyzer.model)
        self.assertIn("classification_report", self.analyzer.results)

    def test_generate_signals(self):
        self.analyzer.load_data()
        self.analyzer.train_model()
        signals_data = self.analyzer.generate_signals()
        self.assertIn("signal", signals_data.columns)

    def test_backtest_strategy(self):
        self.analyzer.load_data()
        self.analyzer.train_model()
        self.analyzer.generate_signals()
        cumulative_returns = self.analyzer.backtest_strategy()
        self.assertIsInstance(cumulative_returns, float)

    def test_visualize(self):
        self.analyzer.load_data()
        self.analyzer.train_model()
        self.analyzer.generate_signals()
        self.analyzer.visualize()
        # Check if the image file was created (this is a basic check)
        import os
        img_path = os.path.join(os.getcwd(), 'docs', 'img', 'automated_trading_algorithm_analysis.png')
        self.assertTrue(os.path.exists(img_path))

if __name__ == '__main__':
    unittest.main()

