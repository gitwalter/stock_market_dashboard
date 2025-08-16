#!/usr/bin/env python3
"""
Unit tests for the analyzer modules
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer.MomentumScore import MomentumScore


class TestMomentumScore(unittest.TestCase):
    """Test cases for MomentumScore class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.momentum = MomentumScore(vola_window=20)
        
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.sample_prices = pd.Series(
            np.random.randn(100).cumsum() + 100,  # Random walk starting at 100
            index=dates
        )
        
        # Create sample multi-column data
        self.sample_multi_prices = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 150,
            'MSFT': np.random.randn(100).cumsum() + 200,
            'GOOGL': np.random.randn(100).cumsum() + 250
        }, index=dates)
    
    def test_initialization(self):
        """Test MomentumScore initialization"""
        self.assertEqual(self.momentum.vola_window, 20)
        
        # Test with custom window
        custom_momentum = MomentumScore(vola_window=50)
        self.assertEqual(custom_momentum.vola_window, 50)
    
    def test_get_score_with_positive_trend(self):
        """Test get_score with upward trending data"""
        # Create upward trending data
        upward_data = pd.Series(np.arange(100) + 100)
        score = self.momentum.get_score(upward_data)
        
        # Score should be positive for upward trend
        self.assertGreater(score, 0)
        self.assertIsInstance(score, float)
    
    def test_get_score_with_negative_trend(self):
        """Test get_score with downward trending data"""
        # Create downward trending data
        downward_data = pd.Series(200 - np.arange(100))
        score = self.momentum.get_score(downward_data)
        
        # Score should be negative for downward trend
        self.assertLess(score, 0)
        self.assertIsInstance(score, float)
    
    def test_get_score_with_flat_data(self):
        """Test get_score with flat data"""
        # Create flat data
        flat_data = pd.Series([100] * 100)
        score = self.momentum.get_score(flat_data)
        
        # Score should be close to zero for flat data
        self.assertAlmostEqual(score, 0, places=1)
    
    def test_get_score_with_insufficient_data(self):
        """Test get_score with insufficient data"""
        # Test with very short series
        short_data = pd.Series([100, 101, 102])
        score = self.momentum.get_score(short_data)
        
        # Should still return a valid score
        self.assertIsInstance(score, float)
    
    def test_get_intraday_momentum(self):
        """Test get_intraday_momentum method"""
        result = self.momentum.get_intraday_momentum(self.sample_multi_prices)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # 3 tickers
        self.assertIn('symbol', result.columns)
        self.assertIn('day_change', result.columns)
        self.assertIn('momentum', result.columns)
        
        # Check that momentum values are between 0 and 1
        momentum_values = result['momentum'].astype(float)
        self.assertTrue(all(0 <= val <= 1 for val in momentum_values))
    
    def test_get_volatility(self):
        """Test get_volatility method"""
        volatility = self.momentum.get_volatility(self.sample_prices)
        
        # Assertions
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0)  # Volatility should be non-negative
    
    def test_get_volatility_with_insufficient_data(self):
        """Test get_volatility with insufficient data"""
        # Test with data shorter than vola_window
        short_data = pd.Series([100, 101, 102, 103, 104])
        volatility = self.momentum.get_volatility(short_data)
        
        # Should handle insufficient data gracefully
        self.assertIsInstance(volatility, float)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty series
        empty_series = pd.Series([])
        with self.assertRaises(Exception):
            self.momentum.get_score(empty_series)
        
        # Test with NaN values
        nan_series = pd.Series([100, np.nan, 102, 103, 104])
        score = self.momentum.get_score(nan_series)
        self.assertIsInstance(score, float)
        
        # Test with zero values - this actually works in the current implementation
        # due to numpy's handling of log(0) which returns -inf
        zero_series = pd.Series([0, 0, 0, 0, 0])
        score = self.momentum.get_score(zero_series)
        self.assertIsInstance(score, float)


class TestMomentumScoreIntegration(unittest.TestCase):
    """Integration tests for MomentumScore"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.momentum = MomentumScore()
        
        # Create realistic price data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range('2024-01-01', periods=252, freq='D')  # One trading year
        
        # Create realistic price movements
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.realistic_prices = pd.Series(prices, index=dates)
    
    def test_realistic_price_analysis(self):
        """Test with realistic price data"""
        score = self.momentum.get_score(self.realistic_prices)
        volatility = self.momentum.get_volatility(self.realistic_prices)
        
        # Assertions for realistic values
        self.assertIsInstance(score, float)
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0)
        
        # Score should be reasonable (not extreme)
        self.assertLess(abs(score), 1000)  # Should not be extremely large
    
    def test_consistency_across_runs(self):
        """Test that results are consistent across multiple runs"""
        scores = []
        volatilities = []
        
        for _ in range(5):
            score = self.momentum.get_score(self.realistic_prices)
            volatility = self.momentum.get_volatility(self.realistic_prices)
            scores.append(score)
            volatilities.append(volatility)
        
        # Results should be consistent
        self.assertAlmostEqual(max(scores), min(scores), places=10)
        self.assertAlmostEqual(max(volatilities), min(volatilities), places=10)


if __name__ == '__main__':
    unittest.main()
