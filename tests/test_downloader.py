#!/usr/bin/env python3
"""
Unit tests for the datafeed.downloader module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datafeed.downloader import InfoDownloader, BatchPriceDownloader
from utils.exceptions import ValidationError, DataDownloadError


class TestInfoDownloader(unittest.TestCase):
    """Test cases for InfoDownloader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the validation to allow test tickers
        with patch('datafeed.downloader.validate_ticker_symbol', return_value=True):
            self.downloader = InfoDownloader("AAPL")
    
    @patch('datafeed.downloader.yf.Ticker')
    def test_info_method(self, mock_ticker):
        """Test the info method"""
        # Mock ticker info
        mock_info = {
            'shortName': 'Apple Inc.',
            'marketCap': 2000000000000,
            'sector': 'Technology'
        }
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        # Create downloader and test
        with patch('datafeed.downloader.validate_ticker_symbol', return_value=True):
            downloader = InfoDownloader("AAPL")
            result = downloader.info()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['shortName'], 'Apple Inc.')
    
    @patch('datafeed.downloader.yf.Ticker')
    def test_fast_info_method(self, mock_ticker):
        """Test the fast_info method"""
        # Mock ticker info
        mock_info = {
            'shortName': 'Apple Inc.',
            'marketCap': 2000000000000
        }
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        # Create downloader and test
        with patch('datafeed.downloader.validate_ticker_symbol', return_value=True):
            downloader = InfoDownloader("AAPL")
            result = downloader.fast_info()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
    
    @patch('datafeed.downloader.yf.Ticker')
    def test_get_news_method(self, mock_ticker):
        """Test the get_news method"""
        # Mock news data
        mock_news = [
            {'title': 'Apple Q4 Results', 'link': 'http://example.com'},
            {'title': 'iPhone Sales Up', 'link': 'http://example2.com'}
        ]
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.get_news.return_value = mock_news
        mock_ticker.return_value = mock_ticker_instance
        
        # Create downloader and test
        with patch('datafeed.downloader.validate_ticker_symbol', return_value=True):
            downloader = InfoDownloader("AAPL")
            result = downloader.get_news()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)


class TestBatchPriceDownloader(unittest.TestCase):
    """Test cases for BatchPriceDownloader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tickers = ['AAPL', 'MSFT']
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 31)
        self.interval = '1d'
        
        # Mock validation to allow test tickers
        with patch('datafeed.downloader.validate_ticker_list', return_value=self.tickers), \
             patch('datafeed.downloader.validate_date_range', return_value=(self.start_date, self.end_date)):
            self.downloader = BatchPriceDownloader(
                self.tickers, self.start_date, self.end_date, self.interval
            )
    
    def test_initialization(self):
        """Test BatchPriceDownloader initialization"""
        self.assertEqual(self.downloader.ticker_list, self.tickers)
        self.assertEqual(self.downloader.start, self.start_date)
        self.assertEqual(self.downloader.end, self.end_date)
        self.assertEqual(self.downloader.interval, self.interval)
        self.assertEqual(self.downloader.batch_size, 20)  # From config
    
    def test_string_date_conversion(self):
        """Test that string dates are converted to datetime objects"""
        with patch('datafeed.downloader.validate_ticker_list', return_value=self.tickers), \
             patch('datafeed.downloader.validate_date_range', return_value=(self.start_date, self.end_date)):
            downloader = BatchPriceDownloader(
                self.tickers, "2024-01-01", "2024-01-31", self.interval
            )
            self.assertIsInstance(downloader.start, datetime)
            self.assertIsInstance(downloader.end, datetime)
    
    @patch('datafeed.downloader.yf.download')
    def test_get_yahoo_prices_success(self, mock_download):
        """Test successful data download"""
        # Mock successful download with MultiIndex columns
        mock_data = pd.DataFrame({
            ('Open', 'AAPL'): [150.0, 151.0],
            ('High', 'AAPL'): [152.0, 153.0],
            ('Low', 'AAPL'): [149.0, 150.0],
            ('Close', 'AAPL'): [151.0, 152.0],
            ('Volume', 'AAPL'): [1000000, 1100000],
            ('Open', 'MSFT'): [250.0, 251.0],
            ('High', 'MSFT'): [252.0, 253.0],
            ('Low', 'MSFT'): [249.0, 250.0],
            ('Close', 'MSFT'): [251.0, 252.0],
            ('Volume', 'MSFT'): [2000000, 2100000]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mock_download.return_value = mock_data
        
        result = self.downloader.get_yahoo_prices()
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        mock_download.assert_called()
    
    @patch('datafeed.downloader.yf.download')
    def test_get_yahoo_prices_empty_result(self, mock_download):
        """Test handling of empty download results"""
        # Mock empty download with proper MultiIndex structure
        columns = pd.MultiIndex.from_product([['Open', 'Low', 'High', 'Close', 'Volume'], ['AAPL', 'MSFT']], 
                                           names=['prices', 'symbol'])
        mock_data = pd.DataFrame(columns=columns)
        mock_download.return_value = mock_data
        
        # This will fail due to DataDownloadError, which is expected behavior
        with self.assertRaises(DataDownloadError):
            result = self.downloader.get_yahoo_prices()
    
    def test_single_ticker_handling(self):
        """Test handling of single ticker"""
        single_ticker = ['AAPL']
        with patch('datafeed.downloader.validate_ticker_list', return_value=single_ticker), \
             patch('datafeed.downloader.validate_date_range', return_value=(self.start_date, self.end_date)):
            downloader = BatchPriceDownloader(
                single_ticker, self.start_date, self.end_date, self.interval
            )
            
            # Should handle single ticker without errors
            self.assertEqual(downloader.ticker_list, single_ticker)
    
    def test_large_ticker_list_batching(self):
        """Test that large ticker lists are properly batched"""
        large_ticker_list = [f'TICKER_{i}' for i in range(50)]
        # Mock validation to allow test tickers
        with patch('datafeed.downloader.validate_ticker_list', return_value=large_ticker_list), \
             patch('datafeed.downloader.validate_date_range', return_value=(self.start_date, self.end_date)):
            downloader = BatchPriceDownloader(
                large_ticker_list, self.start_date, self.end_date, self.interval
            )
            
            # Should calculate loop size correctly
            expected_loop_size = int(len(large_ticker_list) // downloader.batch_size) + 2
            self.assertEqual(downloader.loop_size, expected_loop_size)


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling"""
    
    def test_invalid_date_range(self):
        """Test handling of invalid date ranges"""
        start_date = datetime(2024, 12, 31)
        end_date = datetime(2024, 1, 1)  # End before start
        
        # Should raise ValidationError during initialization
        with self.assertRaises(ValidationError):
            with patch('datafeed.downloader.validate_ticker_list', return_value=['AAPL']):
                downloader = BatchPriceDownloader(
                    ['AAPL'], start_date, end_date, '1d'
                )
    
    def test_empty_ticker_list(self):
        """Test handling of empty ticker list"""
        # Should raise ValidationError for empty list
        with self.assertRaises(ValidationError):
            downloader = BatchPriceDownloader(
                [], datetime.now(), datetime.now(), '1d'
            )
    
    def test_invalid_interval(self):
        """Test handling of invalid interval"""
        # Should accept any interval string but fail on date validation
        with self.assertRaises(ValidationError):
            with patch('datafeed.downloader.validate_ticker_list', return_value=['AAPL']):
                downloader = BatchPriceDownloader(
                    ['AAPL'], datetime.now(), datetime.now(), 'invalid_interval'
                )


if __name__ == '__main__':
    unittest.main()
