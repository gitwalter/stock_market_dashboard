#!/usr/bin/env python3
"""
Enhanced data downloader for Stock Market Dashboard
Provides robust data downloading with error handling, rate limiting, and logging
"""

import pandas as pd
import yfinance as yf
import numpy as np
import time
import os
import requests
from typing import List, Optional, Union
from datetime import datetime

# Import our utilities
from config import config
from utils.exceptions import DataDownloadError, ValidationError, RateLimitError
from utils.validation import validate_ticker_list, validate_date_range, validate_dataframe, validate_ticker_symbol
from utils.logging import logger, monitor_performance
from utils.rate_limiter import yahoo_finance_limiter


def is_streamlit_cloud():
    """Check if running on Streamlit Cloud"""
    return os.environ.get('STREAMLIT_SERVER_PORT') is not None


class CloudCompatibleDownloader:
    """Downloader optimized for Streamlit Cloud deployment"""
    
    @staticmethod
    def download_with_fallbacks(tickers, start_date, end_date, interval, fields):
        """
        Download data with multiple fallback strategies for cloud deployment
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            fields: Fields to download
            
        Returns:
            DataFrame with price data
        """
        # Strategy 1: Standard download
        try:
            logger.log_data_download(tickers, True, 0, "Strategy 1: Standard download")
            result = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='column',
                auto_adjust=True,
                prepost=True,
                threads=True,
                proxy=None
            )
            if not result.empty:
                return result[fields]
        except Exception as e:
            logger.log_error("download_strategy", f"Strategy 1 failed: {e}")
        
        # Strategy 2: Disable threading (often helps on cloud)
        try:
            logger.log_data_download(tickers, True, 0, "Strategy 2: No threading")
            result = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='column',
                auto_adjust=True,
                prepost=False,
                threads=False,
                proxy=None
            )
            if not result.empty:
                return result[fields]
        except Exception as e:
            logger.log_error("download_strategy", f"Strategy 2 failed: {e}")
        
        # Strategy 3: Single ticker downloads (more reliable on cloud)
        if len(tickers) > 1:
            try:
                logger.log_data_download(tickers, True, 0, "Strategy 3: Single ticker downloads")
                results = []
                for ticker in tickers:
                    try:
                        ticker_data = yf.download(
                            tickers=ticker,
                            start=start_date,
                            end=end_date,
                            interval=interval,
                            auto_adjust=True,
                            prepost=False,
                            threads=False,
                            proxy=None
                        )
                        if not ticker_data.empty:
                            results.append(ticker_data)
                        time.sleep(0.1)  # Small delay between requests
                    except Exception as e:
                        logger.log_error("single_ticker", f"Failed to download {ticker}: {e}")
                        continue
                
                if results:
                    # Combine results
                    combined = pd.concat(results, axis=1, keys=tickers)
                    combined.columns = combined.columns.rename("prices", level=0)
                    combined.columns = combined.columns.rename("symbol", level=1)
                    return combined[fields]
            except Exception as e:
                logger.log_error("download_strategy", f"Strategy 3 failed: {e}")
        
        # Strategy 4: Use different interval if intraday fails
        if interval in ['1m', '2m', '5m', '15m', '30m']:
            try:
                logger.log_data_download(tickers, True, 0, f"Strategy 4: Fallback to 1h interval")
                result = yf.download(
                    tickers=tickers,
                    start=start_date,
                    end=end_date,
                    interval='1h',
                    group_by='column',
                    auto_adjust=True,
                    prepost=False,
                    threads=False,
                    proxy=None
                )
                if not result.empty:
                    return result[fields]
            except Exception as e:
                logger.log_error("download_strategy", f"Strategy 4 failed: {e}")
        
        # Strategy 5: Use daily data as last resort
        try:
            logger.log_data_download(tickers, True, 0, "Strategy 5: Daily data fallback")
            result = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                interval='1d',
                group_by='column',
                auto_adjust=True,
                prepost=False,
                threads=False,
                proxy=None
            )
            if not result.empty:
                return result[fields]
        except Exception as e:
            logger.log_error("download_strategy", f"Strategy 5 failed: {e}")
        
        raise DataDownloadError("All download strategies failed")


class InfoDownloader:
    """Enhanced info downloader with error handling and validation"""
    
    def __init__(self, ticker_name: str):
        """
        Initialize info downloader
        
        Args:
            ticker_name: Ticker symbol to download info for
            
        Raises:
            ValidationError: If ticker is invalid
        """
        if not validate_ticker_symbol(ticker_name):
            raise ValidationError(f"Invalid ticker symbol: {ticker_name}")
        
        self.ticker_name = ticker_name
        self.ticker = yf.Ticker(ticker_name)
    
    @monitor_performance("info_download")
    def info(self) -> pd.DataFrame:
        """
        Download ticker information
        
        Returns:
            DataFrame with ticker information
            
        Raises:
            DataDownloadError: If download fails
        """
        try:
            info_dict = self.ticker.info
            if isinstance(info_dict, dict):
                result = pd.DataFrame([info_dict])
            else:
                result = pd.DataFrame(info_dict())
            
            validate_dataframe(result)
            logger.log_data_download([self.ticker_name], True, 0)
            return result
            
        except Exception as e:
            error_msg = f"Failed to download info for {self.ticker_name}: {e}"
            logger.log_data_download([self.ticker_name], False, 0, str(e))
            raise DataDownloadError(error_msg) from e
    
    @monitor_performance("balance_sheet_download")
    def balance_sheet(self) -> pd.DataFrame:
        """
        Download balance sheet data
        
        Returns:
            DataFrame with balance sheet data
            
        Raises:
            DataDownloadError: If download fails
        """
        try:
            result = pd.DataFrame(self.ticker.balance_sheet())
            validate_dataframe(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to download balance sheet for {self.ticker_name}: {e}"
            logger.log_error("balance_sheet_download", error_msg)
            raise DataDownloadError(error_msg) from e
    
    @monitor_performance("fast_info_download")
    def fast_info(self) -> pd.DataFrame:
        """
        Download fast info (basic ticker information)
        
        Returns:
            DataFrame with basic ticker information
            
        Raises:
            DataDownloadError: If download fails
        """
        try:
            info_dict = self.ticker.info
            if isinstance(info_dict, dict):
                result = pd.DataFrame([info_dict])
            else:
                result = pd.DataFrame(info_dict())
            
            validate_dataframe(result)
            return result
            
        except Exception as e:
            error_msg = f"Failed to download fast info for {self.ticker_name}: {e}"
            logger.log_data_download([self.ticker_name], False, 0, str(e))
            return pd.DataFrame()  # Return empty DataFrame instead of raising
    
    @monitor_performance("news_download")
    def get_news(self) -> pd.DataFrame:
        """
        Download news data
        
        Returns:
            DataFrame with news data
            
        Raises:
            DataDownloadError: If download fails
        """
        try:
            result = pd.DataFrame(self.ticker.get_news())
            return result
            
        except Exception as e:
            error_msg = f"Failed to download news for {self.ticker_name}: {e}"
            logger.log_error("news_download", error_msg)
            return pd.DataFrame()  # Return empty DataFrame for news


class RobustBatchPriceDownloader:
    """Enhanced batch price downloader with retry logic and error handling"""
    
    def __init__(self, ticker_list: List[str], start: Union[str, datetime], 
                 end: Union[str, datetime], interval: str):
        """
        Initialize batch price downloader
        
        Args:
            ticker_list: List of ticker symbols
            start: Start date
            end: End date
            interval: Data interval
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        self.ticker_list = validate_ticker_list(ticker_list)
        self.start, self.end = validate_date_range(start, end)
        self.interval = interval
        
        # Use configuration
        self.batch_size = config.batch_size
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.fields = ['Open', 'Low', 'High', 'Close', 'Volume']
        
        # Calculate loop size
        self.loop_size = int(len(self.ticker_list) // self.batch_size) + 2
        
        # Initialize data structure
        iterables = [self.fields, self.ticker_list]
        price_columns = pd.MultiIndex.from_product(iterables, names=["prices", "symbol"])
        self.collected_prices = pd.DataFrame(columns=price_columns, index=pd.to_datetime([]))
    
    @monitor_performance("batch_price_download")
    def get_yahoo_prices(self) -> pd.DataFrame:
        """
        Download prices with retry logic and rate limiting
        
        Returns:
            DataFrame with price data
            
        Raises:
            DataDownloadError: If download fails after all retries
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                result = self._download_prices()
                duration = time.time() - start_time
                logger.log_data_download(self.ticker_list, True, duration)
                return result
                
            except (RateLimitError, DataDownloadError) as e:
                if attempt == self.max_retries - 1:
                    duration = time.time() - start_time
                    logger.log_data_download(self.ticker_list, False, duration, str(e))
                    raise DataDownloadError(f"Failed to download prices after {self.max_retries} attempts: {e}") from e
                
                # Wait before retry
                wait_time = self.retry_delay * (2 ** attempt)
                logger.log_error("retry_wait", f"Waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
    
    def _download_prices(self) -> pd.DataFrame:
        """Internal method to download prices with fallback for intraday data"""
        from datetime import timedelta
        
        # Check if running on Streamlit Cloud
        if is_streamlit_cloud():
            logger.log_data_download(self.ticker_list, True, 0, "Using cloud-compatible downloader")
            try:
                result = CloudCompatibleDownloader.download_with_fallbacks(
                    self.ticker_list, self.start, self.end, self.interval, self.fields
                )
                return result
            except Exception as e:
                logger.log_error("cloud_download", f"Cloud downloader failed: {e}")
                # Fall back to standard method
        
        # Define fallback date ranges for intraday data
        fallback_ranges = []
        if self.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
            # For intraday intervals, try 1-2 days before if current range fails
            current_start = self.start
            current_end = self.end
            
            # Try 1 day before
            fallback_ranges.append((
                current_start - timedelta(days=1),
                current_end - timedelta(days=1),
                "1 day before"
            ))
            
            # Try 2 days before
            fallback_ranges.append((
                current_start - timedelta(days=2),
                current_end - timedelta(days=2),
                "2 days before"
            ))
        
        # Try original date range first
        date_ranges = [(self.start, self.end, "original")] + fallback_ranges
        
        for start_date, end_date, range_description in date_ranges:
            try:
                logger.log_data_download(self.ticker_list, True, 0, f"Trying {range_description}")
                
                for t in range(1, self.loop_size):
                    m = (t - 1) * self.batch_size
                    n = t * self.batch_size
                    batch_list = self.ticker_list[m:n]
                    
                    if len(batch_list) == 0:
                        break
                    
                    logger.log_data_download(batch_list, True, 0)
                    
                    # Use rate limiter for Yahoo Finance calls
                    batch_download = yahoo_finance_limiter.call(
                        yf.download,
                        tickers=batch_list,
                        start=start_date,
                        end=end_date,
                        interval=self.interval,
                        group_by='column',
                        auto_adjust=True,
                        prepost=True,
                        threads=True,
                        proxy=None
                    )[self.fields]
                    
                    # Handle column naming
                    if len(batch_list) > 1:
                        batch_download.columns = batch_download.columns.rename("prices", level=0)
                        batch_download.columns = batch_download.columns.rename("symbol", level=1)
                    
                    # Update collected prices
                    if self.collected_prices.empty:
                        if len(batch_list) > 1:
                            self.collected_prices = pd.concat([self.collected_prices, batch_download], ignore_index=False)
                        else:
                            # Handle single ticker case
                            if isinstance(batch_download.columns, pd.MultiIndex):
                                self.collected_prices = batch_download
                            else:
                                # Create proper MultiIndex for single ticker
                                single_ticker = batch_list[0]
                                multi_cols = pd.MultiIndex.from_product([self.fields, [single_ticker]], names=["prices", "symbol"])
                                self.collected_prices = pd.DataFrame(
                                    batch_download.values,
                                    index=batch_download.index,
                                    columns=multi_cols
                                )
                    else:
                        self.collected_prices.update(batch_download)
                
                # If we get here, the download was successful
                if not self.collected_prices.empty:
                    logger.log_data_download(self.ticker_list, True, 0, f"Success with {range_description}")
                    return self.collected_prices
                    
            except Exception as e:
                logger.log_error("download_fallback", f"Failed with {range_description}: {e}")
                # Reset collected prices for next attempt
                self.collected_prices = pd.DataFrame(columns=self.collected_prices.columns, index=pd.to_datetime([]))
                continue
        
        # If all attempts failed
        raise DataDownloadError("No data downloaded for any ticker after trying all fallback date ranges")


# Backward compatibility
BatchPriceDownloader = RobustBatchPriceDownloader        