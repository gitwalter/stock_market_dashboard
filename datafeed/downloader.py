#!/usr/bin/env python3
"""
Enhanced data downloader for Stock Market Dashboard
Provides robust data downloading with error handling, rate limiting, and logging
"""

import pandas as pd
import yfinance as yf
import numpy as np
import time
from typing import List, Optional, Union
from datetime import datetime

# Import our utilities
from config import config
from utils.exceptions import DataDownloadError, ValidationError, RateLimitError
from utils.validation import validate_ticker_list, validate_date_range, validate_dataframe, validate_ticker_symbol
from utils.logging import logger, monitor_performance
from utils.rate_limiter import yahoo_finance_limiter


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
        """Internal method to download prices"""
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
                start=self.start,
                end=self.end,
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
        
        # Validate final result
        if self.collected_prices.empty:
            raise DataDownloadError("No data downloaded for any ticker")
        
        return self.collected_prices


# Backward compatibility
BatchPriceDownloader = RobustBatchPriceDownloader        