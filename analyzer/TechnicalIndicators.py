#!/usr/bin/env python3
"""
Technical Indicators Analyzer
Provides various technical analysis indicators for stock market analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import talib


class TechnicalIndicators:
    """Comprehensive technical indicators analyzer"""
    
    def __init__(self):
        """Initialize the technical indicators analyzer"""
        pass
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series (typically closing prices)
            period: RSI period (default: 14)
            
        Returns:
            RSI values as pandas Series
        """
        try:
            rsi = talib.RSI(prices.values, timeperiod=period)
            return pd.Series(rsi, index=prices.index)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index)
    
    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        try:
            macd, signal, hist = talib.MACD(prices.values, 
                                          fastperiod=fast_period,
                                          slowperiod=slow_period,
                                          signalperiod=signal_period)
            
            return {
                'macd': pd.Series(macd, index=prices.index),
                'signal': pd.Series(signal, index=prices.index),
                'histogram': pd.Series(hist, index=prices.index)
            }
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return {
                'macd': pd.Series(index=prices.index),
                'signal': pd.Series(index=prices.index),
                'histogram': pd.Series(index=prices.index)
            }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        try:
            upper, middle, lower = talib.BBANDS(prices.values, 
                                              timeperiod=period,
                                              nbdevup=std_dev,
                                              nbdevdn=std_dev)
            
            return {
                'upper': pd.Series(upper, index=prices.index),
                'middle': pd.Series(middle, index=prices.index),
                'lower': pd.Series(lower, index=prices.index)
            }
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            return {
                'upper': pd.Series(index=prices.index),
                'middle': pd.Series(index=prices.index),
                'lower': pd.Series(index=prices.index)
            }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, 
                           close: pd.Series, k_period: int = 14, 
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D values
        """
        try:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                      fastk_period=k_period,
                                      slowk_period=d_period,
                                      slowd_period=d_period)
            
            return {
                'k': pd.Series(slowk, index=close.index),
                'd': pd.Series(slowd, index=close.index)
            }
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            return {
                'k': pd.Series(index=close.index),
                'd': pd.Series(index=close.index)
            }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            period: ATR period
            
        Returns:
            ATR values as pandas Series
        """
        try:
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(atr, index=close.index)
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return pd.Series(index=close.index)
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            period: ADX period
            
        Returns:
            ADX values as pandas Series
        """
        try:
            adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(adx, index=close.index)
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return pd.Series(index=close.index)
    
    def calculate_volume_indicators(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators
        
        Args:
            close: Closing prices
            volume: Volume data
            
        Returns:
            Dictionary with volume indicators
        """
        try:
            # On Balance Volume (OBV)
            obv = talib.OBV(close.values, volume.values)
            
            # Volume Rate of Change
            vroc = talib.ROC(volume.values, timeperiod=10)
            
            # Volume Weighted Average Price (VWAP)
            typical_price = (close + close.shift(1) + close.shift(2)) / 3
            vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            
            return {
                'obv': pd.Series(obv, index=close.index),
                'vroc': pd.Series(vroc, index=close.index),
                'vwap': vwap
            }
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            return {
                'obv': pd.Series(index=close.index),
                'vroc': pd.Series(index=close.index),
                'vwap': pd.Series(index=close.index)
            }
    
    def get_support_resistance(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, window: int = 20) -> Dict[str, float]:
        """
        Calculate support and resistance levels
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            window: Window for finding local extremes
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Find local highs and lows
            highs = high.rolling(window=window, center=True).max()
            lows = low.rolling(window=window, center=True).min()
            
            # Current levels
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]
            
            # Recent levels (last 5 periods)
            recent_highs = highs.tail(5).max()
            recent_lows = lows.tail(5).min()
            
            return {
                'support': current_low,
                'resistance': current_high,
                'recent_support': recent_lows,
                'recent_resistance': recent_highs
            }
        except Exception as e:
            print(f"Error calculating support/resistance: {e}")
            return {
                'support': 0.0,
                'resistance': 0.0,
                'recent_support': 0.0,
                'recent_resistance': 0.0
            }
    
    def get_trend_direction(self, prices: pd.Series, short_period: int = 10, 
                           long_period: int = 30) -> str:
        """
        Determine trend direction using moving averages
        
        Args:
            prices: Price series
            short_period: Short-term MA period
            long_period: Long-term MA period
            
        Returns:
            Trend direction: 'uptrend', 'downtrend', or 'sideways'
        """
        try:
            short_ma = prices.rolling(short_period).mean()
            long_ma = prices.rolling(long_period).mean()
            
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            prev_short = short_ma.iloc[-2]
            prev_long = long_ma.iloc[-2]
            
            # Trend determination
            if current_short > current_long and prev_short > prev_long:
                return 'uptrend'
            elif current_short < current_long and prev_short < prev_long:
                return 'downtrend'
            else:
                return 'sideways'
        except Exception as e:
            print(f"Error determining trend: {e}")
            return 'unknown'
    
    def get_all_indicators(self, ohlc_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all technical indicators for OHLC data
        
        Args:
            ohlc_data: DataFrame with Open, High, Low, Close columns
            
        Returns:
            Dictionary with all calculated indicators
        """
        try:
            indicators = {}
            
            # Basic indicators
            indicators['rsi'] = self.calculate_rsi(ohlc_data['Close'])
            
            macd_data = self.calculate_macd(ohlc_data['Close'])
            indicators.update(macd_data)
            
            bb_data = self.calculate_bollinger_bands(ohlc_data['Close'])
            indicators.update(bb_data)
            
            stoch_data = self.calculate_stochastic(ohlc_data['High'], 
                                                 ohlc_data['Low'], 
                                                 ohlc_data['Close'])
            indicators.update(stoch_data)
            
            indicators['atr'] = self.calculate_atr(ohlc_data['High'], 
                                                 ohlc_data['Low'], 
                                                 ohlc_data['Close'])
            
            indicators['adx'] = self.calculate_adx(ohlc_data['High'], 
                                                 ohlc_data['Low'], 
                                                 ohlc_data['Close'])
            
            # Volume indicators (if volume data available)
            if 'Volume' in ohlc_data.columns:
                vol_data = self.calculate_volume_indicators(ohlc_data['Close'], 
                                                          ohlc_data['Volume'])
                indicators.update(vol_data)
            
            return indicators
        except Exception as e:
            print(f"Error calculating all indicators: {e}")
            return {}
