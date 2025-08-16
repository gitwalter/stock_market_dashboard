#!/usr/bin/env python3
"""
Configuration management for Stock Market Dashboard
Centralizes all configuration settings and removes hard-coded values
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DashboardConfig:
    """Configuration for the dashboard"""
    
    # Data settings
    batch_size: int = 20
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Chart settings
    default_chart_height: int = 600
    default_chart_width: int = 800
    
    # Strategy settings
    default_commission: float = 0.001
    default_slippage: float = 0.001
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bollinger_period: int = 20
    bollinger_std_dev: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    
    # RSI Strategy settings
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_exit: int = 50
    rsi_stop_loss: float = 0.05
    rsi_take_profit: float = 0.10
    
    # Moving averages
    moving_average_periods: List[int] = field(default_factory=lambda: [20, 50, 100, 150, 200])
    moving_average_colors: List[str] = field(default_factory=lambda: ['green', 'blue', 'orange', 'purple', 'red'])
    
    # Ticker lists
    overview_tickers: List[str] = field(default_factory=lambda: [
        '^DJI', '^GSPC', '^NDX', '^IXIC', '^GDAXI', '^HSI', 'EURUSD=X', 'GC=F', 'BTC-USD'
    ])
    
    sector_etf: Dict[str, str] = field(default_factory=lambda: {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Finance': 'XLF',
        'Real Estate': 'XLRE',
        'Energy': 'XLE',
        'Materials': 'XLB',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Utilities': 'XLU',
        'Consumer Staples': 'XLP',
        'Telecommunication': 'XLC'
    })
    
    # Filter options
    instrument_types: List[str] = field(default_factory=lambda: [
        "Stocks", "Currencies", "Commodities", "Cryptocurrencies", "ETF"
    ])
    
    exchanges: List[str] = field(default_factory=lambda: [
        "NASDAQ", "NYSE", "London", "Frankfurt", "Paris", "Amsterdam",
        "BorsaItaliana", "BolsaDeMadrid", "Oslo", "Zurich", "HongKong", 
        "Helsinki", "Copenhagen", "Stockholm"
    ])
    
    industries: List[str] = field(default_factory=lambda: [
        "-", "BasicMaterials", "ConsumerGoods", "Technology",
        "Services", "Financial", "IndustrialGoods", "Healthcare",
        "Conglomerates", "Utilities"
    ])
    
    # Technical indicators for charts
    quant_figure_indicators: List[str] = field(default_factory=lambda: [
        'bb', 'rsi'
    ])
    
    quant_figure_methods: List[str] = field(default_factory=lambda: [
        'add_bollinger_bands', 'add_rsi'
    ])
    
    # App options
    app_options: List[str] = field(default_factory=lambda: [
        "Overview", "Sectors", "Chart", "Watchlist", "Momentum", "Returns", "Backtest"
    ])
    
    # Strategy names
    strategy_names: List[str] = field(default_factory=lambda: [
        "BuyAndHold", "MinerviniMomentum", "TrailingStopLoss", "SmaCross", "RSIStrategy"
    ])
    
    @classmethod
    def from_file(cls, filepath: str) -> 'DashboardConfig':
        """Load configuration from YAML file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        return cls()
    
    @classmethod
    def from_env(cls) -> 'DashboardConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Data settings
        config.batch_size = int(os.getenv('BATCH_SIZE', config.batch_size))
        config.cache_ttl = int(os.getenv('CACHE_TTL', config.cache_ttl))
        config.max_retries = int(os.getenv('MAX_RETRIES', config.max_retries))
        config.retry_delay = float(os.getenv('RETRY_DELAY', config.retry_delay))
        
        # Chart settings
        config.default_chart_height = int(os.getenv('CHART_HEIGHT', config.default_chart_height))
        config.default_chart_width = int(os.getenv('CHART_WIDTH', config.default_chart_width))
        
        # Strategy settings
        config.default_commission = float(os.getenv('COMMISSION', config.default_commission))
        config.default_slippage = float(os.getenv('SLIPPAGE', config.default_slippage))
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = {
            'batch_size': self.batch_size,
            'cache_ttl': self.cache_ttl,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'default_chart_height': self.default_chart_height,
            'default_chart_width': self.default_chart_width,
            'default_commission': self.default_commission,
            'default_slippage': self.default_slippage,
            'rsi_period': self.rsi_period,
            'macd_fast_period': self.macd_fast_period,
            'macd_slow_period': self.macd_slow_period,
            'macd_signal_period': self.macd_signal_period,
            'bollinger_period': self.bollinger_period,
            'bollinger_std_dev': self.bollinger_std_dev,
            'atr_period': self.atr_period,
            'adx_period': self.adx_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'rsi_exit': self.rsi_exit,
            'rsi_stop_loss': self.rsi_stop_loss,
            'rsi_take_profit': self.rsi_take_profit,
            'moving_average_periods': self.moving_average_periods,
            'moving_average_colors': self.moving_average_colors,
            'overview_tickers': self.overview_tickers,
            'sector_etf': self.sector_etf,
            'instrument_types': self.instrument_types,
            'exchanges': self.exchanges,
            'industries': self.industries,
            'quant_figure_indicators': self.quant_figure_indicators,
            'quant_figure_methods': self.quant_figure_methods,
            'app_options': self.app_options,
            'strategy_names': self.strategy_names
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class ConfigManager:
    """Manage configuration from multiple sources"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'config.yaml'
        self.config = self._load_config()
    
    def _load_config(self) -> DashboardConfig:
        """Load configuration with fallback order: file -> env -> defaults"""
        # Try to load from file first
        if os.path.exists(self.config_file):
            try:
                config = DashboardConfig.from_file(self.config_file)
                print(f"✅ Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                print(f"⚠️  Failed to load config file: {e}")
        
        # Fall back to environment variables
        try:
            config = DashboardConfig.from_env()
            print("✅ Loaded configuration from environment variables")
            return config
        except Exception as e:
            print(f"⚠️  Failed to load from environment: {e}")
        
        # Use defaults
        config = DashboardConfig()
        print("✅ Using default configuration")
        return config
    
    def get_config(self) -> DashboardConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"⚠️  Unknown config key: {key}")
    
    def save_config(self, filepath: Optional[str] = None):
        """Save current configuration to file"""
        save_path = filepath or self.config_file
        self.config.save_to_file(save_path)
        print(f"✅ Configuration saved to {save_path}")


# Global configuration instance
config_manager = ConfigManager()
config = config_manager.get_config()
