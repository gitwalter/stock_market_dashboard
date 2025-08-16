#!/usr/bin/env python3
"""
Strategy package for Stock Market Dashboard
Contains all trading strategies for backtesting
"""

from .BuyAndHold import BuyAndHold
from .RSIStrategy import RSIStrategy
from .MinerviniMomentum import MinerviniMomentum
from .SmaCross import SmaCross
from .TrailingStopLoss import TrailingStopLoss

__all__ = [
    'BuyAndHold',
    'RSIStrategy', 
    'MinerviniMomentum',
    'SmaCross',
    'TrailingStopLoss'
]
