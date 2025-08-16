#!/usr/bin/env python3
"""
Test script to check trade analyzer structure
"""

import backtrader as bt
import pandas as pd
import yfinance as yf

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.order = None
        
    def next(self):
        if not self.position:
            if self.data.close[0] > self.data.close[-1]:
                self.order = self.buy()
        else:
            if self.data.close[0] < self.data.close[-1]:
                self.order = self.sell()

def test_trade_analyzer():
    # Download some data
    data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
    
    # Create cerebro
    cerebro = backtrader.Cerebro()
    
    # Add data
    data_feed = backtrader.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    # Add strategy
    cerebro.addstrategy(SimpleStrategy)
    
    # Add analyzers
    cerebro.addanalyzer(backtrader.analyzers.TradeAnalyzer, _name='trades')
    
    # Run
    results = cerebro.run()
    strategy = results[0]
    
    # Check trade analyzer
    trades_analyzer = strategy.analyzers.getbyname('trades')
    print(f"Trades analyzer: {trades_analyzer}")
    print(f"Has total attribute: {hasattr(trades_analyzer, 'total')}")
    if hasattr(trades_analyzer, 'total'):
        print(f"Total attribute: {trades_analyzer.total}")
        print(f"Has total.total: {hasattr(trades_analyzer.total, 'total')}")
        if hasattr(trades_analyzer.total, 'total'):
            print(f"Total trades: {trades_analyzer.total.total}")

if __name__ == "__main__":
    test_trade_analyzer()
