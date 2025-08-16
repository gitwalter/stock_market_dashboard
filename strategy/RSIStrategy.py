#!/usr/bin/env python3
"""
RSI Trading Strategy
A strategy based on Relative Strength Index (RSI) signals
"""

import backtrader as bt
import talib


class RSIStrategy(bt.Strategy):
    """
    RSI-based trading strategy
    
    Buy when RSI goes below oversold threshold and starts to rise
    Sell when RSI goes above overbought threshold and starts to fall
    """
    
    params = (
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
        ('rsi_exit', 50),  # Exit when RSI crosses this level
    )
    
    def __init__(self):
        """Initialize the strategy"""
        self.rsi = {}
        self.order = {}
        self.buyprice = {}
        self.buycomm = {}
        
        # Initialize RSI for each data feed
        for i, d in enumerate(self.datas):
            self.rsi[d] = bt.indicators.RSI(d.close, period=self.params.rsi_period)
            self.order[d] = None
            self.buyprice[d] = None
            self.buycomm[d] = None
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
                self.buyprice[order.data] = order.executed.price
                self.buycomm[order.data] = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order[order.data] = None
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Main strategy logic"""
        for d in self.datas:
            # Skip if we have a pending order
            if self.order[d]:
                continue
            
            # Get current position
            position = self.getposition(d).size
            
            # Get current RSI value
            rsi_value = self.rsi[d][0]
            
            # Trading logic
            if not position:  # No position
                # Buy signal: RSI below oversold and starting to rise
                if (rsi_value < self.params.oversold and 
                    self.rsi[d][-1] < rsi_value):  # RSI is rising
                    
                    self.log(f'BUY CREATE, {d.close[0]:.2f}, RSI: {rsi_value:.2f}')
                    self.order[d] = self.buy(data=d)
            
            else:  # Have position
                # Sell signal: RSI above overbought and starting to fall
                if (rsi_value > self.params.overbought and 
                    self.rsi[d][-1] > rsi_value):  # RSI is falling
                    
                    self.log(f'SELL CREATE, {d.close[0]:.2f}, RSI: {rsi_value:.2f}')
                    self.order[d] = self.sell(data=d)
                
                # Exit signal: RSI crosses the exit level
                elif ((self.rsi[d][-1] < self.params.rsi_exit and rsi_value > self.params.rsi_exit) or
                      (self.rsi[d][-1] > self.params.rsi_exit and rsi_value < self.params.rsi_exit)):
                    
                    self.log(f'EXIT CREATE, {d.close[0]:.2f}, RSI: {rsi_value:.2f}')
                    self.order[d] = self.sell(data=d)
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')


class RSIWithStopLoss(bt.Strategy):
    """
    RSI strategy with stop loss and take profit
    """
    
    params = (
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
        ('stop_loss', 0.05),  # 5% stop loss
        ('take_profit', 0.10),  # 10% take profit
    )
    
    def __init__(self):
        """Initialize the strategy"""
        self.rsi = {}
        self.order = {}
        self.buyprice = {}
        self.stop_order = {}
        self.take_profit_order = {}
        
        for i, d in enumerate(self.datas):
            self.rsi[d] = bt.indicators.RSI(d.close, period=self.params.rsi_period)
            self.order[d] = None
            self.buyprice[d] = None
            self.stop_order[d] = None
            self.take_profit_order[d] = None
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
                self.buyprice[order.data] = order.executed.price
                
                # Set stop loss and take profit orders
                stop_price = order.executed.price * (1 - self.params.stop_loss)
                take_profit_price = order.executed.price * (1 + self.params.take_profit)
                
                self.stop_order[order.data] = self.sell(
                    data=order.data, exectype=bt.Order.Stop, price=stop_price
                )
                self.take_profit_order[order.data] = self.sell(
                    data=order.data, exectype=bt.Order.Limit, price=take_profit_price
                )
            
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                
                # Cancel other orders
                if self.stop_order[order.data]:
                    self.cancel(self.stop_order[order.data])
                if self.take_profit_order[order.data]:
                    self.cancel(self.take_profit_order[order.data])
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order[order.data] = None
    
    def next(self):
        """Main strategy logic"""
        for d in self.datas:
            if self.order[d]:
                continue
            
            position = self.getposition(d).size
            rsi_value = self.rsi[d][0]
            
            if not position:  # No position
                # Buy signal
                if (rsi_value < self.params.oversold and 
                    self.rsi[d][-1] < rsi_value):
                    
                    self.log(f'BUY CREATE, {d.close[0]:.2f}, RSI: {rsi_value:.2f}')
                    self.order[d] = self.buy(data=d)
            
            else:  # Have position
                # Sell signal
                if (rsi_value > self.params.overbought and 
                    self.rsi[d][-1] > rsi_value):
                    
                    self.log(f'SELL CREATE, {d.close[0]:.2f}, RSI: {rsi_value:.2f}')
                    self.order[d] = self.sell(data=d)
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
