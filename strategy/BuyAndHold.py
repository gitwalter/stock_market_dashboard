import backtrader as bt

class BuyAndHold(bt.Strategy):
    
    def __init__(self):
        self.order = dict()
        for i, d in enumerate(self.datas):
            self.order[d] = None
    
    def start(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash

    def next(self):
        for d in self.datas:
            # Skip if not enough data
            if len(d) < 1:
                continue
                
            pos = self.getposition(d).size
            
            # Buy logic - only buy if we don't have a position and no pending order
            if not pos and not self.order[d]:
                close = d.close[0]
                
                # Skip if close price is not valid
                if not close or close <= 0:
                    continue
                    
                equities = len(self.datas)            
                # buy with 95% of cash
                size = int(((self.broker.get_cash() * 0.95 / close)) / equities)
                if size > 0:
                    self.order[d] = self.buy(data=d, size=size)
            
            # Sell logic - close position at the end of the period
            elif pos > 0 and not self.order[d]:
                # Check if this is the last bar
                if len(d) == d.buflen():
                    self.order[d] = self.sell(data=d, size=pos)
                    self.log(f'SELL CREATE, {getattr(d, "_name", "Unknown")}, {d.close[0]:.2f}')

    def stop(self):
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))
        print('Final Value:        {:.2f} USD'.format(self.broker.get_value()))

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}') #Print date and close
        
    def notify_order(self, order):
        dt, dn = self.datetime.date(), getattr(order.data, '_name', 'Unknown')
        print('{} {} Order {} Status {}'.format(
            dt, dn, order.ref, order.getstatusname())
        )
        
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {dn}, {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, {dn}, {order.executed.price:.2f}')
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        # Reset order for this data
        for d in self.datas:
            if self.order[d] == order:
                self.order[d] = None