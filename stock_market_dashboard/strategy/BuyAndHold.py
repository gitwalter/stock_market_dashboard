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
            pos = self.getposition(d).size
            if not pos and not self.order[d]:
                close = d.close[0]
                # Buy all the available cash
                equities = len(self.datas)
                # size = int((self.broker.get_cash() *0.8 / self.data) / equities )
                size = int(((self.broker.get_cash() * 0.95 / close)) / equities)
                self.order[d] = self.buy(data=d, size=size)

    def stop(self):
        # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))
        print('Final Value:        {:.2f} USD'.format(self.broker.get_value()))

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}') #Print date and close
        
    def notify_order(self, order):
        dt, dn = self.datetime.date(), order.data._name
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
          
        
        for d in self.datas:
            if self.order[d] == None:
                 continue
             
            self.order[d] = None