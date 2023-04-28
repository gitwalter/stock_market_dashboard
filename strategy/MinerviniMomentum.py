import backtrader as bt

class MinerviniMomentum(bt.Strategy):
    params = (
        ('sma50', 50),
        ('sma150', 150),
        ('sma200', 200),
        ('high_low_period', 260),
        ('oneplot', True)
    )
    

    def __init__(self):		
        '''
        Create an dictionary of indicators so that we can dynamically add the
        indicators to the strategy using a loop. This mean the strategy will
        work with any numner of data feeds. 
        '''
        self.inds = dict()
        self.order = dict()  
        
        for i, d in enumerate(self.datas):
            self.inds[d] = dict()
            self.inds[d]['sma50'] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma50)
            self.inds[d]['sma150'] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma150)
            self.inds[d]['sma200'] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.sma200)
            self.inds[d]['cross'] = bt.indicators.CrossOver(self.inds[d]['sma50'],self.inds[d]['sma150'])
            self.inds[d]['momentum'] = bt.indicators.Momentum(d.close)
            self.inds[d]['above_sma3'] = d.close > self.inds[d]['sma200']
            self.order[d] = None
            
            self.inds[d]['low_of_52week'] = bt.indicators.Lowest(d.close, period=self.params.high_low_period)
            self.inds[d]['high_of_52week'] = bt.indicators.Highest(d.close, period=self.params.high_low_period)
            if i > 0: #Check we are not on the first loop of data feed:
                if self.p.oneplot == True:
                    d.plotinfo.plotmaster = self.datas[0]

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}') #Print date and close
    
    def next(self):
               
        for d in self.datas:        
           close = d.close[0]
           sma50 = self.inds[d]['sma50'][0]
           sma150 = self.inds[d]['sma150'][0]
           sma200 = self.inds[d]['sma200'][0]
           low_of_52week = self.inds[d]['low_of_52week'][0]
           high_of_52week = self.inds[d]['high_of_52week'][0]
                     
           
           try:
               sma200_20 = self.inds[d]['sma200'][-20]
           except Exception:
                sma200_20 = 0
           
           # Condition 1: Current Price > 150 SMA and > 200 SMA
           if close > sma150 > sma200:
               condition_1 = True
           else:
               condition_1 = False
          
          
           # Condition 2: 150 SMA and > 200 SMA
           if sma150 > sma200:
                condition_2 = True
           else:
                condition_2 = False                
            

           # Condition 3: 200 SMA trending up for at least 1 month
           if sma200 > sma200_20:
               condition_3 = True
           else:
               condition_3 = False
          
           # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
           if sma50 > sma150 and sma50 > sma200:
               condition_4 = True
           else:
               condition_4 = False
             
           # Condition 5: Current Price > 50 SMA
           if close > sma50:
                condition_5 = True
           else:
               condition_5 = False
          
           # Condition 6: Current Price is at least 30% above 52 week low
           if close >= (1.3*low_of_52week):
               condition_6 = True
           else:
               condition_6 = False
            
           # Condition 7: Current Price is within 25% of 52 week high
           if close >= (0.75*high_of_52week):
               condition_7 = True
           else:
               condition_7 = False
           
            

           if condition_1 and condition_2 and condition_3 and \
              condition_4 and condition_5 and condition_6 and condition_7:
               pos = self.getposition(d).size
        
               if not pos and not self.order[d]:
                   equities = len(self.datas)
                   size = int(((self.broker.get_cash() / close)) / equities) - 100
                   if size > 0:
                       self.order[d] = self.buy(data=d, size=size)
                       
               
               
               
           else:
               position_size = self.getposition(d).size
               if position_size and not condition_3:
                   # sell is done by trailing stop
                   # if sma200 crosses below sma200 20 days ago
                    self.order[d] = self.close(data=d, size=position_size)
               
           
    def notify_order(self, order):
        dt, dn = self.datetime.date(), order.data._name
        print('{} {} Order {} Status {}'.format(
            dt, dn, order.ref, order.getstatusname())
        )     
            
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
          self.log('Order Canceled/Margin/Rejected')
        
       
        
        for d in self.datas:
            if self.order[d] == None:
                 continue
             
            self.order[d] = None