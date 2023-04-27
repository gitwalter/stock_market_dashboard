import pandas as pd
import yfinance as yf
import numpy as np

class InfoDownloader:
    def __init__(self,ticker_name):
        self.ticker = yf.Ticker(ticker_name)

    def info(self):
        return pd.DataFrame(self.ticker.info())
    
    def balance_sheet(self):
        return pd.DataFrame(self.ticker.balance_sheet())

    def fast_info(self):
        return pd.DataFrame(self.ticker.fast_info())

    def get_news(self):
        return pd.DataFrame(self.ticker.get_news())


class BatchPriceDownloader:
    batch_size = 20
    loop_size = 0
    fields = ['Open', 'Low', 'High', 'Close', 'Volume']
    def __init__(self, ticker_list, start, end, interval):
        self.ticker_list = ticker_list
        self.loop_size = int(len(self.ticker_list) // self.batch_size) + 2
        self.start = start
        self.end = end
        self.interval = interval
        iterables = [self.fields, self.ticker_list]
        price_columns = pd.MultiIndex.from_product(iterables, names=["prices", "symbol"])
        self.collected_prices = pd.DataFrame(columns=price_columns, index=pd.to_datetime([]))   
        
    def get_yahoo_prices(self):        
        for t in range(1,self.loop_size): # Batch download
            m = (t - 1) * self.batch_size
            n = t * self.batch_size
            batch_list = self.ticker_list[m:n]
            if len(batch_list) == 0:
                break
            
            print(batch_list,m,n)
            batch_download = yf.download(tickers= batch_list,start=self.start, end=self.end, 
                                interval=self.interval,group_by='column',auto_adjust=True, 
                                      prepost=True, threads=True, proxy=None)[self.fields] 

            if len(batch_list) > 1:
                batch_download.columns = batch_download.columns.rename("prices", level=0)
                batch_download.columns = batch_download.columns.rename("symbol", level=1)                       
            
            if self.collected_prices.empty:
                if len(batch_list) > 1:
                    self.collected_prices = pd.concat([self.collected_prices, batch_download], ignore_index=False)
                else:
                    self.collected_prices = batch_download
            else:
                self.collected_prices.update(batch_download)
                    
        return self.collected_prices
        