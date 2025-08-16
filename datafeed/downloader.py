import pandas as pd
import yfinance as yf
import numpy as np

class InfoDownloader:
    def __init__(self,ticker_name):
        self.ticker = yf.Ticker(ticker_name)

    def info(self):
        info_dict = self.ticker.info
        if isinstance(info_dict, dict):
            return pd.DataFrame([info_dict])
        else:
            return pd.DataFrame(info_dict())
    
    def balance_sheet(self):
        return pd.DataFrame(self.ticker.balance_sheet())

    def fast_info(self):
        # fast_info method doesn't exist in yfinance, let's use basic info instead
        try:
            info_dict = self.ticker.info
            if isinstance(info_dict, dict):
                return pd.DataFrame([info_dict])
            else:
                return pd.DataFrame(info_dict())
        except Exception as e:
            print(f"Error getting fast info: {e}")
            return pd.DataFrame()

    def get_news(self):
        return pd.DataFrame(self.ticker.get_news())


class BatchPriceDownloader:
    batch_size = 20
    loop_size = 0
    fields = ['Open', 'Low', 'High', 'Close', 'Volume']
    def __init__(self, ticker_list, start, end, interval):
        self.ticker_list = ticker_list
        self.loop_size = int(len(self.ticker_list) // self.batch_size) + 2
        # Convert string dates to datetime if needed
        if isinstance(start, str):
            self.start = pd.to_datetime(start)
        else:
            self.start = start
        if isinstance(end, str):
            self.end = pd.to_datetime(end)
        else:
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
                    # Handle single ticker case properly
                    if isinstance(batch_download.columns, pd.MultiIndex):
                        self.collected_prices = batch_download
                    else:
                        # Create proper MultiIndex for single ticker
                        single_ticker = batch_list[0]
                        multi_cols = pd.MultiIndex.from_product([self.fields, [single_ticker]], names=["prices", "symbol"])
                        self.collected_prices = pd.DataFrame(batch_download.values, 
                                                           index=batch_download.index, 
                                                           columns=multi_cols)
            else:
                self.collected_prices.update(batch_download)
                    
        return self.collected_prices
        