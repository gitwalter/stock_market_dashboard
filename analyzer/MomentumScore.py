import numpy as np
import pandas as pd
from scipy import stats 
from scipy.stats import percentileofscore as score

class MomentumScore:
    def __init__(self, vola_window = 20):        
        self.vola_window = vola_window

    def get_score(self,ts):
        """
        Input:  Price time series.
        Output: Annualized exponential regression slope, 
                multiplied by the R2
        """
        # Make a list of consecutive numbers
        x = np.arange(len(ts)) 
        # Get logs
        ts = np.float64(ts)
        log_ts = np.log(ts) 
        # Calculate regression values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
        # Annualize percent
        annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
        #Adjust for fitness
        score = annualized_slope * (r_value ** 2)
        return score
    
    def get_intraday_momentum(self,ts):
        dc = []
        for i in ts.columns:
            dc.append(ts[i].pct_change().sum())

        intraday_momentum = pd.DataFrame(columns = ['symbol', 'day_change'])
        intraday_momentum['symbol'] = ts.columns
        intraday_momentum['day_change'] = dc

            # CALCULATING MOMENTUM

        intraday_momentum['momentum'] = 'N/A'
        for i in range(len(intraday_momentum)):
            intraday_momentum.loc[i, 'momentum'] = score(intraday_momentum.day_change, intraday_momentum.loc[i, 'day_change'])/100
    
        intraday_momentum['momentum'] = intraday_momentum['momentum'].astype(float)
        return intraday_momentum

    def get_volatility(self,ts):
        # ts = np.float64(ts)
        return ts.pct_change().rolling(self.vola_window).std().iloc[-1]