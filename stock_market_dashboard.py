from datetime import timedelta
from datetime import datetime
import os

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pandas as pd

import yfinance as yf
from pytickersymbols import PyTickerSymbols
import cufflinks as cf

import backtrader

from datafeed.downloader import BatchPriceDownloader
from datafeed.downloader import InfoDownloader
from datafeed.etoro import SymbolScraper


from analyzer.MomentumScore import MomentumScore

def start():
    """Set directory and create application"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = StockMarketDashboard()
    return app


class StockMarketDashboard:
    """Stock market dashboard with quant figure charting and backtest"""
    ONE_DAY = '1d'
    ONE_MINUTE = '1m'

    @st.cache_data()
    def load_etoro_file(_self):
        """Load file containing eToro instruments"""
        path = "etoro.csv"
        try:
            symbols = pd.read_csv(path, delimiter=";", index_col="symbol")
        except:
            symbols = SymbolScraper().get()
            symbols.to_csv("etoro.csv", index=False, sep=";")

        return symbols

    @st.cache_data
    def download_instrument_price(_self, tickers_list, start_date, end_date, interval):
        """Download instrument price from yahoo"""
        downloader = BatchPriceDownloader(
            tickers_list, start_date, end_date, interval)
        prices = downloader.get_yahoo_prices()
        return prices

    def filter_symbols(self):
        """Filter symbols by type, exchange and industry"""
        symbols = self.load_etoro_file()

        self.selected_types = st.sidebar.multiselect(
            "Type", self.types, default=["Stocks"])
        self.selected_exchanges = st.sidebar.multiselect(
            "Exchange", self.exchanges, default=["Frankfurt"])
        self.selected_industries = st.sidebar.multiselect(
            "Industry", self.industries)

        if len(self.selected_exchanges) != 0:
            symbols = symbols[symbols["exchange"].isin(
                self.selected_exchanges)]

        if len(self.selected_types) != 0:
            symbols = symbols[symbols["instrument type"].isin(
                self.selected_types)]

        if len(self.selected_industries) != 0:
            symbols = symbols[symbols["industry"].isin(
                self.selected_industries)]

        symbols = symbols.sort_values(by='name', ascending=True)

        return symbols

    def get_symbol(self):
        """Get selected symbol"""
        
        self.symbols = self.filter_symbols()

        self.symbol_name = st.selectbox("Select Instrument", self.symbols.name)
        self.symbol = self.symbols.loc[self.symbols['name']
                                       == self.symbol_name].index[0]

    def get_symbols_from_multiselect(self):
        """Get symbols"""
        self.symbols = self.filter_symbols()
        self.selected_symbol_names = st.sidebar.multiselect(
            "Select Instruments", self.symbols.name)


        if self.selected_symbol_names:
            # set selected symbols filtered by name
            self.selected_symbols = self.symbols.loc[self.symbols['name'].isin(
                self.selected_symbol_names)]
        else:
            # no name selected take symbols filtered by type, exchange, industry
            self.selected_symbols = self.symbols

    def handle_option_backtest(self):
        """Execute backtest for selected instrument and strategy"""
        self.get_symbol()
        strategy_name = st.selectbox('Strategy', self.strategies)
        execute = st.button(label="Execute")
        self.get_start_end()
        if execute:
            cerebro = backtrader.Cerebro()  # create a "Cerebro" engine instance
            # Create a data feed
            data = backtrader.feeds.PandasData(dataname=yf.download(
                self.symbol, self.start_date, self.end_date, auto_adjust=True))
            if data._dataname.empty:
                st.write('No pricedata for ' + self.symbol)
            else:
                cerebro.adddata(data)  # Add the data feed
                # dynamic create instance of strategy class
                strategy = globals()[strategy_name]

                cerebro.addstrategy(strategy)  # Add the trading strategy
                cerebro.addsizer(backtrader.sizers.SizerFix, stake=100)
                cerebro.run()  # run it all
                matplotlib.use('Agg')
                st.pyplot(cerebro.plot(iplot=True)[0][0])

    def get_start_end(self):
        """Get start and end date"""
        default_start_date, default_end_date = self.get_default_start_end()
        self.start_date = st.sidebar.date_input(
            'Start date', default_start_date)
        self.end_date = st.sidebar.date_input('End date', default_end_date)

    def get_default_start_end(self):
        """Calculate default start and end date"""
        today = datetime.now().date()
        default_start_date = today - timedelta(days=731)
        return default_start_date, today

    def handle_option_overview(self):
        """Display overview"""
        self.get_chart_indicators()
        self.get_start_end()

        for ticker in self.overview_tickers:
            self.symbol = ticker
            self.symbol_name = ticker
            self.quant_figure_minutes()
            self.quant_figure_days()

    def handle_option_watchlist(self):
        """Plot quant figure for instruments of csv file"""
    
        stocks = self.get_selected_stocks()
        self.get_chart_indicators()
        self.get_start_end()

        show_minutes = st.sidebar.checkbox('Show Minutes')
        show_days = st.sidebar.checkbox('Show Days')

        uploaded_file = st.file_uploader(
            "Choose a csv-file with columns '''Symbol''' and '''Name''' for stock symbols and names")
        if uploaded_file is not None:
            tickers = pd.read_csv(uploaded_file)
            st.write(tickers)
            for ticker in tickers.iterrows():
                self.symbol = ticker[1]['Symbol']
                self.symbol_name = ticker[1]['Name']
                if show_minutes:
                    self.quant_figure_minutes()
                if show_days:
                    self.quant_figure_days()

        if st.sidebar.button('Display Charts') and uploaded_file is None:
            for ticker in stocks:
                self.symbol = ticker
                self.symbol_name = ticker
                if show_minutes:
                    self.quant_figure_minutes()
                if show_days:
                    self.quant_figure_days()

    def plot_quant_figure(self, prices, name, title):
        """Plot quant figure of instrument"""
        # nothing to plot
        if prices.empty:
            return

        st.write(title)
        # Create candlestick chart
        qf = cf.QuantFig(prices, legend='bottom', name=name, title=title)
        counter = 0
        for check in self.exponential_moving_averages['checked']:
            if check:
                qf.add_ema(periods=self.exponential_moving_averages['values'][counter],
                           color=self.exponential_moving_averages['colors'][counter])
            counter += 1

        counter = 0
        for check in self.simple_moving_averages['checked']:
            if check:
                qf.add_sma(
                    periods=self.simple_moving_averages['values'][counter], color=self.simple_moving_averages['colors'][counter])
            counter += 1

        counter = 0
        for check in self.quant_figure_indicators['checked']:
            if check:
                quantfigure_method = self.quant_figure_indicators['methods'][counter]
                getattr(qf, quantfigure_method)()
            counter += 1

        fig = qf.iplot(asFigure=True, dimensions=(800, 600))

        st.plotly_chart(fig)

    def handle_option_chart(self):
        """Display chart of instrument"""
        self.get_symbol()
        # Set start and end point to fetch data
        self.get_start_end()

        self.get_chart_indicators()

        if st.checkbox('Display News'):
            info_downloader = InfoDownloader(self.symbol_name)
            ticker_news = info_downloader.get_news()
            if not ticker_news.empty:
                st.write(ticker_news[['title', 'link']])

        st.write(self.symbols.loc[self.symbols['name'] == self.symbol_name])

        browse_charts = st.sidebar.checkbox('Browse Charts')

        display_quantfigure = False

        if not browse_charts:
            display_quantfigure = st.sidebar.button('Show Charts')

        if display_quantfigure or browse_charts:
            self.quant_figure_minutes()
            self.quant_figure_days()

    def handle_option_returns(self):
        """Calculate return per period for selected stocks"""
        self.get_symbols_from_multiselect()
        self.get_start_end()
        calculate_returns = st.sidebar.button('Calculate Returns')
        if not self.selected_symbols.empty and calculate_returns:
            prices = self.download_instrument_price(self.selected_symbols.index.tolist(
            ), self.start_date, self.end_date, self.ONE_DAY)
            # download = BatchPriceDownloader(self.selected_symbols.index.tolist(
            # ), self.start_date, self.end_date, self.ONE_DAY)
            # prices = download.get_yahoo_prices()

            st.header('Daily Returns')
            daily_returns = prices['Close'].pct_change()

            # Find the last date in the DataFrame
            last_date = daily_returns.index[-1]

            # Select rows corresponding to the last date
            last_date_returns = daily_returns.loc[last_date]

            # Transpose the DataFrame to have ticker names as the index
            daily_stock_returns = last_date_returns.transpose()

            # Rename the index to 'Ticker'
            daily_stock_returns.index.name = 'Ticker'
            
            daily_stock_returns = daily_stock_returns.to_frame().merge(self.selected_symbols[['name', 'industry']], left_index=True, right_index=True)


            st.dataframe(daily_stock_returns)

            st.header('Weekly Returns')
            weekly_returns = prices['Close'].resample('W').ffill().pct_change()
            st.dataframe(weekly_returns)
            last_weekly_returns = weekly_returns.loc[weekly_returns.index[-1]]
            last_weekly_stock_returns = last_weekly_returns.transpose()
            last_weekly_stock_returns.index.name = 'Ticker'
            last_weekly_stock_returns = last_weekly_stock_returns.to_frame().merge(self.selected_symbols[['name', 'industry']], left_index=True, right_index=True)
            
            st.dataframe(last_weekly_stock_returns)
            if len(self.selected_symbols) < 11:
                fig, ax = plt.subplots()
                if len(weekly_returns.columns) >= 2:
                    for index, row in self.selected_symbols.iterrows():
                        (weekly_returns[index] + 1).cumprod().plot()
                        ax.plot(label=index)
                        ax.legend()
                else:
                    (weekly_returns + 1).cumprod().plot()

                st.pyplot(fig)

            st.header('Monthly Returns')
            monthly_returns = prices['Close'].resample(
                'M').ffill().pct_change()
            st.dataframe(monthly_returns)
            last_monthly_returns = monthly_returns.loc[monthly_returns.index[-1]]
            last_monthly_stock_returns = last_monthly_returns.transpose()
            last_monthly_stock_returns.index.name = 'Ticker'
            last_monthly_stock_returns = last_monthly_stock_returns.to_frame().merge(self.selected_symbols[['name', 'industry']], left_index=True, right_index=True)
            st.dataframe(last_monthly_stock_returns)


            st.header('Yearly Returns')
            yearly_returns = prices['Close'].resample('Y').ffill().pct_change()
            st.dataframe(yearly_returns)
            last_yearly_returns = yearly_returns.loc[yearly_returns.index[-1]]
            last_yearly_stock_returns = last_yearly_returns.transpose()
            last_yearly_stock_returns.index.name = 'Ticker'
            last_yearly_stock_returns = last_yearly_stock_returns.to_frame().merge(self.selected_symbols[['name', 'industry']], left_index=True, right_index=True)
            st.dataframe(last_yearly_stock_returns)

            before_last_yearly_returns = yearly_returns.loc[yearly_returns.index[-2]]
            before_last_yearly_stock_returns = before_last_yearly_returns.transpose()
            before_last_yearly_stock_returns.index.name = 'Ticker'
            before_last_yearly_stock_returns = before_last_yearly_stock_returns.to_frame().merge(self.selected_symbols[['name', 'industry']], left_index=True, right_index=True)
            st.dataframe(before_last_yearly_stock_returns)


    def handle_option_momentum(self):
        """Run momentum calculation for selected stocks"""
        stocks = self.get_selected_stocks()

        self.get_start_end()

        if st.sidebar.button("Calculate Momentum"):
            momentum_score = MomentumScore()
            last_trading_day, collected_prices = self.get_prices_last_trading_day(
                stocks)
            collected_prices = collected_prices['Close']
            intraday_momentum = momentum_score.get_intraday_momentum(
                collected_prices)

            st.write(last_trading_day)
            st.dataframe(intraday_momentum)

            collected_prices = self.download_instrument_price(
                stocks, self.start_date, self.end_date, self.ONE_DAY)['Close']
            collected_prices = collected_prices.dropna(axis='columns')

            columns = ['Score',
                       'Volatility']

            analysis_collected = pd.DataFrame(columns=columns)

            analysis_collected.index.name = 'Instrument'

            for ticker in collected_prices:
                ticker_price = collected_prices.filter(like=ticker)

                analysis_ticker = {'Score': momentum_score.get_score(ticker_price[ticker]),
                                   'Volatility': momentum_score.get_volatility(ticker_price[ticker])}
                analysis_df = pd.DataFrame(analysis_ticker, index=[ticker])
                analysis_df.index.name = 'Instrument'
                analysis_collected = pd.concat(
                    [analysis_collected, analysis_df], axis=0)

            st.dataframe(analysis_collected)

    def get_selected_stocks(self):
        """Determine selected stocks depending on market and user input"""
        selected_market = st.sidebar.selectbox(
            options=["SP500", "NASDAQ100", "DJ30", "DAX", "eToro"], label="Market")

        stock_data = PyTickerSymbols()

        if selected_market == 'DJ30':
            stocks = stock_data.get_dow_jones_nyc_yahoo_tickers()
        if selected_market == 'DAX':
            stocks = stock_data.get_dax_frankfurt_yahoo_tickers()
        if selected_market == 'SP500':
            stocks = stock_data.get_sp_500_nyc_yahoo_tickers()
        if selected_market == 'NASDAQ100':
            stocks = stock_data.get_nasdaq_100_nyc_yahoo_tickers()

        if selected_market == 'eToro':
            # self.selected_symbols = self.filter_symbols()
            self.get_symbols_from_multiselect()
            stocks = self.selected_symbols.index.tolist()

        return stocks

    def handle_option_sectors(self):
        """Display sector ETF charts"""
        self.get_chart_indicators()
        self.get_start_end()

        for sector in self.sector_etf:
            self.symbol = self.sector_etf[sector]
            self.symbol_name = sector + ' ' + self.sector_etf[sector]
            self.quant_figure_minutes()
            self.quant_figure_days()

    def quant_figure_days(self):
        """Display quant figure for a period of days"""
        df_ticker_period = self.download_instrument_price(
            [self.symbol], self.start_date, self.end_date, self.ONE_DAY)
        if df_ticker_period.empty:
            st.error('Load of daily prices for ' +
                     self.symbol_name + ' failed', icon='🚨')
        else:
            title = self.symbol_name + ' ' + \
                str(self.start_date) + ' - ' + \
                str(self.end_date) + ' Interval ' + self.ONE_DAY
            self.plot_quant_figure(df_ticker_period, self.symbol, title=title)

    def quant_figure_minutes(self):
        """Display quant figure for last trading day"""
        symbols = []
        symbols.append(self.symbol)
        last_trading_day, df_ticker_last_trading_day = self.get_prices_last_trading_day(
            symbols)

        if df_ticker_last_trading_day.empty:
            st.error('Load of minute prices for ' +
                     self.symbol_name + ' failed', icon='🚨')
        else:
            title = self.symbol_name + ' ' + \
                str(last_trading_day) + ' Interval ' + self.ONE_MINUTE
            self.plot_quant_figure(
                df_ticker_last_trading_day, self.symbol, title=title)

    def get_prices_last_trading_day(self, symbols):
        """Get last daily price data"""
        last_trading_day = datetime.now().date()
        df_ticker_last_trading_day = self.download_instrument_price(
            symbols, last_trading_day, last_trading_day, self.ONE_MINUTE)
        if df_ticker_last_trading_day.empty:
            US_BUSINESS_DAY = CustomBusinessDay(
                calendar=USFederalHolidayCalendar())
            trials = 0
            while trials < 3:
                last_trading_day = (datetime.now().date() -
                                    trials * US_BUSINESS_DAY).date()
                df_ticker_last_trading_day = self.download_instrument_price(
                    symbols, last_trading_day, None, self.ONE_MINUTE)
                if not df_ticker_last_trading_day.empty:
                    break
                trials += 1
        return last_trading_day, df_ticker_last_trading_day

    def get_chart_indicators(self):
        """Get chart indicators EMA, BB, ATR, MACD, RSI, VOL"""
        self.display_checkboxes_moving_averages(
            self.exponential_moving_averages)
        self.display_checkboxes_quant_figure_indicators()

    def display_checkboxes_moving_averages(self, moving_average):
        """Display checkboxes for configured moving averages"""
        columns = st.sidebar.columns(len(moving_average['values']))

        counter = 0
        for column in columns:
            with column:
                label = moving_average['name'] + \
                    str(moving_average['values'][counter])
                moving_average['checked'][counter] = st.checkbox(
                    label, value=moving_average['checked'][counter])
                counter += 1

    def display_checkboxes_quant_figure_indicators(self):
        columns = st.sidebar.columns(
            len(self.quant_figure_indicators['names']))
        counter = 0
        for column in columns:
            with column:
                label = self.quant_figure_indicators['names'][counter]
                self.quant_figure_indicators['checked'][counter] = st.checkbox(
                    label, value=self.quant_figure_indicators['checked'][counter])
                counter += 1

    def __init__(self):
        """Initialize members"""
        self.options = ["Overview", "Sectors", "Chart",
                        "Watchlist", "Momentum", "Returns", "Backtest"]
        self.types = ["Stocks", "Currencies",
                      "Commodities", "Cryptocurrencies", "ETF"]
        self.exchanges = ["NASDAQ", "NYSE", "London", "Frankfurt", "Paris", "Amsterdam",
                          "BorsaItaliana", "BolsaDeMadrid", "Oslo", "Zurich", "HongKong", "Helsinki", "Copenhagen", "Stockholm"]
        self.industries = ["-", "BasicMaterials", "ConsumerGoods", "Technology",
                           "Services", "Financial", "IndustrialGoods", "Healthcare"
                           "Conglomerates", "Utilities"]

        self.sector_etf = {'Technology': 'XLK', 'Healthcare': 'XLV', 'Finance': 'XLF', 'Real Estate': 'XLRE', 'Energy': 'XLE', 'Materials': 'XLB',
                           'Consumer Discretionary': 'XLY', 'Industrials': 'XLI', 'Utilities': 'XLU', 'Consumer Staples': 'XLP', 'Telecommunication': 'XLC'}

        self.strategies = ["BuyAndHold", "MinerviniMomentum",
                           "TrailingStopLoss", "SmaCross"]
        self.overview_tickers = [
            '^DJI', '^GSPC', '^NDX', '^IXIC', '^GDAXI', '^HSI', 'EURUSD=X', 'GC=F', 'BTC-USD']

        moving_average_colors = ['green', 'blue', 'orange', 'purple', 'red']
        moving_average_values = [20, 50, 100, 150, 200]
        emas_checked = [False for i in range(len(moving_average_values))]
        smas_checked = [False for i in range(len(moving_average_values))]
        self.exponential_moving_averages = {
            "name": "EMA", "values": moving_average_values, "checked": emas_checked, "colors": moving_average_colors}
        self.simple_moving_averages = {
            "name": "SMA", "values": moving_average_values, "checked": smas_checked, "colors": moving_average_colors}

        quant_figure_indicators = ['atr', 'bb', 'macd', 'rsi', 'vol']
        quant_figure_methods = [
            'add_atr', 'add_bollinger_bands', 'add_macd', 'add_rsi', 'add_volume']
        quant_figure_indicators_checked = [
            False for i in range(len(quant_figure_indicators))]
        self.quant_figure_indicators = {"names": quant_figure_indicators,
                                        "methods": quant_figure_methods, "checked": quant_figure_indicators_checked}

        self.symbol = None
        self.symbol_name = None
        self.option = None

    def run(self):
        """Handle different options of app"""
        self.option = st.sidebar.selectbox('Options', self.options)
        if self.option == 'Backtest':
            self.handle_option_backtest()
        if self.option == 'Chart':
            self.handle_option_chart()
        if self.option == 'Overview':
            self.handle_option_overview()
        if self.option == 'Watchlist':
            self.handle_option_watchlist()
        if self.option == 'Returns':
            self.handle_option_returns()
        if self.option == 'Momentum':
            self.handle_option_momentum()
        if self.option == 'Sectors':
            self.handle_option_sectors()


application = start()
application.run()
