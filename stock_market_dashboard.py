#!/usr/bin/env python3
"""
Enhanced Stock Market Dashboard
A comprehensive financial analysis and trading strategy backtesting application
"""

from datetime import timedelta, datetime
import os
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pandas as pd
import yfinance as yf
from pytickersymbols import PyTickerSymbols
import cufflinks as cf
import backtrader

# Import our enhanced modules
from config import config
from utils.exceptions import DashboardError, DataDownloadError, ValidationError
from utils.validation import validate_ticker_symbol, sanitize_user_input
from utils.logging import logger, monitor_performance
from datafeed.downloader import BatchPriceDownloader, InfoDownloader
from datafeed.etoro import SymbolScraper
from analyzer.MomentumScore import MomentumScore


def start():
    """Set directory and create application"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = StockMarketDashboard()
    return app


class StockMarketDashboard:
    """Enhanced stock market dashboard with configuration management and error handling"""
    
    def __init__(self):
        """Initialize dashboard with configuration"""
        # Use configuration for constants
        self.ONE_DAY = '1d'
        self.ONE_MINUTE = '1m'
        
        # Load configuration values
        self.overview_tickers = config.overview_tickers
        self.sector_etf = config.sector_etf
        self.types = config.instrument_types
        self.exchanges = config.exchanges
        self.industries = config.industries
        self.strategies = config.strategy_names
        self.app_options = config.app_options
        
        # Initialize components
        self.momentum = MomentumScore()
        
        # Set up Streamlit page
        st.set_page_config(
            page_title="Stock Market Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    @st.cache_data(ttl=config.cache_ttl)
    def load_etoro_file(_self):
        """Load file containing eToro instruments with error handling"""
        path = "etoro.csv"
        try:
            symbols = pd.read_csv(path, delimiter=";", index_col="symbol")
            logger.log_data_download(["etoro_file"], True, 0)
            return symbols
        except Exception as e:
            logger.log_error("etoro_file_load", f"Failed to load etoro.csv: {e}")
            try:
                symbols = SymbolScraper().get()
                symbols.to_csv("etoro.csv", index=False, sep=";")
                logger.log_data_download(["etoro_scrape"], True, 0)
                return symbols
            except Exception as scrape_error:
                logger.log_error("etoro_scrape", f"Failed to scrape symbols: {scrape_error}")
                st.error("Failed to load instrument data")
                return pd.DataFrame()

    @st.cache_data(ttl=config.cache_ttl)
    def download_instrument_price(_self, tickers_list, start_date, end_date, interval):
        """Download instrument price from yahoo with error handling"""
        try:
            downloader = BatchPriceDownloader(tickers_list, start_date, end_date, interval)
            prices = downloader.get_yahoo_prices()
            return prices
        except DataDownloadError as e:
            logger.log_error("price_download", str(e))
            st.error(f"Failed to download price data: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.log_error("price_download", f"Unexpected error: {e}")
            st.error(f"Unexpected error downloading data: {e}")
            return pd.DataFrame()

    def filter_symbols(self):
        """Filter symbols by type, exchange and industry with validation"""
        try:
            symbols = self.load_etoro_file()
            
            if symbols.empty:
                st.warning("No symbols available")
                return pd.DataFrame()

            self.selected_types = st.sidebar.multiselect(
                "Type", self.types, default=["Stocks"])
            self.selected_exchanges = st.sidebar.multiselect(
                "Exchange", self.exchanges, default=["Frankfurt"])
            self.selected_industries = st.sidebar.multiselect(
                "Industry", self.industries)

            # Apply filters
            if len(self.selected_exchanges) != 0:
                symbols = symbols[symbols["exchange"].isin(self.selected_exchanges)]

            if len(self.selected_types) != 0:
                symbols = symbols[symbols["instrument type"].isin(self.selected_types)]

            if len(self.selected_industries) != 0:
                symbols = symbols[symbols["industry"].isin(self.selected_industries)]

            symbols = symbols.sort_values(by='name', ascending=True)
            return symbols
            
        except Exception as e:
            logger.log_error("symbol_filter", f"Failed to filter symbols: {e}")
            st.error("Failed to filter symbols")
            return pd.DataFrame()

    def get_symbol(self):
        """Get selected symbol with validation"""
        try:
            self.symbols = self.filter_symbols()
            
            if self.symbols.empty:
                st.warning("No symbols available")
                return

            self.symbol_name = st.selectbox("Select Instrument", self.symbols.name)
            self.symbol = self.symbols.loc[self.symbols['name'] == self.symbol_name].index[0]
            
            # Validate symbol
            if not validate_ticker_symbol(self.symbol):
                st.error(f"Invalid symbol format: {self.symbol}")
                return
                
        except Exception as e:
            logger.log_error("symbol_selection", f"Failed to get symbol: {e}")
            st.error("Failed to get symbol")

    def get_symbols_from_multiselect(self):
        """Get symbols from multiselect with validation"""
        try:
            self.symbols = self.filter_symbols()
            
            if self.symbols.empty:
                st.warning("No symbols available")
                return

            self.selected_symbol_names = st.sidebar.multiselect(
                "Select Instruments", self.symbols.name)

            if self.selected_symbol_names:
                # Set selected symbols filtered by name
                self.selected_symbols = self.symbols.loc[
                    self.symbols['name'].isin(self.selected_symbol_names)
                ]
            else:
                # No name selected, take symbols filtered by type, exchange, industry
                self.selected_symbols = self.symbols

        except Exception as e:
            logger.log_error("multiselect_symbols", f"Failed to get symbols: {e}")
            st.error("Failed to get symbols")

    @monitor_performance("backtest_execution")
    def handle_option_backtest(self):
        """Execute backtest for selected instrument and strategy with error handling"""
        try:
            self.get_symbol()
            
            if not hasattr(self, 'symbol') or not self.symbol:
                return
                
            strategy_name = st.selectbox('Strategy', self.strategies)
            execute = st.button(label="Execute")
            self.get_start_end()
            
            if execute:
                with st.spinner("Executing backtest..."):
                    cerebro = backtrader.Cerebro()
                    
                    # Create a data feed
                    data = backtrader.feeds.PandasData(
                        dataname=yf.download(
                            self.symbol, 
                            self.start_date, 
                            self.end_date, 
                            auto_adjust=True
                        )
                    )
                    
                    if data._dataname.empty:
                        st.write('No price data for ' + self.symbol)
                        return
                    
                    cerebro.adddata(data)
                    
                    # Dynamic create instance of strategy class
                    try:
                        strategy = globals()[strategy_name]
                        cerebro.addstrategy(strategy)
                    except KeyError:
                        st.error(f"Strategy {strategy_name} not found")
                        return

                    cerebro.addsizer(backtrader.sizers.SizerFix, stake=100)
                    cerebro.run()
                    
                    matplotlib.use('Agg')
                    st.pyplot(cerebro.plot(iplot=True)[0][0])
                    
        except Exception as e:
            logger.log_error("backtest", f"Backtest failed: {e}")
            st.error(f"Backtest failed: {e}")

    def get_start_end(self):
        """Get start and end date with validation"""
        try:
            default_start_date, default_end_date = self.get_default_start_end()
            self.start_date = st.sidebar.date_input('Start date', default_start_date)
            self.end_date = st.sidebar.date_input('End date', default_end_date)
            
            # Validate date range
            if self.start_date >= self.end_date:
                st.error("Start date must be before end date")
                return
                
        except Exception as e:
            logger.log_error("date_selection", f"Failed to get dates: {e}")
            st.error("Failed to get dates")

    def get_default_start_end(self):
        """Calculate default start and end date"""
        today = datetime.now().date()
        default_start_date = today - timedelta(days=731)
        return default_start_date, today

    @monitor_performance("overview_display")
    def handle_option_overview(self):
        """Display overview with error handling"""
        try:
            self.get_chart_indicators()
            self.get_start_end()

            for ticker in self.overview_tickers:
                try:
                    self.symbol = ticker
                    self.symbol_name = ticker
                    self.quant_figure_minutes()
                    self.quant_figure_days()
                except Exception as e:
                    logger.log_error("overview_ticker", f"Failed to process {ticker}: {e}")
                    st.warning(f"Failed to process {ticker}")
                    
        except Exception as e:
            logger.log_error("overview", f"Overview failed: {e}")
            st.error("Failed to display overview")

    @monitor_performance("watchlist_display")
    def handle_option_watchlist(self):
        """Plot quant figure for instruments of csv file with error handling"""
        try:
            self.get_chart_indicators()
            self.get_start_end()

            show_minutes = st.sidebar.checkbox('Show Minutes')
            show_days = st.sidebar.checkbox('Show Days')

            uploaded_file = st.file_uploader(
                    "Choose a csv-file with columns 'Symbol' and 'Name' for stock symbols and names"
                )
                
            if uploaded_file is not None:
                try:
                    tickers = pd.read_csv(uploaded_file)
                    st.write(tickers)
                    
                    for ticker in tickers.iterrows():
                        try:
                            self.symbol = ticker[1]['Symbol']
                            self.symbol_name = ticker[1]['Name']
                            
                            if show_minutes:
                                self.quant_figure_minutes()
                            if show_days:
                                self.quant_figure_days()
                                
                        except Exception as e:
                            logger.log_error("watchlist_ticker", f"Failed to process ticker: {e}")
                            st.warning(f"Failed to process ticker")
                            
                except Exception as e:
                    logger.log_error("watchlist_file", f"Failed to read file: {e}")
                    st.error("Failed to read uploaded file")
                    
        except Exception as e:
            logger.log_error("watchlist", f"Watchlist failed: {e}")
            st.error("Failed to display watchlist")

    def plot_quant_figure(self, prices, name, title):
        """Plot quant figure of instrument with error handling"""
        try:
            # Nothing to plot
            if prices.empty:
                st.warning(f"No data available for {name}")
                return

            st.write(title)
            
            # Handle MultiIndex columns from yfinance
            if isinstance(prices.columns, pd.MultiIndex):
                # For MultiIndex, we need to select the specific ticker columns
                if name in prices.columns.get_level_values(1):
                    # Select columns for this specific ticker
                    ticker_prices = prices.loc[:, (slice(None), name)]
                    # Flatten the MultiIndex columns for cufflinks
                    ticker_prices.columns = ticker_prices.columns.get_level_values(0)
                    qf = cf.QuantFig(ticker_prices, legend='bottom', name=name, title=title)
                else:
                    st.warning(f"No data found for {name} in the provided DataFrame.")
                    return
            else:
                # Original behavior for single-level columns
                qf = cf.QuantFig(prices, legend='bottom', name=name, title=title)

            # Add indicators based on configuration
            for indicator in config.quant_figure_indicators:
                try:
                    if indicator == 'atr':
                        qf.add_atr()
                    elif indicator == 'bb':
                        qf.add_bollinger_bands()
                    elif indicator == 'macd':
                        qf.add_macd()
                    elif indicator == 'rsi':
                        qf.add_rsi()
                    elif indicator == 'vol':
                        qf.add_volume()
                except Exception as e:
                    logger.log_error("indicator", f"Failed to add {indicator}: {e}")

            st.plotly_chart(qf.iplot(asFigure=True))
            
        except Exception as e:
            logger.log_error("quant_figure", f"Failed to plot quant figure for {name}: {e}")
            st.error(f"Failed to plot chart for {name}")

    def quant_figure_minutes(self):
        """Plot quant figure for minutes data"""
        try:
            prices = self.download_instrument_price(
                [self.symbol], self.start_date, self.end_date, self.ONE_MINUTE
            )
            self.plot_quant_figure(prices, self.symbol, f"{self.symbol_name} - Minutes")
        except Exception as e:
            logger.log_error("quant_figure_minutes", f"Failed for {self.symbol}: {e}")
            st.error(f"Failed to plot minutes chart for {self.symbol}")

    def quant_figure_days(self):
        """Plot quant figure for days data"""
        try:
            prices = self.download_instrument_price(
                [self.symbol], self.start_date, self.end_date, self.ONE_DAY
            )
            self.plot_quant_figure(prices, self.symbol, f"{self.symbol_name} - Days")
        except Exception as e:
            logger.log_error("quant_figure_days", f"Failed for {self.symbol}: {e}")
            st.error(f"Failed to plot days chart for {self.symbol}")

    def get_chart_indicators(self):
        """Get chart indicators from user input"""
        try:
            self.selected_indicators = st.sidebar.multiselect(
                "Chart Indicators", 
                config.quant_figure_indicators,
                default=config.quant_figure_indicators
            )
        except Exception as e:
            logger.log_error("chart_indicators", f"Failed to get indicators: {e}")
            self.selected_indicators = config.quant_figure_indicators

    def handle_option_sectors(self):
        """Handle sector analysis with error handling"""
        try:
            self.get_start_end()

            for sector, etf in self.sector_etf.items():
                try:
                    self.symbol = etf
                    self.symbol_name = sector
                    self.quant_figure_days()
                except Exception as e:
                    logger.log_error("sector_analysis", f"Failed to process {sector}: {e}")
                    st.warning(f"Failed to process {sector}")
                    
        except Exception as e:
            logger.log_error("sectors", f"Sector analysis failed: {e}")
            st.error("Failed to display sector analysis")

    def handle_option_momentum(self):
        """Handle momentum analysis with error handling"""
        try:
            self.get_symbols_from_multiselect()
            self.get_start_end()

            if hasattr(self, 'selected_symbols') and not self.selected_symbols.empty:
                symbols = self.selected_symbols.index.tolist()
                prices = self.download_instrument_price(symbols, self.start_date, self.end_date, self.ONE_DAY)
                
                if not prices.empty:
                    momentum_data = self.momentum.get_intraday_momentum(prices)
                    st.write("Momentum Analysis")
                    st.dataframe(momentum_data)
                else:
                    st.warning("No price data available for momentum analysis")
            else:
                st.warning("Please select symbols for momentum analysis")
                
        except Exception as e:
            logger.log_error("momentum", f"Momentum analysis failed: {e}")
            st.error("Failed to perform momentum analysis")

    def handle_option_returns(self):
        """Handle returns analysis with error handling"""
        try:
            self.get_symbols_from_multiselect()
            self.get_start_end()
       
            if hasattr(self, 'selected_symbols') and not self.selected_symbols.empty:
                symbols = self.selected_symbols.index.tolist()
                prices = self.download_instrument_price(symbols, self.start_date, self.end_date, self.ONE_DAY)
                
                if not prices.empty:
                    # Calculate returns
                    returns = prices.pct_change().dropna()
                    st.write("Returns Analysis")
                    st.dataframe(returns.describe())
                    
                    # Plot returns
                    st.line_chart(returns)
                else:
                    st.warning("No price data available for returns analysis")
            else:
                st.warning("Please select symbols for returns analysis")
                
        except Exception as e:
            logger.log_error("returns", f"Returns analysis failed: {e}")
            st.error("Failed to perform returns analysis")

    def run(self):
        """Main application runner with error handling"""
        try:
            st.title("📈 Stock Market Dashboard")
            st.markdown("A comprehensive financial analysis and trading strategy backtesting platform")
            
            # Sidebar configuration
            st.sidebar.title("Configuration")
            
            # Main application selection
            option = st.sidebar.selectbox("Select Option", self.app_options)
            
            # Route to appropriate handler
            if option == "Overview":
                self.handle_option_overview()
            elif option == "Sectors":
                self.handle_option_sectors()
            elif option == "Chart":
                self.get_symbol()
                if hasattr(self, 'symbol') and self.symbol:
                    self.get_start_end()
                    self.quant_figure_days()
            elif option == "Watchlist":
                self.handle_option_watchlist()
            elif option == "Momentum":
                self.handle_option_momentum()
            elif option == "Returns":
                self.handle_option_returns()
            elif option == "Backtest":
                self.handle_option_backtest()
            else:
                st.error("Unknown option selected")
                
        except Exception as e:
            logger.log_error("main_app", f"Application failed: {e}")
            st.error("Application encountered an error. Please check the logs for details.")


if __name__ == "__main__":
    app = start()
    app.run()
