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
import pyfolio as pf

# Import our enhanced modules
from config import config
from utils.exceptions import DashboardError, DataDownloadError, ValidationError
from utils.validation import validate_ticker_symbol, sanitize_user_input
from utils.logging import logger, monitor_performance
from datafeed.downloader import BatchPriceDownloader, InfoDownloader
from datafeed.etoro import SymbolScraper
from analyzer.MomentumScore import MomentumScore

# Import trading strategies
from strategy.BuyAndHold import BuyAndHold
from strategy.RSIStrategy import RSIStrategy
from strategy.MinerviniMomentum import MinerviniMomentum
from strategy.SmaCross import SmaCross
from strategy.TrailingStopLoss import TrailingStopLoss


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
        """Execute backtest for selected instrument and strategy with comprehensive analysis"""
        try:
            # Backtest mode selection
            backtest_mode = st.sidebar.selectbox(
                "Backtest Mode",
                ["Single Asset", "Portfolio Analysis"],
                help="Choose between single asset backtest or portfolio analysis"
            )
            
            if backtest_mode == "Single Asset":
                self._handle_single_asset_backtest()
            else:
                self._handle_portfolio_backtest()
                    
        except Exception as e:
            logger.log_error("backtest", f"Backtest failed: {e}")
            st.error(f"Backtest failed: {e}")

    def _handle_single_asset_backtest(self):
        """Handle single asset backtest with comprehensive analysis"""
        # Use the same symbol selection logic as chart view
        self.get_symbol()
        
        if not hasattr(self, 'symbol') or not self.symbol:
            st.warning("Please select a symbol for backtesting")
            return
            
        # Strategy selection - allow multiple strategies for comparison
        selected_strategies = st.multiselect(
            'Select Strategies to Compare', 
            self.strategies,
            default=[self.strategies[0]] if self.strategies else [],
            help="Select multiple strategies to compare their performance"
        )
        
        if not selected_strategies:
            st.warning("Please select at least one strategy")
            return
            
        execute = st.button(label="Execute Strategy Comparison Backtest")
        self.get_start_end()
        
        if execute:
            with st.spinner(f"Executing backtest comparison for {len(selected_strategies)} strategies..."):
                try:
                    # Download data using our robust downloader
                    prices = self.download_instrument_price(
                        [self.symbol], self.start_date, self.end_date, '1d'
                    )
                    
                    if prices.empty:
                        st.error(f'No price data for {self.symbol}')
                        return
                    
                    # Handle MultiIndex columns from yfinance
                    if isinstance(prices.columns, pd.MultiIndex):
                        # Select columns for this specific ticker
                        if self.symbol in prices.columns.get_level_values(1):
                            ticker_prices = prices.loc[:, (slice(None), self.symbol)]
                            # Flatten the MultiIndex columns
                            ticker_prices.columns = ticker_prices.columns.get_level_values(0)
                        else:
                            st.error(f'No data found for {self.symbol}')
                            return
                    else:
                        ticker_prices = prices
                    
                    # Ensure column names are strings (not tuples)
                    if isinstance(ticker_prices.columns[0], tuple):
                        ticker_prices.columns = ticker_prices.columns.get_level_values(0)
                    
                    # Ensure we have the required columns for backtrader
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in ticker_prices.columns:
                            st.error(f'Missing required column: {col}')
                            return
                    
                    # Run backtest for each strategy
                    all_results = []
                    comparison_data = []
                    cerebro_instances = []
                     
                    for i, strategy_name in enumerate(selected_strategies):
                         try:
                             cerebro = backtrader.Cerebro()
                             
                             # Create a data feed
                             data = backtrader.feeds.PandasData(
                                 dataname=ticker_prices,
                                 datetime=None,  # Use index as datetime
                                 open='Open',
                                 high='High',
                                 low='Low',
                                 close='Close',
                                 volume='Volume',
                                 openinterest=-1  # Not used
                             )
                             
                             cerebro.adddata(data)
                             
                             # Dynamic create instance of strategy class
                             try:
                                 strategy = globals()[strategy_name]
                                 cerebro.addstrategy(strategy)
                             except KeyError:
                                 st.error(f"Strategy {strategy_name} not found")
                                 continue

                             # Add comprehensive analyzers
                             cerebro.addanalyzer(backtrader.analyzers.SharpeRatio, _name='sharpe')
                             cerebro.addanalyzer(backtrader.analyzers.DrawDown, _name='drawdown')
                             cerebro.addanalyzer(backtrader.analyzers.Returns, _name='returns')
                             cerebro.addanalyzer(backtrader.analyzers.TradeAnalyzer, _name='trades')
                             cerebro.addanalyzer(backtrader.analyzers.VWR, _name='vwr')
                             cerebro.addanalyzer(backtrader.analyzers.SQN, _name='sqn')
                             cerebro.addanalyzer(backtrader.analyzers.PyFolio, _name='pyfolio')

                             cerebro.addsizer(backtrader.sizers.SizerFix, stake=100)
                             
                             try:
                                 results = cerebro.run()
                                 
                                 if results and len(results) > 0:
                                     all_results.append((strategy_name, results[0]))
                                     cerebro_instances.append((strategy_name, cerebro))
                                     
                                     # Collect comparison data
                                     strategy_data = self._extract_strategy_metrics(results[0], strategy_name)
                                     comparison_data.append(strategy_data)
                                 else:
                                     st.warning(f"No results returned for {strategy_name}")
                                     
                             except Exception as run_error:
                                 st.error(f"Failed to run {strategy_name}: {run_error}")
                                 logger.log_error("strategy_run", f"Strategy {strategy_name} run failed: {run_error}")
                                 continue
                             
                         except Exception as e:
                             st.error(f"Failed to run {strategy_name}: {e}")
                             logger.log_error("strategy_backtest", f"Strategy {strategy_name} failed: {e}")
                             continue
                    
                    if all_results:
                        # Display strategy comparison
                        self._display_strategy_comparison(comparison_data, self.symbol)
                        
                        # Display individual results
                        for strategy_name, result in all_results:
                            with st.expander(f"📊 Detailed Results: {strategy_name}"):
                                self._display_backtest_results(result, strategy_name, self.symbol)
                                self._display_pyfolio_analysis(result, f"{self.symbol} - {strategy_name}")
                        
                        # Combined plot (if multiple strategies)
                        if len(all_results) > 1:
                            self._plot_strategy_comparison(all_results, self.symbol)
                        
                        # Display backtrader plots for each strategy
                        for strategy_name, result in all_results:
                            with st.expander(f"📈 Trade Visualization: {strategy_name}"):
                                # Find the corresponding cerebro instance
                                cerebro_instance = None
                                for name, cerebro in cerebro_instances:
                                    if name == strategy_name:
                                        cerebro_instance = cerebro
                                        break
                                self._display_backtrader_plot(result, strategy_name, self.symbol, cerebro_instance)
                    
                except Exception as e:
                    st.error(f"Backtest execution failed: {e}")
                    logger.log_error("backtest_execution", f"Detailed error: {e}")

    def _handle_portfolio_backtest(self):
        """Handle portfolio backtest with multiple assets"""
        st.subheader("Portfolio Backtest Analysis")
        
        # Portfolio selection
        portfolio_type = st.selectbox(
            "Portfolio Type",
            ["Major Indices", "Custom Selection", "Sector ETFs"],
            help="Choose portfolio composition"
        )
        
        if portfolio_type == "Major Indices":
            portfolio_symbols = [
                '^GSPC', '^DJI', '^IXIC', '^GDAXI', '^FTSE', '^N225',
                '^HSI', '^BSESN', '^AXJO', '^BVSP'
            ]
            selected_symbols = st.multiselect(
                "Select Indices",
                portfolio_symbols,
                default=['^GSPC', '^DJI', '^IXIC']
            )
        elif portfolio_type == "Sector ETFs":
            portfolio_symbols = list(self.sector_etf.values())
            selected_symbols = st.multiselect(
                "Select Sector ETFs",
                portfolio_symbols,
                default=list(self.sector_etf.values())[:3]
            )
        else:  # Custom Selection
            # Use the same symbol selection logic as single asset backtest
            self.get_symbols_from_multiselect()
            
            if hasattr(self, 'selected_symbols') and not self.selected_symbols.empty:
                # Get the symbol names for display
                symbol_names = self.selected_symbols['name'].tolist()
                selected_symbol_names = st.multiselect(
                    "Select Custom Symbols",
                    symbol_names,
                    default=symbol_names[:3] if len(symbol_names) >= 3 else symbol_names,
                    help="Select stocks from the filtered list based on your exchange and industry preferences"
                )
                
                # Convert selected names back to symbols
                if selected_symbol_names:
                    selected_symbols = self.selected_symbols.loc[
                        self.selected_symbols['name'].isin(selected_symbol_names)
                    ].index.tolist()
                else:
                    selected_symbols = []
            else:
                st.warning("Please configure filters (Exchange, Industry) to see available symbols")
                selected_symbols = []
        
        if not selected_symbols:
            st.warning("Please select at least one symbol")
            return
        
        # Strategy selection - allow multiple strategies for comparison
        selected_strategies = st.multiselect(
            'Select Strategies to Compare', 
            self.strategies,
            default=[self.strategies[0]] if self.strategies else [],
            help="Select multiple strategies to compare their performance"
        )
        
        if not selected_strategies:
            st.warning("Please select at least one strategy")
            return
        
        execute = st.button(label="Execute Portfolio Strategy Comparison")
        self.get_start_end()
        
        if execute:
            with st.spinner(f"Executing portfolio backtest for {len(selected_symbols)} symbols with {len(selected_strategies)} strategies..."):
                try:
                    # Download data for all selected symbols
                    prices = self.download_instrument_price(
                        selected_symbols, self.start_date, self.end_date, '1d'
                    )
                    
                    if prices.empty:
                        st.error('No price data available for selected symbols')
                        return

                    # Run backtest for each strategy
                    all_portfolio_results = []
                    portfolio_comparison_data = []
                    portfolio_cerebro_instances = []
                    
                    for strategy_name in selected_strategies:
                        try:
                            cerebro = backtrader.Cerebro()
                            
                            # Add data feeds for each symbol
                            for symbol in selected_symbols:
                                if isinstance(prices.columns, pd.MultiIndex):
                                    if symbol in prices.columns.get_level_values(1):
                                        symbol_prices = prices.loc[:, (slice(None), symbol)]
                                        symbol_prices.columns = symbol_prices.columns.get_level_values(0)
                                    else:
                                        st.warning(f'No data found for {symbol}')
                                        continue
                                else:
                                    symbol_prices = prices
                                
                                # Ensure column names are strings (not tuples)
                                if isinstance(symbol_prices.columns[0], tuple):
                                    symbol_prices.columns = symbol_prices.columns.get_level_values(0)
                                
                                # Ensure we have the required columns
                                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if not all(col in symbol_prices.columns for col in required_columns):
                                    st.warning(f'Missing required columns for {symbol}')
                                    continue
                                
                                data = backtrader.feeds.PandasData(
                                    dataname=symbol_prices,
                                    datetime=None,
                                    open='Open',
                                    high='High',
                                    low='Low',
                                    close='Close',
                                    volume='Volume',
                                    openinterest=-1,
                                    name=symbol
                                )
                                cerebro.adddata(data)
                            
                            # Add strategy
                            try:
                                strategy = globals()[strategy_name]
                                cerebro.addstrategy(strategy)
                            except KeyError:
                                st.error(f"Strategy {strategy_name} not found")
                                continue

                            # Add comprehensive analyzers
                            cerebro.addanalyzer(backtrader.analyzers.SharpeRatio, _name='sharpe')
                            cerebro.addanalyzer(backtrader.analyzers.DrawDown, _name='drawdown')
                            cerebro.addanalyzer(backtrader.analyzers.Returns, _name='returns')
                            cerebro.addanalyzer(backtrader.analyzers.TradeAnalyzer, _name='trades')
                            cerebro.addanalyzer(backtrader.analyzers.VWR, _name='vwr')
                            cerebro.addanalyzer(backtrader.analyzers.SQN, _name='sqn')
                            cerebro.addanalyzer(backtrader.analyzers.PyFolio, _name='pyfolio')

                            cerebro.addsizer(backtrader.sizers.SizerFix, stake=100)
                            
                            try:
                                results = cerebro.run()
                                
                                if results and len(results) > 0:
                                    all_portfolio_results.append((strategy_name, results[0]))
                                    portfolio_cerebro_instances.append((strategy_name, cerebro))
                                    
                                    # Collect comparison data
                                    strategy_data = self._extract_strategy_metrics(results[0], strategy_name)
                                    strategy_data['Portfolio Size'] = len(selected_symbols)
                                    portfolio_comparison_data.append(strategy_data)
                                else:
                                    st.warning(f"No results returned for {strategy_name} on portfolio")
                                    
                            except Exception as run_error:
                                st.error(f"Failed to run {strategy_name} on portfolio: {run_error}")
                                logger.log_error("portfolio_strategy_run", f"Strategy {strategy_name} portfolio run failed: {run_error}")
                                continue
                            
                        except Exception as e:
                            st.error(f"Failed to run {strategy_name} on portfolio: {e}")
                            logger.log_error("portfolio_strategy_backtest", f"Strategy {strategy_name} failed: {e}")
                            continue
                    
                    if all_portfolio_results:
                        # Display portfolio strategy comparison
                        self._display_portfolio_strategy_comparison(portfolio_comparison_data, selected_symbols)
                        
                        # Display individual portfolio results
                        for strategy_name, result in all_portfolio_results:
                            with st.expander(f"📊 Portfolio Results: {strategy_name}"):
                                self._display_backtest_results(result, strategy_name, f"Portfolio ({len(selected_symbols)} assets)")
                                self._display_pyfolio_analysis(result, f"Portfolio - {strategy_name}")
                        
                        # Combined portfolio plot (if multiple strategies)
                        if len(all_portfolio_results) > 1:
                            self._plot_portfolio_strategy_comparison(all_portfolio_results, selected_symbols)
                        
                        # Display backtrader plots for each portfolio strategy
                        for strategy_name, result in all_portfolio_results:
                            with st.expander(f"📈 Portfolio Trade Visualization: {strategy_name}"):
                                # Find the corresponding cerebro instance
                                cerebro_instance = None
                                for name, cerebro in portfolio_cerebro_instances:
                                    if name == strategy_name:
                                        cerebro_instance = cerebro
                                        break
                                self._display_backtrader_plot(result, strategy_name, f"Portfolio ({len(selected_symbols)} assets)", cerebro_instance)
                    
                except Exception as e:
                    st.error(f"Portfolio backtest execution failed: {e}")
                    logger.log_error("portfolio_backtest_execution", f"Detailed error: {e}")

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

            # Add indicators based on user selection
            for indicator in self.selected_indicators:
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

    def _extract_strategy_metrics(self, strategy, strategy_name):
        """Extract key metrics from strategy results for comparison using PyFolio"""
        metrics = {
            'Strategy': strategy_name,
            'Total Return': 0.0,
            'Annual Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Volatility': 0.0,
            'Total Trades': 0,
            'Win Rate': 0.0,
            'SQN': 0.0
        }
    
        try:
            if hasattr(strategy, 'analyzers') and strategy.analyzers:
                # Try to get PyFolio data first for comprehensive metrics
                pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
                if pyfolio_analyzer:
                    try:
                        returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
                        if returns is not None and len(returns) > 0:
                            # Calculate comprehensive metrics from PyFolio returns
                            total_return = (1 + returns).prod() - 1
                            annual_return = returns.mean() * 252
                            volatility = returns.std() * np.sqrt(252)
                            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                            
                            # Calculate max drawdown
                            cumulative_returns = (1 + returns).cumprod()
                            running_max = cumulative_returns.expanding().max()
                            drawdown = (cumulative_returns / running_max - 1)
                            max_drawdown = drawdown.min()
                            
                            metrics.update({
                                'Total Return': total_return,
                                'Annual Return': annual_return * 100,  # Convert to percentage
                                'Sharpe Ratio': sharpe_ratio,
                                'Max Drawdown': max_drawdown,
                                'Volatility': volatility * 100  # Convert to percentage
                            })
                            
                            # Get trade information from transactions if available
                            if transactions is not None and len(transactions) > 0:
                                total_trades = len(transactions)
                                winning_trades = len(transactions[transactions['pnl'] > 0])
                                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                                
                                metrics.update({
                                    'Total Trades': total_trades,
                                    'Win Rate': win_rate
                                })
                    except Exception as pyfolio_error:
                        # Silently fall back to individual analyzers
                        pass
                
                # Fallback to individual analyzers if PyFolio fails or if Total Return is still 0.0
                if metrics['Total Return'] == 0.0:
                    # Returns
                    returns_analyzer = strategy.analyzers.getbyname('returns')
                    if returns_analyzer:
                        if hasattr(returns_analyzer, 'rtot'):
                            metrics['Total Return'] = returns_analyzer.rtot
                        if hasattr(returns_analyzer, 'rnorm100'):
                            metrics['Annual Return'] = returns_analyzer.rnorm100
                    
                    # Risk metrics
                    sharpe_analyzer = strategy.analyzers.getbyname('sharpe')
                    if sharpe_analyzer and hasattr(sharpe_analyzer, 'ratio') and sharpe_analyzer.ratio is not None:
                        metrics['Sharpe Ratio'] = sharpe_analyzer.ratio
                    
                    drawdown_analyzer = strategy.analyzers.getbyname('drawdown')
                    if drawdown_analyzer and hasattr(drawdown_analyzer, 'max') and drawdown_analyzer.max is not None:
                        metrics['Max Drawdown'] = drawdown_analyzer.max
                    
                    vwr_analyzer = strategy.analyzers.getbyname('vwr')
                    if vwr_analyzer and hasattr(vwr_analyzer, 'vwr') and vwr_analyzer.vwr is not None:
                        metrics['Volatility'] = vwr_analyzer.vwr
                    
                    sqn_analyzer = strategy.analyzers.getbyname('sqn')
                    if sqn_analyzer and hasattr(sqn_analyzer, 'sqn') and sqn_analyzer.sqn is not None:
                        metrics['SQN'] = sqn_analyzer.sqn
                    
                    # Trade metrics - use get_analysis() method
                    trades_analyzer = strategy.analyzers.getbyname('trades')
                    if trades_analyzer:
                        try:
                            analysis = trades_analyzer.get_analysis()
                            if analysis and 'total' in analysis and 'total' in analysis['total']:
                                total_trades = analysis['total']['total']
                                metrics['Total Trades'] = total_trades
                                
                                if total_trades > 0:
                                    won_trades = analysis.get('won', {}).get('total', 0)
                                    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                                    metrics['Win Rate'] = win_rate
                        except Exception as e:
                            # Fallback to old method if get_analysis fails
                            if hasattr(trades_analyzer, 'total') and hasattr(trades_analyzer.total, 'total') and trades_analyzer.total.total > 0:
                                metrics['Total Trades'] = trades_analyzer.total.total
                                if hasattr(trades_analyzer, 'won') and hasattr(trades_analyzer.won, 'total'):
                                    won_trades = trades_analyzer.won.total
                                    total_trades = trades_analyzer.total.total
                                    if total_trades > 0:
                                        metrics['Win Rate'] = (won_trades / total_trades) * 100
                            
                            # Additional fallback for BuyAndHold strategy (always 1 trade)
                            if strategy.__class__.__name__ == 'BuyAndHold':
                                metrics['Total Trades'] = 1
                                # Calculate win rate based on final value vs initial value
                                if hasattr(strategy, 'broker') and hasattr(strategy, 'val_start'):
                                    final_value = strategy.broker.get_value()
                                    initial_value = strategy.val_start
                                    if final_value > initial_value:
                                        metrics['Win Rate'] = 100.0
                                    else:
                                        metrics['Win Rate'] = 0.0
                    
        except Exception as e:
            st.error(f"Failed to extract metrics for {strategy_name}: {e}")
        return metrics

    def _display_backtest_results(self, strategy, strategy_name, symbol):
        """Display detailed backtest results"""
        try:
            st.subheader(f"📊 {strategy_name} Results for {symbol}")
            
            if hasattr(strategy, 'analyzers'):
                # Returns
                returns_analyzer = strategy.analyzers.getbyname('returns')
                if returns_analyzer:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if hasattr(returns_analyzer, 'rtot'):
                            st.metric("Total Return", f"{returns_analyzer.rtot:.2%}")
                    with col2:
                        if hasattr(returns_analyzer, 'rnorm100'):
                            st.metric("Annual Return", f"{returns_analyzer.rnorm100:.2f}%")
                    with col3:
                        if hasattr(returns_analyzer, 'rnorm'):
                            st.metric("Normalized Return", f"{returns_analyzer.rnorm:.2f}")
                
                # Risk metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    sharpe_analyzer = strategy.analyzers.getbyname('sharpe')
                    if sharpe_analyzer and hasattr(sharpe_analyzer, 'ratio') and sharpe_analyzer.ratio is not None:
                        st.metric("Sharpe Ratio", f"{sharpe_analyzer.ratio:.3f}")
                
                with col2:
                    drawdown_analyzer = strategy.analyzers.getbyname('drawdown')
                    if drawdown_analyzer and hasattr(drawdown_analyzer, 'max') and drawdown_analyzer.max is not None:
                        st.metric("Max Drawdown", f"{drawdown_analyzer.max:.2%}")
                
                with col3:
                    vwr_analyzer = strategy.analyzers.getbyname('vwr')
                    if vwr_analyzer and hasattr(vwr_analyzer, 'vwr') and vwr_analyzer.vwr is not None:
                        st.metric("Volatility", f"{vwr_analyzer.vwr:.2f}")
                
                # Trade analysis - use get_analysis() method
                trades_analyzer = strategy.analyzers.getbyname('trades')
                if trades_analyzer:
                    try:
                        analysis = trades_analyzer.get_analysis()
                        
                        if analysis and 'total' in analysis and 'total' in analysis['total']:
                            total_trades = analysis['total']['total']
                            
                            if total_trades > 0:
                                st.subheader("📈 Trade Analysis")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Trades", total_trades)
                                with col2:
                                    won_trades = analysis.get('won', {}).get('total', 0)
                                    st.metric("Winning Trades", won_trades)
                                with col3:
                                    lost_trades = analysis.get('lost', {}).get('total', 0)
                                    st.metric("Losing Trades", lost_trades)
                                with col4:
                                    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                                    st.metric("Win Rate", f"{win_rate:.1f}%")
                            else:
                                # Special handling for BuyAndHold strategy
                                if strategy.__class__.__name__ == 'BuyAndHold':
                                    st.subheader("📈 Trade Analysis")
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Total Trades", 1)
                                    with col2:
                                        # Check if it's a winning trade
                                        if hasattr(strategy, 'broker') and hasattr(strategy, 'val_start'):
                                            final_value = strategy.broker.get_value()
                                            initial_value = strategy.val_start
                                            if final_value > initial_value:
                                                st.metric("Winning Trades", 1)
                                            else:
                                                st.metric("Winning Trades", 0)
                                        else:
                                            st.metric("Winning Trades", 0)
                                    with col3:
                                        # Check if it's a losing trade
                                        if hasattr(strategy, 'broker') and hasattr(strategy, 'val_start'):
                                            final_value = strategy.broker.get_value()
                                            initial_value = strategy.val_start
                                            if final_value <= initial_value:
                                                st.metric("Losing Trades", 1)
                                            else:
                                                st.metric("Losing Trades", 0)
                                        else:
                                            st.metric("Losing Trades", 0)
                                    with col4:
                                        # Calculate win rate
                                        if hasattr(strategy, 'broker') and hasattr(strategy, 'val_start'):
                                            final_value = strategy.broker.get_value()
                                            initial_value = strategy.val_start
                                            if final_value > initial_value:
                                                st.metric("Win Rate", "100.0%")
                                            else:
                                                st.metric("Win Rate", "0.0%")
                                        else:
                                            st.metric("Win Rate", "0.0%")
                                else:
                                    st.info("No trades executed during the backtest period")
                        else:
                            st.info("No trade data available")
                    except Exception as e:
                        st.warning(f"Could not retrieve trade analysis: {e}")
                else:
                    st.info("No trade analyzer available")
                    
        except Exception as e:
            st.error(f"Failed to display backtest results: {e}")

    def _display_pyfolio_analysis(self, strategy, title):
        """Display PyFolio analysis results"""
        try:
            st.subheader(f"📊 PyFolio Analysis: {title}")
            
            if hasattr(strategy, 'analyzers'):
                pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
                if pyfolio_analyzer:
                    try:
                        returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
                        
                        if returns is not None and len(returns) > 0:
                            # Display key metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total_return = (1 + returns).prod() - 1
                                st.metric("Total Return", f"{total_return:.2%}")
                            
                            with col2:
                                annual_return = returns.mean() * 252
                                st.metric("Annual Return", f"{annual_return:.2%}")
                            
                            with col3:
                                volatility = returns.std() * np.sqrt(252)
                                st.metric("Annual Volatility", f"{volatility:.2%}")
                            
                            # Calculate and display drawdown
                            cumulative_returns = (1 + returns).cumprod()
                            running_max = cumulative_returns.expanding().max()
                            drawdown = (cumulative_returns / running_max - 1)
                            max_drawdown = drawdown.min()
                            
                            st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
                            
                            # Display returns chart
                            st.line_chart(returns.cumsum())
                            
                        if transactions is not None and len(transactions) > 0:
                            st.subheader("📈 Trade Summary")
                            st.dataframe(transactions)
                            
                    except Exception as e:
                        st.warning(f"PyFolio analysis not available: {e}")
                else:
                    st.info("PyFolio analyzer not found")
            else:
                st.info("No analyzers available")
                
        except Exception as e:
            st.error(f"Failed to display PyFolio analysis: {e}")

    def _display_strategy_comparison(self, comparison_data, symbol):
        """Display strategy comparison table"""
        try:
            st.subheader(f"📊 Strategy Comparison for {symbol}")
            
            if comparison_data:
                # Create DataFrame for display
                df = pd.DataFrame(comparison_data)
                
                # Format the display DataFrame
                display_df = df.copy()
                display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.2%}")
                display_df['Annual Return'] = display_df['Annual Return'].apply(lambda x: f"{x:.2f}%")
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
                display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2f}%")
                display_df['Total Trades'] = display_df['Total Trades'].apply(lambda x: f"{x:.0f}")
                display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
                display_df['SQN'] = display_df['SQN'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No comparison data available")
                
        except Exception as e:
            st.error(f"Failed to display strategy comparison: {e}")

    def _display_portfolio_strategy_comparison(self, comparison_data, symbols):
        """Display portfolio strategy comparison table"""
        try:
            st.subheader(f"📊 Portfolio Strategy Comparison ({len(symbols)} assets)")
            
            if comparison_data:
                # Create DataFrame for display
                df = pd.DataFrame(comparison_data)
                
                # Format the display DataFrame
                display_df = df.copy()
                display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.2%}")
                display_df['Annual Return'] = display_df['Annual Return'].apply(lambda x: f"{x:.2f}%")
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
                display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.2f}%")
                display_df['Total Trades'] = display_df['Total Trades'].apply(lambda x: f"{x:.0f}")
                display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
                display_df['SQN'] = display_df['SQN'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No portfolio comparison data available")
                
        except Exception as e:
            st.error(f"Failed to display portfolio strategy comparison: {e}")

    def _plot_strategy_comparison(self, all_results, symbol):
        """Plot strategy comparison charts"""
        try:
            st.subheader(f"📈 Strategy Comparison Charts for {symbol}")
            
            # Extract metrics for plotting
            strategies = []
            total_returns = []
            max_drawdowns = []
            trade_counts = []
            
            for strategy_name, result in all_results:
                strategies.append(strategy_name)
                
                # Extract metrics
                metrics = self._extract_strategy_metrics(result, strategy_name)
                total_returns.append(metrics['Total Return'])
                max_drawdowns.append(metrics['Max Drawdown'])
                trade_counts.append(metrics['Total Trades'])
            
            # Create comparison charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Total Returns")
                returns_df = pd.DataFrame({
                    'Strategy': strategies,
                    'Total Return': total_returns
                })
                st.bar_chart(returns_df.set_index('Strategy'))
            
            with col2:
                st.subheader("Max Drawdown")
                drawdown_df = pd.DataFrame({
                    'Strategy': strategies,
                    'Max Drawdown': max_drawdowns
                })
                st.bar_chart(drawdown_df.set_index('Strategy'))
            
            with col3:
                st.subheader("Total Trades")
                trades_df = pd.DataFrame({
                    'Strategy': strategies,
                    'Total Trades': trade_counts
                })
                st.bar_chart(trades_df.set_index('Strategy'))
                
        except Exception as e:
            st.error(f"Failed to plot strategy comparison: {e}")

    def _plot_portfolio_strategy_comparison(self, all_results, symbols):
        """Plot portfolio strategy comparison charts"""
        try:
            st.subheader(f"📈 Portfolio Strategy Comparison Charts ({len(symbols)} assets)")
            
            # Extract metrics for plotting
            strategies = []
            total_returns = []
            max_drawdowns = []
            trade_counts = []
            
            for strategy_name, result in all_results:
                strategies.append(strategy_name)
                
                # Extract metrics
                metrics = self._extract_strategy_metrics(result, strategy_name)
                total_returns.append(metrics['Total Return'])
                max_drawdowns.append(metrics['Max Drawdown'])
                trade_counts.append(metrics['Total Trades'])
            
            # Create comparison charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Total Returns")
                returns_df = pd.DataFrame({
                    'Strategy': strategies,
                    'Total Return': total_returns
                })
                st.bar_chart(returns_df.set_index('Strategy'))
            
            with col2:
                st.subheader("Max Drawdown")
                drawdown_df = pd.DataFrame({
                    'Strategy': strategies,
                    'Max Drawdown': max_drawdowns
                })
                st.bar_chart(drawdown_df.set_index('Strategy'))
            
            with col3:
                st.subheader("Total Trades")
                trades_df = pd.DataFrame({
                    'Strategy': strategies,
                    'Total Trades': trade_counts
                })
                st.bar_chart(trades_df.set_index('Strategy'))
                
        except Exception as e:
            st.error(f"Failed to plot portfolio strategy comparison: {e}")

    def _display_backtrader_plot(self, strategy, strategy_name, symbol, cerebro_instance=None):
        """Display backtrader plots with trades, buys, sells, and P&L calculation"""
        try:
            st.subheader(f"📈 Trade Visualization: {strategy_name} for {symbol}")
            
            if cerebro_instance:
                # Set matplotlib backend for Streamlit
                matplotlib.use('Agg')
                
                try:
                    # Generate the plot - backtrader.plot() returns a list of figure lists
                    plot_result = cerebro_instance.plot(style='candlestick', barup='green', bardown='red')
                    
                    # Handle the plot result structure
                    if plot_result and len(plot_result) > 0:
                        # plot_result is a list of figure lists, get the first one
                        first_figure_list = plot_result[0]
                        
                        if first_figure_list and len(first_figure_list) > 0:
                            # Get the first figure from the first figure list
                            fig = first_figure_list[0]
                            
                            # Display the figure
                            st.pyplot(fig)
                            
                            # Close the figure to free memory
                            plt.close(fig)
                        else:
                            st.warning("No figures generated in plot result")
                    else:
                        st.warning("No plot generated")
                        
                except Exception as plot_error:
                    st.error(f"Failed to generate plot: {plot_error}")
                    # Fallback to placeholder
                    st.info("Trade visualization not available")
            else:
                st.info("No cerebro instance available for plotting")
            
            # Display trade summary
            if hasattr(strategy, 'analyzers'):
                trades_analyzer = strategy.analyzers.getbyname('trades')
                if trades_analyzer:
                    try:
                        analysis = trades_analyzer.get_analysis()
                        
                        if analysis and 'total' in analysis and 'total' in analysis['total']:
                            total_trades = analysis['total']['total']
                            
                            if total_trades > 0:
                                st.write("**Trade Summary:**")
                                won_trades = analysis.get('won', {}).get('total', 0)
                                lost_trades = analysis.get('lost', {}).get('total', 0)
                                win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                                
                                trade_summary = {
                                    "Metric": ["Total Trades", "Winning Trades", "Losing Trades", "Win Rate"],
                                    "Value": [
                                        total_trades,
                                        won_trades,
                                        lost_trades,
                                        f"{win_rate:.1f}%"
                                    ]
                                }
                                st.table(pd.DataFrame(trade_summary))
                            else:
                                # Special handling for BuyAndHold strategy
                                if strategy.__class__.__name__ == 'BuyAndHold':
                                    st.write("**Trade Summary:**")
                                    # Check if it's a winning trade
                                    if hasattr(strategy, 'broker') and hasattr(strategy, 'val_start'):
                                        final_value = strategy.broker.get_value()
                                        initial_value = strategy.val_start
                                        if final_value > initial_value:
                                            won_trades = 1
                                            lost_trades = 0
                                            win_rate = 100.0
                                        else:
                                            won_trades = 0
                                            lost_trades = 1
                                            win_rate = 0.0
                                    else:
                                        won_trades = 0
                                        lost_trades = 0
                                        win_rate = 0.0
                                    
                                    trade_summary = {
                                        "Metric": ["Total Trades", "Winning Trades", "Losing Trades", "Win Rate"],
                                        "Value": [
                                            1,
                                            won_trades,
                                            lost_trades,
                                            f"{win_rate:.1f}%"
                                        ]
                                    }
                                    st.table(pd.DataFrame(trade_summary))
                                else:
                                    st.write("No trades executed during the backtest period.")
                        else:
                            st.write("No trade data available.")
                    except Exception as e:
                        st.write("Could not retrieve trade summary.")
                    
        except Exception as e:
            st.error(f"Failed to display backtrader plot: {e}")

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
                    self.get_chart_indicators()
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
