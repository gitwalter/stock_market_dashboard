# Stock Market Dashboard

A Streamlit-based stock market dashboard application that provides real-time market data, sector analysis, and trading strategy backtesting.

## Features

- **Real-time Market Data**: Live price data for major indices and stocks
- **Sector Analysis**: Performance analysis of sector ETFs
- **Trading Strategies**: Backtesting framework for trading strategies
- **Interactive Charts**: Candlestick charts and technical indicators
- **Data Download**: Robust data fetching from Yahoo Finance

## Project Structure

```
stock_market_dashboard/
├── stock_market_dashboard.py    # Main Streamlit application
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── datafeed/                    # Data fetching modules
│   ├── downloader.py           # Yahoo Finance data downloader
│   └── etoro.py                # eToro data integration
├── config/                      # Configuration files
│   ├── __init__.py
│   └── secrets.py              # API keys (template)
├── analyzer/                    # Data analysis modules
└── strategy/                    # Trading strategy modules
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock_market_dashboard
   ```

2. **Use the backtrader conda environment** (recommended):
   ```bash
   conda activate backtrader
   ```

   **Note**: This project is configured to work with the `backtrader` conda environment which has all the correct dependencies and SSL libraries installed.

3. **Alternative: Create a new conda environment**:
   ```bash
   conda create -n stock_dashboard python=3.9
   conda activate stock_dashboard
   pip install -r requirements.txt
   ```

4. **Configure API keys** (optional):
   - Edit `config/secrets.py` to add your API keys if needed

## Usage

### Run the Dashboard

```bash
streamlit run stock_market_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Data Downloader

The project includes a robust data downloader that fetches data from Yahoo Finance:

```python
from datafeed.downloader import BatchPriceDownloader
from datetime import datetime, timedelta

# Download data for multiple tickers
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
tickers = ['AAPL', 'MSFT', 'GOOGL']

downloader = BatchPriceDownloader(tickers, start_date, end_date, '1d')
data = downloader.get_yahoo_prices()
```

## Key Components

### Data Downloader (`datafeed/downloader.py`)

- `InfoDownloader`: Fetches company information
- `BatchPriceDownloader`: Downloads historical price data for multiple tickers
- Supports various intervals (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo)
- Handles rate limiting and error recovery

### Main Application (`stock_market_dashboard.py`)

- Streamlit web interface
- Overview section with major indices
- Sector analysis with ETF performance
- Interactive charts using Plotly and Cufflinks
- Trading strategy integration

## Dependencies

Key packages:
- `streamlit`: Web application framework
- `yfinance`: Yahoo Finance data fetching
- `pandas`: Data manipulation
- `plotly`: Interactive charts
- `cufflinks`: Financial charting
- `backtrader`: Trading strategy backtesting

## Troubleshooting

### Common Issues

1. **SSL DLL Error**: Make sure you're using the correct virtual environment
2. **Rate Limiting**: Yahoo Finance may limit requests; the downloader includes retry logic
3. **Cufflinks Error**: This may occur when importing the app directly; run with `streamlit run`

### Environment Issues

If you encounter environment-related errors:
1. Ensure you're using the correct Python environment
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check that all packages are compatible

## Development

The project is structured for easy development and testing:

- Core data fetching is isolated in `datafeed/`
- Configuration is centralized in `config/`
- Main application logic is in `stock_market_dashboard.py`

## License

This project is for educational and personal use. Please respect Yahoo Finance's terms of service when using their data.
