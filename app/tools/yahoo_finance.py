import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List, Tuple

from app.schemas.base import AssetRequest, PriceData, TimeFrame, AssetType, DataSource

logger = logging.getLogger(__name__)

# Mapping from our timeframe enum to Yahoo Finance intervals
TIMEFRAME_MAP = {
    TimeFrame.ONE_MIN: "1m",
    TimeFrame.FIVE_MIN: "5m",
    TimeFrame.FIFTEEN_MIN: "15m",
    TimeFrame.THIRTY_MIN: "30m",
    TimeFrame.ONE_HOUR: "1h",
    TimeFrame.FOUR_HOUR: "4h", # Note: YF doesn't have 4h, will use 1h and resample
    TimeFrame.ONE_DAY: "1d",
    TimeFrame.ONE_WEEK: "1wk",
    TimeFrame.ONE_MONTH: "1mo",
}

# Yahoo Finance has limitations on data ranges based on interval
PERIOD_LIMITS = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=60),
    "15m": timedelta(days=60),
    "30m": timedelta(days=60),
    "1h": timedelta(days=730),
    "1d": timedelta(days=10*365),  # ~10 years
    "1wk": timedelta(days=50*365),  # ~50 years
    "1mo": timedelta(days=50*365),  # ~50 years
}


def get_symbol_with_suffix(symbol: str, asset_type: AssetType) -> str:
    """Add necessary suffix for Yahoo Finance based on asset type."""
    if asset_type == AssetType.STOCK:
        return symbol
    elif asset_type == AssetType.CRYPTO:
        return f"{symbol}-USD"
    elif asset_type == AssetType.FOREX:
        return f"{symbol}=X"
    return symbol


def handle_timeframe_limitations(
    timeframe: TimeFrame, 
    start_date: Optional[datetime], 
    end_date: Optional[datetime]
) -> Tuple[datetime, datetime]:
    """Handle Yahoo Finance limitations on timeframes and adjust dates accordingly."""
    yf_interval = TIMEFRAME_MAP[timeframe]
    
    # Set default end date to now if not provided
    if end_date is None:
        end_date = datetime.now()
    
    # Calculate max period based on interval
    max_period = PERIOD_LIMITS.get(yf_interval, timedelta(days=30))
    
    # Set default start date if not provided
    if start_date is None:
        start_date = end_date - max_period
    
    # Enforce period limits
    if end_date - start_date > max_period:
        logger.warning(f"Period exceeds Yahoo Finance limit for {yf_interval}. Adjusting start date.")
        start_date = end_date - max_period
    
    return start_date, end_date


def fetch_yahoo_finance_data(asset_request: AssetRequest) -> PriceData:
    """Fetch financial data from Yahoo Finance."""
    symbol = get_symbol_with_suffix(asset_request.symbol, asset_request.asset_type)
    timeframe = asset_request.timeframe
    
    # Handle time limitations
    start_date, end_date = handle_timeframe_limitations(
        timeframe, 
        asset_request.start_date, 
        asset_request.end_date
    )
    
    yf_interval = TIMEFRAME_MAP[timeframe]
    
    try:
        logger.info(f"Fetching {symbol} data from Yahoo Finance with interval {yf_interval}")
        
        # Download data
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=yf_interval,
            progress=False
        )
        
        # Handle empty data
        if data.empty:
            logger.error(f"No data returned for {symbol}")
            raise ValueError(f"No data found for {symbol} with the specified parameters")
        
        # Special case for 4h timeframe which isn't directly supported
        if timeframe == TimeFrame.FOUR_HOUR:
            data = data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        # Convert to PriceData model
        price_data = PriceData(
            timestamp=data.index.tolist(),
            open=data['Open'].tolist(),
            high=data['High'].tolist(),
            low=data['Low'].tolist(),
            close=data['Close'].tolist(),
            volume=data['Volume'].tolist() if 'Volume' in data.columns else None
        )
        
        return price_data
        
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
        raise ValueError(f"Failed to retrieve data for {symbol}: {str(e)}") 