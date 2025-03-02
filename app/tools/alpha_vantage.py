import pandas as pd
import requests
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

from app.schemas.base import AssetRequest, PriceData, TimeFrame, AssetType, DataSource

logger = logging.getLogger(__name__)

# Get Alpha Vantage API key from environment variable
API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')

# Base URL for Alpha Vantage API
BASE_URL = "https://www.alphavantage.co/query"

# Mapping from our timeframe enum to Alpha Vantage intervals
TIMEFRAME_MAP = {
    TimeFrame.ONE_MIN: "1min",
    TimeFrame.FIVE_MIN: "5min",
    TimeFrame.FIFTEEN_MIN: "15min",
    TimeFrame.THIRTY_MIN: "30min",
    TimeFrame.ONE_HOUR: "60min",
    TimeFrame.FOUR_HOUR: "4hour",  # Not directly supported by Alpha Vantage
    TimeFrame.ONE_DAY: "daily",
    TimeFrame.ONE_WEEK: "weekly",
    TimeFrame.ONE_MONTH: "monthly",
}

# Define function to use based on timeframe
def get_function_for_timeframe(timeframe: TimeFrame, asset_type: AssetType) -> str:
    """Get the appropriate Alpha Vantage function based on timeframe and asset type."""
    # For crypto
    if asset_type == AssetType.CRYPTO:
        if timeframe in [TimeFrame.ONE_MIN, TimeFrame.FIVE_MIN, TimeFrame.FIFTEEN_MIN, 
                        TimeFrame.THIRTY_MIN, TimeFrame.ONE_HOUR]:
            return "CRYPTO_INTRADAY"
        else:
            return "DIGITAL_CURRENCY_DAILY"
    
    # For stocks
    if timeframe in [TimeFrame.ONE_MIN, TimeFrame.FIVE_MIN, TimeFrame.FIFTEEN_MIN, 
                    TimeFrame.THIRTY_MIN, TimeFrame.ONE_HOUR]:
        return "TIME_SERIES_INTRADAY"
    elif timeframe == TimeFrame.ONE_DAY:
        return "TIME_SERIES_DAILY"
    elif timeframe == TimeFrame.ONE_WEEK:
        return "TIME_SERIES_WEEKLY"
    elif timeframe == TimeFrame.ONE_MONTH:
        return "TIME_SERIES_MONTHLY"
    
    # Default for unsupported timeframes
    return "TIME_SERIES_DAILY"


def format_symbol_for_alpha_vantage(symbol: str, asset_type: AssetType) -> str:
    """Format symbol for Alpha Vantage API based on asset type."""
    if asset_type == AssetType.FOREX:
        # Alpha Vantage uses format like "EUR/USD" for forex
        if "/" not in symbol:
            # If symbol is like "EURUSD", convert to "EUR/USD"
            if len(symbol) == 6:
                return f"{symbol[:3]}/{symbol[3:]}"
        return symbol
    return symbol


def fetch_alpha_vantage_data(asset_request: AssetRequest) -> PriceData:
    """Fetch financial data from Alpha Vantage."""
    if not API_KEY:
        logger.error("Alpha Vantage API key not found in environment variables")
        raise ValueError("Alpha Vantage API key not configured")
    
    symbol = format_symbol_for_alpha_vantage(asset_request.symbol, asset_request.asset_type)
    timeframe = asset_request.timeframe
    
    # Get the appropriate function
    function = get_function_for_timeframe(timeframe, asset_request.asset_type)
    
    # Prepare parameters for API request
    params = {
        "function": function,
        "apikey": API_KEY,
    }
    
    # Add additional parameters based on asset type and function
    if asset_request.asset_type == AssetType.CRYPTO:
        if function == "CRYPTO_INTRADAY":
            params.update({
                "symbol": symbol,
                "market": "USD",
                "interval": TIMEFRAME_MAP[timeframe],
                "outputsize": "full"
            })
        else:
            params.update({
                "symbol": symbol,
                "market": "USD"
            })
    else:
        params.update({
            "symbol": symbol
        })
        
        if "INTRADAY" in function:
            params.update({
                "interval": TIMEFRAME_MAP[timeframe],
                "outputsize": "full"
            })
    
    try:
        logger.info(f"Fetching {symbol} data from Alpha Vantage with function {function}")
        
        # Make API request
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        
        if "Information" in data and "Thank you" not in data["Information"]:
            logger.warning(f"Alpha Vantage API info: {data['Information']}")
        
        # Parse data based on the function used
        df = parse_alpha_vantage_response(data, function, asset_request.asset_type)
        
        # Handle empty data
        if df.empty:
            logger.error(f"No data returned from Alpha Vantage for {symbol}")
            raise ValueError(f"No data found for {symbol} with the specified parameters")
        
        # Special case for 4h timeframe which isn't directly supported
        if timeframe == TimeFrame.FOUR_HOUR:
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Convert to PriceData model
        price_data = PriceData(
            timestamp=df.index.tolist(),
            open=df['open'].tolist(),
            high=df['high'].tolist(),
            low=df['low'].tolist(),
            close=df['close'].tolist(),
            volume=df['volume'].tolist() if 'volume' in df.columns else None
        )
        
        return price_data
        
    except Exception as e:
        logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {str(e)}")
        raise ValueError(f"Failed to retrieve data from Alpha Vantage for {symbol}: {str(e)}")


def parse_alpha_vantage_response(data: Dict, function: str, asset_type: AssetType) -> pd.DataFrame:
    """Parse the Alpha Vantage API response into a DataFrame."""
    if function == "TIME_SERIES_INTRADAY":
        key = next((k for k in data.keys() if "Time Series" in k), None)
        if not key:
            raise ValueError("Unexpected response format from Alpha Vantage")
        
        time_series = data[key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = [c.split('. ')[1].lower() for c in df.columns]
        
    elif function == "TIME_SERIES_DAILY":
        if "Time Series (Daily)" not in data:
            raise ValueError("Unexpected response format from Alpha Vantage")
        
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = [c.split('. ')[1].lower() for c in df.columns]
        
    elif function == "TIME_SERIES_WEEKLY":
        if "Weekly Time Series" not in data:
            raise ValueError("Unexpected response format from Alpha Vantage")
        
        time_series = data["Weekly Time Series"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = [c.split('. ')[1].lower() for c in df.columns]
        
    elif function == "TIME_SERIES_MONTHLY":
        if "Monthly Time Series" not in data:
            raise ValueError("Unexpected response format from Alpha Vantage")
        
        time_series = data["Monthly Time Series"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = [c.split('. ')[1].lower() for c in df.columns]
        
    elif function == "CRYPTO_INTRADAY":
        key = next((k for k in data.keys() if "Time Series" in k), None)
        if not key:
            raise ValueError("Unexpected response format from Alpha Vantage")
        
        time_series = data[key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns
        df.columns = [c.split('. ')[1].lower() for c in df.columns]
        
    elif function == "DIGITAL_CURRENCY_DAILY":
        key = next((k for k in data.keys() if "Time Series" in k), None)
        if not key:
            raise ValueError("Unexpected response format from Alpha Vantage")
        
        time_series = data[key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # For crypto daily, we get multiple columns per value (USD, etc.)
        # Use the USD values
        df = df[[c for c in df.columns if "USD" in c]]
        
        # Rename columns to remove currency indicator
        df.columns = [c.split(' ')[1].lower() for c in df.columns]
        
    else:
        raise ValueError(f"Unsupported Alpha Vantage function: {function}")
    
    # Convert string values to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Sort by date ascending
    df = df.sort_index()
    
    return df 