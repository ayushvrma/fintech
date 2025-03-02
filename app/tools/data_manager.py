import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from app.schemas.base import (
    AssetRequest, PriceData, AssetType, DataSource, TimeFrame, ChartData
)
from app.tools.yahoo_finance import fetch_yahoo_finance_data
from app.tools.alpha_vantage import fetch_alpha_vantage_data

logger = logging.getLogger(__name__)

class DataManager:
    """Manager for fetching financial data from various sources."""
    
    @staticmethod
    def fetch_data(asset_request: AssetRequest) -> Tuple[PriceData, DataSource]:
        """
        Fetch data for a given asset from the most appropriate source.
        Returns the price data and the source used.
        """
        # If a specific data source is requested, use it
        if asset_request.data_source:
            return DataManager._fetch_from_source(asset_request, asset_request.data_source)
        
        # Otherwise, try sources in order of preference
        sources_to_try = [
            DataSource.YAHOO_FINANCE,
            DataSource.ALPHA_VANTAGE
            # Add more sources as they become available
        ]
        
        # For crypto, try specialized sources first
        if asset_request.asset_type == AssetType.CRYPTO:
            sources_to_try = [
                DataSource.ALPHA_VANTAGE, 
                DataSource.YAHOO_FINANCE,
                # Can add BINANCE, CCXT, etc.
            ]
        
        # Try each source until one works
        last_error = None
        for source in sources_to_try:
            try:
                price_data, source_used = DataManager._fetch_from_source(asset_request, source)
                return price_data, source_used
            except Exception as e:
                logger.warning(f"Failed to fetch data from {source} for {asset_request.symbol}: {str(e)}")
                last_error = e
        
        # If we get here, all sources failed
        error_msg = f"Failed to fetch data for {asset_request.symbol} from any source"
        if last_error:
            error_msg += f": {str(last_error)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    @staticmethod
    def _fetch_from_source(asset_request: AssetRequest, source: DataSource) -> Tuple[PriceData, DataSource]:
        """Fetch data from a specific source."""
        if source == DataSource.YAHOO_FINANCE:
            price_data = fetch_yahoo_finance_data(asset_request)
            return price_data, DataSource.YAHOO_FINANCE
        
        elif source == DataSource.ALPHA_VANTAGE:
            price_data = fetch_alpha_vantage_data(asset_request)
            return price_data, DataSource.ALPHA_VANTAGE
        
        # Add more data sources as needed
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    @staticmethod
    def batch_fetch_data(asset_requests: List[AssetRequest]) -> Dict[str, Tuple[PriceData, DataSource]]:
        """
        Fetch data for multiple assets in batch.
        Returns a dictionary mapping symbols to (price_data, source) tuples.
        """
        results = {}
        errors = {}
        
        for request in asset_requests:
            try:
                price_data, source = DataManager.fetch_data(request)
                results[request.symbol] = (price_data, source)
            except Exception as e:
                logger.error(f"Error fetching data for {request.symbol}: {str(e)}")
                errors[request.symbol] = str(e)
        
        if not results and errors:
            raise ValueError(f"Failed to fetch data for any requested assets: {errors}")
        
        return results
    
    @staticmethod
    def align_multiple_assets_data(
        data_dict: Dict[str, PriceData], 
        resample_rule: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Align data from multiple assets to the same timestamps.
        Optionally resample to a specific frequency.
        
        Args:
            data_dict: Dictionary mapping symbols to PriceData objects
            resample_rule: Optional pandas resample rule (e.g., '1D', '1H')
            
        Returns:
            Dictionary mapping symbols to aligned DataFrames
        """
        # Convert PriceData to DataFrames
        dfs = {}
        for symbol, price_data in data_dict.items():
            df = pd.DataFrame({
                'open': price_data.open,
                'high': price_data.high,
                'low': price_data.low,
                'close': price_data.close,
                'volume': price_data.volume if price_data.volume else [None] * len(price_data.timestamp)
            }, index=price_data.timestamp)
            dfs[symbol] = df
        
        # Find common date range
        min_dates = []
        max_dates = []
        for df in dfs.values():
            min_dates.append(df.index.min())
            max_dates.append(df.index.max())
        
        start_date = max(min_dates)
        end_date = min(max_dates)
        
        # Align dataframes
        aligned_dfs = {}
        for symbol, df in dfs.items():
            # Filter to common date range
            aligned_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
            
            # Resample if requested
            if resample_rule:
                aligned_df = aligned_df.resample(resample_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            aligned_dfs[symbol] = aligned_df
        
        return aligned_dfs
    
    @staticmethod
    def convert_dataframe_to_price_data(df: pd.DataFrame) -> PriceData:
        """Convert a pandas DataFrame to a PriceData object."""
        return PriceData(
            timestamp=df.index.tolist(),
            open=df['open'].tolist(),
            high=df['high'].tolist(),
            low=df['low'].tolist(),
            close=df['close'].tolist(),
            volume=df['volume'].tolist() if 'volume' in df.columns and not df['volume'].isna().all() else None
        ) 