from enum import Enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"


class TimeFrame(str, Enum):
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class TechnicalIndicator(str, Enum):
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGE = "moving_average"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLUME = "volume"
    ATR = "atr"
    STOCHASTIC = "stochastic"
    ICHIMOKU = "ichimoku"


class DataSource(str, Enum):
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    BINANCE = "binance"
    CCXT = "ccxt"


class AssetRequest(BaseModel):
    """Request parameters for asset data."""
    symbol: str = Field(..., description="The ticker symbol of the asset")
    asset_type: AssetType = Field(..., description="Type of the financial asset")
    timeframe: TimeFrame = Field(TimeFrame.ONE_DAY, description="Time interval for data points")
    start_date: Optional[datetime] = Field(None, description="Start date for data retrieval")
    end_date: Optional[datetime] = Field(None, description="End date for data retrieval")
    data_source: Optional[DataSource] = Field(None, description="Preferred data source")


class PriceData(BaseModel):
    """Price data for an asset."""
    timestamp: List[datetime] = Field(..., description="Timestamps for data points")
    open: List[float] = Field(..., description="Opening prices")
    high: List[float] = Field(..., description="High prices")
    low: List[float] = Field(..., description="Low prices")
    close: List[float] = Field(..., description="Closing prices")
    volume: Optional[List[float]] = Field(None, description="Volume data")


class ChartData(BaseModel):
    """Chart data including path to saved chart."""
    asset_info: AssetRequest = Field(..., description="Information about the requested asset")
    price_data: PriceData = Field(..., description="Price data series")
    chart_path: Optional[str] = Field(None, description="Path to the saved chart image")
    chart_html: Optional[str] = Field(None, description="HTML representation of interactive chart")
    data_source: DataSource = Field(..., description="Data source used")


class AnalysisRequest(BaseModel):
    """Request for technical analysis on an asset."""
    asset_info: AssetRequest = Field(..., description="Information about the asset to analyze")
    indicators: List[TechnicalIndicator] = Field(..., description="Technical indicators to apply")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for indicators")


class IndicatorResult(BaseModel):
    """Result of a technical indicator calculation."""
    indicator: TechnicalIndicator = Field(..., description="Technical indicator applied")
    values: Dict[str, List[float]] = Field(..., description="Calculated indicator values")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for the indicator")


class AnalysisResult(BaseModel):
    """Results of technical analysis on an asset."""
    asset_info: AssetRequest = Field(..., description="Information about the analyzed asset")
    price_data: PriceData = Field(..., description="Price data used for analysis")
    indicators: List[IndicatorResult] = Field(..., description="Calculated technical indicators")
    chart_paths: Optional[List[str]] = Field(None, description="Paths to generated chart images")
    chart_htmls: Optional[List[str]] = Field(None, description="HTML representations of charts")
    insights: Optional[str] = Field(None, description="AI-generated insights about the analysis")


class UserQuery(BaseModel):
    """User input for the chatbot."""
    query: str = Field(..., description="User's question or request")


class ChatbotResponse(BaseModel):
    """Response from the chatbot."""
    answer: str = Field(..., description="Text response to the user's query")
    charts: Optional[List[str]] = Field(None, description="Paths to chart images")
    analysis: Optional[AnalysisResult] = Field(None, description="Technical analysis results")
    error: Optional[str] = Field(None, description="Error message if any") 