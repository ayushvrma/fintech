from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from app.schemas.base import (
    AssetRequest, PriceData, ChartData, AnalysisResult, 
    UserQuery, ChatbotResponse, TechnicalIndicator
)


class GraphState(BaseModel):
    """Base state for LangGraph workflows."""
    completed: bool = Field(False, description="Whether the workflow has completed")
    error: Optional[str] = Field(None, description="Error message if any occurred")


class QueryParsingState(GraphState):
    """State for query parsing node."""
    raw_query: str = Field(..., description="The raw query from the user")
    parsed_assets: Optional[List[AssetRequest]] = Field(None, description="Parsed asset requests")
    parsed_indicators: Optional[List[TechnicalIndicator]] = Field(None, description="Parsed technical indicators")
    parsed_parameters: Optional[Dict[str, Any]] = Field(None, description="Parsed parameters for indicators")
    query_type: Optional[str] = Field(None, description="Type of query (price_check, analysis, comparison)")
    timeframe_specified: Optional[bool] = Field(False, description="Whether timeframe was specified")


class DataRetrievalState(GraphState):
    """State for data retrieval node."""
    asset_requests: List[AssetRequest] = Field(..., description="List of assets to retrieve data for")
    retrieved_data: Optional[Dict[str, PriceData]] = Field(None, description="Retrieved price data for each asset")
    data_sources_used: Optional[Dict[str, str]] = Field(None, description="Data sources used for each asset")
    retrieval_errors: Optional[Dict[str, str]] = Field(None, description="Errors during data retrieval")


class ChartGenerationState(GraphState):
    """State for chart generation node."""
    price_data: Dict[str, PriceData] = Field(..., description="Price data for each asset")
    asset_requests: List[AssetRequest] = Field(..., description="Asset requests with parameters")
    generated_charts: Optional[Dict[str, ChartData]] = Field(None, description="Generated chart data")
    chart_platform: Optional[str] = Field(None, description="Platform used for chart generation")
    chart_errors: Optional[Dict[str, str]] = Field(None, description="Errors during chart generation")


class AnalysisState(GraphState):
    """State for technical analysis node."""
    price_data: Dict[str, PriceData] = Field(..., description="Price data for each asset")
    asset_requests: List[AssetRequest] = Field(..., description="Asset requests with parameters")
    indicators: List[TechnicalIndicator] = Field(..., description="Technical indicators to apply")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for indicators")
    analysis_results: Optional[Dict[str, AnalysisResult]] = Field(None, description="Results of analysis")
    analysis_errors: Optional[Dict[str, str]] = Field(None, description="Errors during analysis")


class ResponseGenerationState(GraphState):
    """State for response generation node."""
    raw_query: str = Field(..., description="Original user query")
    charts: Optional[Dict[str, ChartData]] = Field(None, description="Generated charts")
    analysis_results: Optional[Dict[str, AnalysisResult]] = Field(None, description="Analysis results")
    generated_response: Optional[ChatbotResponse] = Field(None, description="Generated response")


class FinancialChatbotState(BaseModel):
    """Complete state for financial chatbot workflow."""
    query_parsing: QueryParsingState
    data_retrieval: Optional[DataRetrievalState] = None
    chart_generation: Optional[ChartGenerationState] = None
    analysis: Optional[AnalysisState] = None
    response_generation: Optional[ResponseGenerationState] = None
    final_response: Optional[ChatbotResponse] = None 