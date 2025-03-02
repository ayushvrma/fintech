import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict, cast
from datetime import datetime, timedelta
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

from app.schemas.graph_state import (
    QueryParsingState, DataRetrievalState, ChartGenerationState,
    AnalysisState, ResponseGenerationState
)
from app.schemas.base import (
    AssetRequest, AssetType, TimeFrame, TechnicalIndicator, 
    DataSource, PriceData, ChartData, AnalysisResult
)
from app.tools.data_manager import DataManager
from app.tools.chart_generator import ChartGenerator
from app.tools.technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)

# Initialize language model
model = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

def query_parsing_node(state: QueryParsingState) -> QueryParsingState:
    """Parse the user query into structured asset requests and analysis parameters."""
    
    # Define prompt for parsing
    prompt = ChatPromptTemplate.from_template("""
    You are a financial assistant that helps users analyze stocks and cryptocurrencies.
    Parse the following user query and extract the relevant information:

    User Query: {query}

    Extract the following information in JSON format:
    1. Asset symbols mentioned (e.g., AAPL, BTC, ETH)
    2. Asset types (stock, crypto, etf, forex)
    3. Timeframe mentioned (e.g., 1d, 1w, 1m, etc.)
    4. Technical indicators requested (e.g., RSI, MACD, Bollinger Bands)
    5. Any specific parameters for the indicators
    6. Type of query (price_check, analysis, comparison)

    Response JSON format:
    ```json
    {
      "assets": [
        {
          "symbol": "AAPL",
          "asset_type": "stock",
          "timeframe": "1d"
        }
      ],
      "indicators": ["rsi", "macd"],
      "parameters": {
        "rsi": {"length": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9}
      },
      "query_type": "analysis",
      "timeframe_specified": true
    }
    ```

    Provide only the JSON object with no other text.
    """)
    
    # Run parsing chain
    parsing_chain = prompt | model | StrOutputParser()
    result = parsing_chain.invoke({"query": state.raw_query})
    
    # Extract JSON from the result
    json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
    if json_match:
        result = json_match.group(1)
    
    try:
        parsed_data = json.loads(result)
        
        # Convert parsed data to structured format
        assets = []
        for asset_info in parsed_data.get("assets", []):
            # Map string timeframe to TimeFrame enum
            timeframe = TimeFrame.ONE_DAY  # Default
            time_str = asset_info.get("timeframe", "1d").lower()
            
            if time_str in ["1m", "1min"]:
                timeframe = TimeFrame.ONE_MIN
            elif time_str in ["5m", "5min"]:
                timeframe = TimeFrame.FIVE_MIN
            elif time_str in ["15m", "15min"]:
                timeframe = TimeFrame.FIFTEEN_MIN
            elif time_str in ["30m", "30min"]:
                timeframe = TimeFrame.THIRTY_MIN
            elif time_str in ["1h", "1hour"]:
                timeframe = TimeFrame.ONE_HOUR
            elif time_str in ["4h", "4hour"]:
                timeframe = TimeFrame.FOUR_HOUR
            elif time_str in ["1d", "daily"]:
                timeframe = TimeFrame.ONE_DAY
            elif time_str in ["1w", "weekly"]:
                timeframe = TimeFrame.ONE_WEEK
            elif time_str in ["1M", "monthly"]:
                timeframe = TimeFrame.ONE_MONTH
            
            # Map string asset type to AssetType enum
            asset_type = AssetType.STOCK  # Default
            type_str = asset_info.get("asset_type", "stock").lower()
            
            if type_str == "crypto":
                asset_type = AssetType.CRYPTO
            elif type_str == "etf":
                asset_type = AssetType.ETF
            elif type_str == "forex":
                asset_type = AssetType.FOREX
            
            # Create AssetRequest
            assets.append(
                AssetRequest(
                    symbol=asset_info["symbol"],
                    asset_type=asset_type,
                    timeframe=timeframe,
                    # Default dates will be set in the data fetching step
                    data_source=None  # Let the system decide
                )
            )
        
        # Map indicator strings to TechnicalIndicator enum
        indicators = []
        for ind_str in parsed_data.get("indicators", []):
            ind_str = ind_str.lower()
            if ind_str == "rsi":
                indicators.append(TechnicalIndicator.RSI)
            elif ind_str == "macd":
                indicators.append(TechnicalIndicator.MACD)
            elif ind_str in ["bollinger", "bollinger_bands", "bb"]:
                indicators.append(TechnicalIndicator.BOLLINGER_BANDS)
            elif ind_str in ["ma", "sma", "ema", "moving_average", "moving averages"]:
                indicators.append(TechnicalIndicator.MOVING_AVERAGE)
            elif ind_str in ["support", "resistance", "support_resistance", "sr"]:
                indicators.append(TechnicalIndicator.SUPPORT_RESISTANCE)
            elif ind_str == "volume":
                indicators.append(TechnicalIndicator.VOLUME)
            elif ind_str == "atr":
                indicators.append(TechnicalIndicator.ATR)
            elif ind_str in ["stoch", "stochastic"]:
                indicators.append(TechnicalIndicator.STOCHASTIC)
            elif ind_str == "ichimoku":
                indicators.append(TechnicalIndicator.ICHIMOKU)
        
        # Update state with parsed information
        state.parsed_assets = assets
        state.parsed_indicators = indicators
        state.parsed_parameters = parsed_data.get("parameters", {})
        state.query_type = parsed_data.get("query_type", "price_check")
        state.timeframe_specified = parsed_data.get("timeframe_specified", False)
        state.completed = True
        
    except Exception as e:
        logger.error(f"Error parsing query: {str(e)}")
        state.error = f"Failed to parse query: {str(e)}"
    
    return state


def data_retrieval_node(state: DataRetrievalState) -> DataRetrievalState:
    """Retrieve financial data for the requested assets."""
    
    if not state.asset_requests:
        state.error = "No assets specified for data retrieval"
        return state
    
    try:
        # Fetch data for all requested assets
        results = DataManager.batch_fetch_data(state.asset_requests)
        
        # Process and store results
        retrieved_data = {}
        data_sources = {}
        
        for symbol, (price_data, source) in results.items():
            retrieved_data[symbol] = price_data
            data_sources[symbol] = source.value
        
        state.retrieved_data = retrieved_data
        state.data_sources_used = data_sources
        state.completed = True
        
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        state.error = f"Failed to retrieve data: {str(e)}"
        state.retrieval_errors = {"general": str(e)}
    
    return state


def chart_generation_node(state: ChartGenerationState) -> ChartGenerationState:
    """Generate charts for the retrieved financial data."""
    
    if not state.price_data or not state.asset_requests:
        state.error = "Missing price data or asset information for chart generation"
        return state
    
    try:
        generated_charts = {}
        chart_errors = {}
        
        # Choose a chart platform
        chart_platform = "plotly"  # Default to using plotly for interactive charts
        state.chart_platform = chart_platform
        
        # Generate charts for each asset
        for idx, asset_request in enumerate(state.asset_requests):
            symbol = asset_request.symbol
            
            if symbol not in state.price_data:
                chart_errors[symbol] = f"No price data available for {symbol}"
                continue
            
            # Get price data
            price_data = state.price_data[symbol]
            
            try:
                # Generate chart
                chart_data = ChartGenerator.generate_chart(
                    asset_request=asset_request,
                    price_data=price_data,
                    data_source=DataSource(asset_request.data_source.value if asset_request.data_source else "yahoo_finance"),
                    chart_type="candlestick",
                    platform=chart_platform
                )
                
                generated_charts[symbol] = chart_data
                
            except Exception as e:
                logger.error(f"Error generating chart for {symbol}: {str(e)}")
                chart_errors[symbol] = f"Failed to generate chart: {str(e)}"
        
        state.generated_charts = generated_charts
        state.chart_errors = chart_errors
        state.completed = True
        
    except Exception as e:
        logger.error(f"Error in chart generation: {str(e)}")
        state.error = f"Failed to generate charts: {str(e)}"
    
    return state


def technical_analysis_node(state: AnalysisState) -> AnalysisState:
    """Perform technical analysis on the financial data."""
    
    if not state.price_data or not state.indicators:
        state.error = "Missing price data or indicators for technical analysis"
        return state
    
    try:
        analysis_results = {}
        analysis_errors = {}
        
        # Perform analysis for each asset
        for asset_request in state.asset_requests:
            symbol = asset_request.symbol
            
            if symbol not in state.price_data:
                analysis_errors[symbol] = f"No price data available for {symbol}"
                continue
            
            # Get price data
            price_data = state.price_data[symbol]
            
            try:
                # Perform analysis
                result = TechnicalAnalyzer.analyze(
                    asset_request=asset_request,
                    price_data=price_data,
                    indicators=state.indicators,
                    parameters=state.parameters
                )
                
                analysis_results[symbol] = result
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                analysis_errors[symbol] = f"Failed to analyze: {str(e)}"
        
        state.analysis_results = analysis_results
        state.analysis_errors = analysis_errors
        state.completed = True
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}")
        state.error = f"Failed to perform technical analysis: {str(e)}"
    
    return state


def response_generation_node(state: ResponseGenerationState) -> ResponseGenerationState:
    """Generate a response to the user's query based on the analysis results."""
    
    # Define prompt for generating response
    prompt = ChatPromptTemplate.from_template("""
    You are a sophisticated financial analysis chatbot that provides insights on stocks and cryptocurrencies.
    Generate a detailed response to the user's query based on the analysis performed.

    User Query: {raw_query}

    Available Charts:
    {charts_info}

    Analysis Results:
    {analysis_info}

    Generate a helpful, professional response that addresses the user's question.
    Include specific insights from the analysis, key levels from technical indicators,
    and potential trends or patterns you observe.
    
    Be specific about the timeframe of the data (e.g., "Over the past week...") and
    include precise values for important metrics (e.g., "The RSI is currently at 65.3, indicating...").
    
    For technical indicators, explain their significance and what they suggest about
    potential future price movements.
    
    DO NOT include placeholders like [CHART] or [IMAGE] in your response.
    """)
    
    try:
        # Prepare chart information for prompt
        charts_info = "No charts available."
        if state.charts:
            charts_info = "\n".join([
                f"- {symbol}: Chart available for {chart.asset_info.asset_type.value.upper()} "
                f"with {chart.asset_info.timeframe.value} timeframe"
                for symbol, chart in state.charts.items()
            ])
        
        # Prepare analysis information for prompt
        analysis_info = "No technical analysis available."
        if state.analysis_results:
            analysis_sections = []
            
            for symbol, analysis in state.analysis_results.items():
                indicators_info = []
                
                for ind_result in analysis.indicators:
                    ind_name = ind_result.indicator.value
                    ind_values = ind_result.values
                    
                    # Get the most recent value for each component of the indicator
                    latest_values = {}
                    for key, values in ind_values.items():
                        if values and len(values) > 0:
                            latest_values[key] = values[-1]
                    
                    indicators_info.append(f"- {ind_name.upper()}: {latest_values}")
                
                section = f"\n{symbol}:\n" + "\n".join(indicators_info)
                analysis_sections.append(section)
            
            analysis_info = "\n".join(analysis_sections)
        
        # Run response generation chain
        response_chain = prompt | model | StrOutputParser()
        response_text = response_chain.invoke({
            "raw_query": state.raw_query,
            "charts_info": charts_info,
            "analysis_info": analysis_info
        })
        
        # Create response object
        from app.schemas.base import ChatbotResponse
        
        chart_paths = []
        if state.charts:
            for chart in state.charts.values():
                if chart.chart_path:
                    chart_paths.append(chart.chart_path)
        
        response = ChatbotResponse(
            answer=response_text,
            charts=chart_paths if chart_paths else None,
            analysis=next(iter(state.analysis_results.values())) if state.analysis_results else None
        )
        
        state.generated_response = response
        state.completed = True
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        state.error = f"Failed to generate response: {str(e)}"
    
    return state 