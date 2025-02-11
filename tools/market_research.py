from typing import Optional, List
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from nsepython import nse_eq, nse_fno
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from loguru import logger

class NSEDataTool(BaseTool):
    name = "nse_data_tool"
    description = "Get real-time and historical data from NSE for stocks and F&O"

    def _run(self, symbol: str, data_type: str = "stock") -> dict:
        try:
            if data_type == "stock":
                data = nse_eq(symbol)
            elif data_type == "fno":
                data = nse_fno(symbol)
            logger.info(f"Retrieved {data_type} data for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching NSE data: {e}")
            return {"error": str(e)}

class MarketNewsTool(BaseTool):
    name = "market_news_tool"
    description = "Search for latest market news and analysis"
    
    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def _run(self, query: str) -> List[str]:
        try:
            results = self.search.run(f"Indian stock market {query} news last 24 hours")
            logger.info(f"Retrieved news for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

class TechnicalAnalysisTool(BaseTool):
    name = "technical_analysis_tool"
    description = "Perform technical analysis on stocks"

    def _run(self, symbol: str, period: str = "1y") -> dict:
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            hist = stock.history(period=period)
            
            # Calculate basic technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            
            logger.info(f"Completed technical analysis for {symbol}")
            return {
                "current_price": hist['Close'][-1],
                "sma_20": hist['SMA_20'][-1],
                "sma_50": hist['SMA_50'][-1],
                "rsi": hist['RSI'][-1],
                "volume": hist['Volume'][-1]
            }
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {"error": str(e)}

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class OptionsChainAnalyzer(BaseTool):
    name = "options_chain_analyzer"
    description = "Analyze options chain for a given symbol"

    def _run(self, symbol: str) -> dict:
        try:
            options_data = nse_fno(symbol)
            
            # Process options chain
            calls = options_data.get('calls', [])
            puts = options_data.get('puts', [])
            
            analysis = {
                'max_pain': self._calculate_max_pain(calls, puts),
                'pcr': self._calculate_pcr(calls, puts),
                'high_oi_strikes': self._get_high_oi_strikes(calls, puts)
            }
            
            logger.info(f"Completed options chain analysis for {symbol}")
            return analysis
        except Exception as e:
            logger.error(f"Error in options chain analysis: {e}")
            return {"error": str(e)}

    def _calculate_max_pain(self, calls, puts):
        # Implementation for max pain calculation
        pass

    def _calculate_pcr(self, calls, puts):
        total_put_oi = sum(put.get('openInterest', 0) for put in puts)
        total_call_oi = sum(call.get('openInterest', 0) for call in calls)
        return total_put_oi / total_call_oi if total_call_oi else 0

    def _get_high_oi_strikes(self, calls, puts):
        # Get strikes with highest open interest
        pass 