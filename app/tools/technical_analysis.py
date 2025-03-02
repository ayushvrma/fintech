import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, List, Optional, Union, Any

from app.schemas.base import (
    PriceData, TechnicalIndicator, AssetRequest, IndicatorResult, AnalysisResult
)

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Tool for performing technical analysis on financial data."""
    
    @staticmethod
    def analyze(
        asset_request: AssetRequest,
        price_data: PriceData,
        indicators: List[TechnicalIndicator],
        parameters: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Perform technical analysis on price data using the specified indicators.
        
        Args:
            asset_request: Information about the requested asset
            price_data: Price data for the asset
            indicators: List of technical indicators to apply
            parameters: Optional parameters for the indicators
            
        Returns:
            AnalysisResult containing the indicators and their values
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'open': price_data.open,
            'high': price_data.high,
            'low': price_data.low,
            'close': price_data.close,
            'volume': price_data.volume if price_data.volume else [None] * len(price_data.timestamp)
        }, index=price_data.timestamp)
        
        indicator_results = []
        
        # Process each requested indicator
        for indicator in indicators:
            try:
                # Get indicator-specific parameters
                ind_params = parameters.get(indicator.value, {}) if parameters else {}
                
                # Calculate indicator
                ind_result = TechnicalAnalyzer._calculate_indicator(df, indicator, ind_params)
                
                indicator_results.append(
                    IndicatorResult(
                        indicator=indicator,
                        values=ind_result,
                        parameters=ind_params
                    )
                )
            except Exception as e:
                logger.error(f"Error calculating {indicator.value}: {str(e)}")
                # Continue with other indicators rather than failing completely
        
        # Create the analysis result
        result = AnalysisResult(
            asset_info=asset_request,
            price_data=price_data,
            indicators=indicator_results
        )
        
        return result
    
    @staticmethod
    def _calculate_indicator(
        df: pd.DataFrame, 
        indicator: TechnicalIndicator, 
        parameters: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Calculate a specific technical indicator."""
        if indicator == TechnicalIndicator.RSI:
            return TechnicalAnalyzer._calculate_rsi(df, parameters)
        elif indicator == TechnicalIndicator.MACD:
            return TechnicalAnalyzer._calculate_macd(df, parameters)
        elif indicator == TechnicalIndicator.BOLLINGER_BANDS:
            return TechnicalAnalyzer._calculate_bollinger_bands(df, parameters)
        elif indicator == TechnicalIndicator.MOVING_AVERAGE:
            return TechnicalAnalyzer._calculate_moving_averages(df, parameters)
        elif indicator == TechnicalIndicator.SUPPORT_RESISTANCE:
            return TechnicalAnalyzer._calculate_support_resistance(df, parameters)
        elif indicator == TechnicalIndicator.VOLUME:
            return TechnicalAnalyzer._calculate_volume_indicators(df, parameters)
        elif indicator == TechnicalIndicator.ATR:
            return TechnicalAnalyzer._calculate_atr(df, parameters)
        elif indicator == TechnicalIndicator.STOCHASTIC:
            return TechnicalAnalyzer._calculate_stochastic(df, parameters)
        elif indicator == TechnicalIndicator.ICHIMOKU:
            return TechnicalAnalyzer._calculate_ichimoku(df, parameters)
        else:
            raise ValueError(f"Unsupported indicator: {indicator.value}")
    
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Relative Strength Index (RSI)."""
        length = parameters.get('length', 14)
        
        # Calculate RSI using pandas_ta
        rsi = ta.rsi(df['close'], length=length)
        
        # Fill NaN values with forward fill then backward fill
        rsi = rsi.fillna(method='ffill').fillna(method='bfill')
        
        return {
            'rsi': rsi.tolist()
        }
    
    @staticmethod
    def _calculate_macd(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        fast = parameters.get('fast', 12)
        slow = parameters.get('slow', 26)
        signal = parameters.get('signal', 9)
        
        # Calculate MACD using pandas_ta
        macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        
        # Extract components
        macd_line = macd['MACD_' + str(fast) + '_' + str(slow) + '_' + str(signal)]
        signal_line = macd['MACDs_' + str(fast) + '_' + str(slow) + '_' + str(signal)]
        histogram = macd['MACDh_' + str(fast) + '_' + str(slow) + '_' + str(signal)]
        
        # Fill NaN values
        macd_line = macd_line.fillna(method='ffill').fillna(method='bfill')
        signal_line = signal_line.fillna(method='ffill').fillna(method='bfill')
        histogram = histogram.fillna(method='ffill').fillna(method='bfill')
        
        return {
            'macd': macd_line.tolist(),
            'signal': signal_line.tolist(),
            'histogram': histogram.tolist()
        }
    
    @staticmethod
    def _calculate_bollinger_bands(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands."""
        length = parameters.get('length', 20)
        std_dev = parameters.get('std_dev', 2)
        
        # Calculate Bollinger Bands using pandas_ta
        bbands = ta.bbands(df['close'], length=length, std=std_dev)
        
        # Extract components
        lower = bbands['BBL_' + str(length) + '_' + str(std_dev)]
        middle = bbands['BBM_' + str(length) + '_' + str(std_dev)]
        upper = bbands['BBU_' + str(length) + '_' + str(std_dev)]
        
        # Fill NaN values
        lower = lower.fillna(method='ffill').fillna(method='bfill')
        middle = middle.fillna(method='ffill').fillna(method='bfill')
        upper = upper.fillna(method='ffill').fillna(method='bfill')
        
        return {
            'lower': lower.tolist(),
            'middle': middle.tolist(),
            'upper': upper.tolist()
        }
    
    @staticmethod
    def _calculate_moving_averages(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate various Moving Averages."""
        result = {}
        
        # Simple Moving Average (SMA)
        sma_periods = parameters.get('sma_periods', [20, 50, 200])
        if not isinstance(sma_periods, list):
            sma_periods = [sma_periods]
            
        for period in sma_periods:
            sma = ta.sma(df['close'], length=period)
            sma = sma.fillna(method='ffill').fillna(method='bfill')
            result[f'sma_{period}'] = sma.tolist()
        
        # Exponential Moving Average (EMA)
        ema_periods = parameters.get('ema_periods', [12, 26])
        if not isinstance(ema_periods, list):
            ema_periods = [ema_periods]
            
        for period in ema_periods:
            ema = ta.ema(df['close'], length=period)
            ema = ema.fillna(method='ffill').fillna(method='bfill')
            result[f'ema_{period}'] = ema.tolist()
        
        return result
    
    @staticmethod
    def _calculate_support_resistance(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate support and resistance levels."""
        window = parameters.get('window', 10)
        
        # Simple implementation of support and resistance using local minima and maxima
        close = df['close']
        
        # Find local maxima and minima
        local_max = []
        local_min = []
        
        for i in range(window, len(close) - window):
            if all(close[i] > close[i-j] for j in range(1, window + 1)) and all(close[i] > close[i+j] for j in range(1, window + 1)):
                local_max.append((i, close[i]))
            elif all(close[i] < close[i-j] for j in range(1, window + 1)) and all(close[i] < close[i+j] for j in range(1, window + 1)):
                local_min.append((i, close[i]))
        
        # Get the most recent levels
        max_levels = sorted([price for _, price in local_max], reverse=True)[:5]
        min_levels = sorted([price for _, price in local_min])[:5]
        
        # Create arrays for plotting
        resistance = [None] * len(close)
        support = [None] * len(close)
        
        # Fill in the arrays with the support and resistance levels
        for i in range(len(close)):
            # For each price point, find the closest resistance level above
            closest_resistance = None
            for level in max_levels:
                if level > close[i]:
                    closest_resistance = level
                    break
            
            # For each price point, find the closest support level below
            closest_support = None
            for level in reversed(min_levels):
                if level < close[i]:
                    closest_support = level
                    break
            
            resistance[i] = closest_resistance
            support[i] = closest_support
        
        return {
            'resistance_levels': max_levels,
            'support_levels': min_levels,
            'resistance': resistance,
            'support': support
        }
    
    @staticmethod
    def _calculate_volume_indicators(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate volume-based indicators."""
        result = {}
        
        # Ensure we have volume data
        if 'volume' not in df.columns or df['volume'].isna().all():
            return {'volume': [0] * len(df)}
        
        # Simple volume
        result['volume'] = df['volume'].fillna(0).tolist()
        
        # Volume Moving Average
        volume_ma_period = parameters.get('volume_ma_period', 20)
        volume_ma = ta.sma(df['volume'], length=volume_ma_period)
        result['volume_ma'] = volume_ma.fillna(method='ffill').fillna(0).tolist()
        
        # Volume Weighted Average Price (VWAP)
        try:
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            result['vwap'] = vwap.fillna(method='ffill').fillna(method='bfill').tolist()
        except:
            # VWAP might fail if there are NaN values in volume
            pass
        
        # On-Balance Volume (OBV)
        try:
            obv = ta.obv(df['close'], df['volume'])
            result['obv'] = obv.fillna(method='ffill').fillna(0).tolist()
        except:
            pass
        
        return result
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Average True Range (ATR)."""
        length = parameters.get('length', 14)
        
        # Calculate ATR using pandas_ta
        atr = ta.atr(df['high'], df['low'], df['close'], length=length)
        
        # Fill NaN values
        atr = atr.fillna(method='ffill').fillna(method='bfill')
        
        return {
            'atr': atr.tolist()
        }
    
    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Stochastic Oscillator."""
        k_period = parameters.get('k_period', 14)
        d_period = parameters.get('d_period', 3)
        
        # Calculate Stochastic using pandas_ta
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period)
        
        # Extract components
        stoch_k = stoch['STOCHk_' + str(k_period) + '_' + str(d_period) + '_3']
        stoch_d = stoch['STOCHd_' + str(k_period) + '_' + str(d_period) + '_3']
        
        # Fill NaN values
        stoch_k = stoch_k.fillna(method='ffill').fillna(method='bfill')
        stoch_d = stoch_d.fillna(method='ffill').fillna(method='bfill')
        
        return {
            'k': stoch_k.tolist(),
            'd': stoch_d.tolist()
        }
    
    @staticmethod
    def _calculate_ichimoku(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, List[float]]:
        """Calculate Ichimoku Cloud."""
        tenkan_period = parameters.get('tenkan', 9)
        kijun_period = parameters.get('kijun', 26)
        senkou_span_b_period = parameters.get('senkou_span_b', 52)
        
        # Calculate Ichimoku using pandas_ta
        ichimoku = ta.ichimoku(
            df['high'], 
            df['low'], 
            tenkan=tenkan_period, 
            kijun=kijun_period, 
            senkou=senkou_span_b_period
        )
        
        # Extract components
        tenkan = ichimoku['ITS_' + str(tenkan_period)]
        kijun = ichimoku['IKS_' + str(kijun_period)]
        senkou_a = ichimoku['ISA_' + str(tenkan_period)]
        senkou_b = ichimoku['ISB_' + str(senkou_span_b_period)]
        chikou = ichimoku['ICS_' + str(tenkan_period)]
        
        # Fill NaN values
        tenkan = tenkan.fillna(method='ffill').fillna(method='bfill')
        kijun = kijun.fillna(method='ffill').fillna(method='bfill')
        senkou_a = senkou_a.fillna(method='ffill').fillna(method='bfill')
        senkou_b = senkou_b.fillna(method='ffill').fillna(method='bfill')
        chikou = chikou.fillna(method='ffill').fillna(method='bfill')
        
        return {
            'tenkan': tenkan.tolist(),
            'kijun': kijun.tolist(),
            'senkou_a': senkou_a.tolist(),
            'senkou_b': senkou_b.tolist(),
            'chikou': chikou.tolist()
        } 