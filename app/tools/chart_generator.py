import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union

from app.schemas.base import (
    AssetRequest, PriceData, ChartData, DataSource, TechnicalIndicator
)

logger = logging.getLogger(__name__)

# Create directory for saving charts if it doesn't exist
CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)


class ChartGenerator:
    """Generator for financial charts using different visualization libraries."""
    
    @staticmethod
    def generate_chart(
        asset_request: AssetRequest,
        price_data: PriceData,
        data_source: DataSource,
        indicators: Optional[Dict[str, Dict[str, List[float]]]] = None,
        chart_type: str = "candlestick",
        platform: str = "plotly"
    ) -> ChartData:
        """
        Generate a chart for the given asset and price data.
        
        Args:
            asset_request: Asset request details
            price_data: Price data for the asset
            data_source: Source of the data
            indicators: Optional technical indicators to include on the chart
            chart_type: Type of chart (candlestick, line, ohlc)
            platform: Visualization platform to use (matplotlib, plotly)
            
        Returns:
            ChartData object with chart paths
        """
        if platform.lower() == "matplotlib":
            chart_path, _ = ChartGenerator._generate_matplotlib_chart(
                asset_request, price_data, indicators, chart_type
            )
            return ChartData(
                asset_info=asset_request,
                price_data=price_data,
                chart_path=chart_path,
                data_source=data_source
            )
        elif platform.lower() == "plotly":
            chart_path, chart_html = ChartGenerator._generate_plotly_chart(
                asset_request, price_data, indicators, chart_type
            )
            return ChartData(
                asset_info=asset_request,
                price_data=price_data,
                chart_path=chart_path,
                chart_html=chart_html,
                data_source=data_source
            )
        else:
            raise ValueError(f"Unsupported chart platform: {platform}")
    
    @staticmethod
    def _generate_matplotlib_chart(
        asset_request: AssetRequest,
        price_data: PriceData,
        indicators: Optional[Dict[str, Dict[str, List[float]]]] = None,
        chart_type: str = "candlestick"
    ) -> Tuple[str, None]:
        """Generate a chart using Matplotlib."""
        symbol = asset_request.symbol
        asset_type = asset_request.asset_type
        timeframe = asset_request.timeframe
        
        # Convert data to DataFrame for easier manipulation
        df = pd.DataFrame({
            'open': price_data.open,
            'high': price_data.high,
            'low': price_data.low,
            'close': price_data.close,
            'volume': price_data.volume if price_data.volume else [0] * len(price_data.timestamp)
        }, index=price_data.timestamp)
        
        # Determine the number of subplots needed
        n_subplots = 1
        if indicators:
            # Add a subplot for indicators that need separate plots (e.g., volume, RSI)
            separate_indicators = ['volume', 'rsi', 'stochastic']
            for ind_name in indicators.keys():
                if ind_name.lower() in separate_indicators:
                    n_subplots += 1
        
        # Create figure with subplots
        if n_subplots > 1:
            fig, axs = plt.subplots(n_subplots, 1, figsize=(12, 8 * n_subplots), 
                                    gridspec_kw={'height_ratios': [3] + [1] * (n_subplots - 1)}, 
                                    sharex=True)
        else:
            fig, axs = plt.subplots(1, 1, figsize=(12, 8))
            axs = [axs]  # Make it a list for consistent indexing
        
        # Main price chart
        ax = axs[0]
        
        if chart_type.lower() == "candlestick":
            # Plot candlestick chart
            width = 0.6
            width2 = 0.1
            
            up = df[df.close >= df.open]
            down = df[df.close < df.open]
            
            # Plot up candles
            ax.bar(up.index, up.high - up.low, width=width2, bottom=up.low, color='green', alpha=0.5)
            ax.bar(up.index, up.close - up.open, width=width, bottom=up.open, color='green')
            
            # Plot down candles
            ax.bar(down.index, down.high - down.low, width=width2, bottom=down.low, color='red', alpha=0.5)
            ax.bar(down.index, down.open - down.close, width=width, bottom=down.close, color='red')
            
        elif chart_type.lower() == "line":
            # Plot line chart
            ax.plot(df.index, df.close, label='Close Price', color='blue')
            
        elif chart_type.lower() == "ohlc":
            # Plot OHLC chart (simplified version)
            for i, (idx, row) in enumerate(df.iterrows()):
                ax.plot([idx, idx], [row.low, row.high], color='black', linewidth=1)
                ax.plot([idx, idx - 0.4], [row.open, row.open], color='black', linewidth=1)
                ax.plot([idx, idx + 0.4], [row.close, row.close], color='black', linewidth=1)
                
        # Set title and labels
        title = f"{symbol} ({asset_type.value.upper()}) - {timeframe.value} Timeframe"
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(df) > 30:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add indicators that go on the main price chart
        subplot_idx = 1
        if indicators:
            for ind_name, ind_values in indicators.items():
                if ind_name.lower() == 'bollinger_bands':
                    if 'middle' in ind_values and 'upper' in ind_values and 'lower' in ind_values:
                        ax.plot(df.index, ind_values['middle'], label='BB Middle', color='orange', linewidth=1)
                        ax.plot(df.index, ind_values['upper'], label='BB Upper', color='orange', linestyle='--', linewidth=1)
                        ax.plot(df.index, ind_values['lower'], label='BB Lower', color='orange', linestyle='--', linewidth=1)
                        ax.fill_between(df.index, ind_values['upper'], ind_values['lower'], color='orange', alpha=0.1)
                
                elif ind_name.lower() == 'moving_average':
                    for ma_name, ma_values in ind_values.items():
                        color = 'purple' if 'sma' in ma_name.lower() else 'blue'
                        ax.plot(df.index, ma_values, label=ma_name.upper(), color=color, linewidth=1)
                
                elif ind_name.lower() == 'macd':
                    # MACD gets its own subplot
                    macd_ax = axs[subplot_idx]
                    if 'macd' in ind_values and 'signal' in ind_values:
                        macd_ax.plot(df.index, ind_values['macd'], label='MACD', color='blue', linewidth=1)
                        macd_ax.plot(df.index, ind_values['signal'], label='Signal', color='red', linewidth=1)
                        if 'histogram' in ind_values:
                            macd_ax.bar(df.index, ind_values['histogram'], label='Histogram', color='green', alpha=0.5)
                    macd_ax.set_ylabel('MACD', fontsize=12)
                    macd_ax.legend(loc='upper left')
                    macd_ax.grid(True, alpha=0.3)
                    subplot_idx += 1
                
                elif ind_name.lower() == 'rsi':
                    # RSI gets its own subplot
                    rsi_ax = axs[subplot_idx]
                    if 'rsi' in ind_values:
                        rsi_ax.plot(df.index, ind_values['rsi'], label='RSI', color='purple', linewidth=1)
                        rsi_ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                        rsi_ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                        rsi_ax.fill_between(df.index, ind_values['rsi'], 70, where=(np.array(ind_values['rsi']) >= 70), color='red', alpha=0.3)
                        rsi_ax.fill_between(df.index, ind_values['rsi'], 30, where=(np.array(ind_values['rsi']) <= 30), color='green', alpha=0.3)
                    rsi_ax.set_ylabel('RSI', fontsize=12)
                    rsi_ax.set_ylim(0, 100)
                    rsi_ax.legend(loc='upper left')
                    rsi_ax.grid(True, alpha=0.3)
                    subplot_idx += 1
                
                elif ind_name.lower() == 'volume':
                    # Volume gets its own subplot
                    vol_ax = axs[subplot_idx]
                    vol_ax.bar(df.index, df.volume, label='Volume', color='blue', alpha=0.5)
                    vol_ax.set_ylabel('Volume', fontsize=12)
                    vol_ax.legend(loc='upper left')
                    vol_ax.grid(True, alpha=0.3)
                    
                    # Format y-axis to show volume in millions/billions
                    def millions(x, pos):
                        return f'{x/1000000:.1f}M'
                    vol_ax.yaxis.set_major_formatter(FuncFormatter(millions))
                    
                    subplot_idx += 1
        
        # Add legend to main chart
        ax.legend(loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Generate unique filename
        filename = f"{symbol}_{asset_type.value}_{timeframe.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(CHART_DIR, filename)
        
        plt.savefig(file_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return file_path, None
    
    @staticmethod
    def _generate_plotly_chart(
        asset_request: AssetRequest,
        price_data: PriceData,
        indicators: Optional[Dict[str, Dict[str, List[float]]]] = None,
        chart_type: str = "candlestick"
    ) -> Tuple[str, str]:
        """Generate a chart using Plotly."""
        symbol = asset_request.symbol
        asset_type = asset_request.asset_type
        timeframe = asset_request.timeframe
        
        # Convert data to DataFrame for easier manipulation
        df = pd.DataFrame({
            'open': price_data.open,
            'high': price_data.high,
            'low': price_data.low,
            'close': price_data.close,
            'volume': price_data.volume if price_data.volume else [0] * len(price_data.timestamp)
        }, index=price_data.timestamp)
        
        # Determine the number of subplots needed
        n_indicators = 0
        indicator_names = []
        if indicators:
            for ind_name in indicators.keys():
                if ind_name.lower() in ['volume', 'rsi', 'stochastic', 'macd']:
                    n_indicators += 1
                    indicator_names.append(ind_name.lower())
        
        # Create figure with subplots
        if n_indicators > 0:
            fig = make_subplots(
                rows=n_indicators + 1, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f"{symbol} Price"] + indicator_names,
                row_heights=[0.6] + [0.4/n_indicators] * n_indicators
            )
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} Price"])
        
        # Main price chart
        if chart_type.lower() == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df.open,
                    high=df.high,
                    low=df.low,
                    close=df.close,
                    name="Price"
                ),
                row=1, col=1
            )
            
        elif chart_type.lower() == "line":
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.close,
                    mode='lines',
                    name="Close Price",
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
        elif chart_type.lower() == "ohlc":
            fig.add_trace(
                go.Ohlc(
                    x=df.index,
                    open=df.open,
                    high=df.high,
                    low=df.low,
                    close=df.close,
                    name="Price"
                ),
                row=1, col=1
            )
        
        # Add indicators
        subplot_idx = 2  # Start additional indicators from row 2
        if indicators:
            for ind_name, ind_values in indicators.items():
                if ind_name.lower() == 'bollinger_bands':
                    if 'middle' in ind_values and 'upper' in ind_values and 'lower' in ind_values:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ind_values['middle'],
                                mode='lines',
                                name="BB Middle",
                                line=dict(color='orange', width=1)
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ind_values['upper'],
                                mode='lines',
                                name="BB Upper",
                                line=dict(color='orange', width=1, dash='dash')
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ind_values['lower'],
                                mode='lines',
                                name="BB Lower",
                                line=dict(color='orange', width=1, dash='dash'),
                                fill='tonexty',
                                fillcolor='rgba(255,165,0,0.1)'
                            ),
                            row=1, col=1
                        )
                
                elif ind_name.lower() == 'moving_average':
                    for ma_name, ma_values in ind_values.items():
                        color = 'purple' if 'sma' in ma_name.lower() else 'blue'
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ma_values,
                                mode='lines',
                                name=ma_name.upper(),
                                line=dict(color=color, width=1)
                            ),
                            row=1, col=1
                        )
                
                elif ind_name.lower() == 'macd':
                    if 'macd' in ind_values and 'signal' in ind_values:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ind_values['macd'],
                                mode='lines',
                                name="MACD",
                                line=dict(color='blue', width=1)
                            ),
                            row=subplot_idx, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ind_values['signal'],
                                mode='lines',
                                name="Signal",
                                line=dict(color='red', width=1)
                            ),
                            row=subplot_idx, col=1
                        )
                        if 'histogram' in ind_values:
                            colors = ['green' if val >= 0 else 'red' for val in ind_values['histogram']]
                            fig.add_trace(
                                go.Bar(
                                    x=df.index,
                                    y=ind_values['histogram'],
                                    name="Histogram",
                                    marker=dict(color=colors)
                                ),
                                row=subplot_idx, col=1
                            )
                    subplot_idx += 1
                
                elif ind_name.lower() == 'rsi':
                    if 'rsi' in ind_values:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=ind_values['rsi'],
                                mode='lines',
                                name="RSI",
                                line=dict(color='purple', width=1)
                            ),
                            row=subplot_idx, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[0], df.index[-1]],
                                y=[70, 70],
                                mode='lines',
                                name="Overbought",
                                line=dict(color='red', width=1, dash='dash')
                            ),
                            row=subplot_idx, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[0], df.index[-1]],
                                y=[30, 30],
                                mode='lines',
                                name="Oversold",
                                line=dict(color='green', width=1, dash='dash'),
                            ),
                            row=subplot_idx, col=1
                        )
                    # Update y-axis range for RSI
                    fig.update_yaxes(range=[0, 100], row=subplot_idx, col=1)
                    subplot_idx += 1
                
                elif ind_name.lower() == 'volume':
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df.volume,
                            name="Volume",
                            marker=dict(color='blue', opacity=0.5)
                        ),
                        row=subplot_idx, col=1
                    )
                    subplot_idx += 1
        
        # Update layout and appearance
        fig.update_layout(
            title=f"{symbol} ({asset_type.value.upper()}) - {timeframe.value} Timeframe",
            xaxis_rangeslider_visible=False,
            height=600 + (300 * n_indicators),
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Update price axis format 
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        # Generate unique filename
        filename = f"{symbol}_{asset_type.value}_{timeframe.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Save static image
        img_path = os.path.join(CHART_DIR, f"{filename}.png")
        fig.write_image(img_path)
        
        # Save HTML file for interactive display
        html_path = os.path.join(CHART_DIR, f"{filename}.html")
        fig.write_html(html_path)
        
        return img_path, html_path 