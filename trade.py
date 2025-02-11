import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf  # for getting forex data

class ForexBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades_history = []
        self.data = None
        
    def fetch_data(self):
        """Fetch forex data from Yahoo Finance"""
        self.data = yf.download(f'{self.symbol}=X', 
                              start=self.start_date,
                              end=self.end_date,
                              interval='1m')  # 1-minute data for HFT
        return self.data
    
    def calculate_signals(self):
        """Implement your trading strategy here"""
        df = self.data.copy()
        
        # Example indicators (you can modify these based on your strategy)
        df['SMA_fast'] = df['Close'].rolling(window=10).mean()
        df['SMA_slow'] = df['Close'].rolling(window=30).mean()
        
        # Generate buy/sell signals
        df['Signal'] = 0
        df.loc[df['SMA_fast'] > df['SMA_slow'], 'Signal'] = 1  # Buy signal
        df.loc[df['SMA_fast'] < df['SMA_slow'], 'Signal'] = -1  # Sell signal
        
        return df
    
    def execute_backtest(self):
        """Run the backtest"""
        if self.data is None:
            self.fetch_data()
            
        signals_df = self.calculate_signals()
        position = 0
        
        for i in range(1, len(signals_df)):
            current_price = signals_df['Close'].iloc[i]
            signal = signals_df['Signal'].iloc[i]
            
            if signal == 1 and position <= 0:  # Buy
                position = 1
                self.trades_history.append({
                    'timestamp': signals_df.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'capital': self.current_capital
                })
                
            elif signal == -1 and position >= 0:  # Sell
                position = -1
                self.trades_history.append({
                    'timestamp': signals_df.index[i],
                    'type': 'SELL',
                    'price': current_price,
                    'capital': self.current_capital
                })
                
        return pd.DataFrame(self.trades_history)

# Example usage
if __name__ == "__main__":
    # Test with EUR/USD pair
    backtester = ForexBacktester(
        symbol='EURUSD',
        start_date='2024-01-01',
        end_date='2024-03-14',
        initial_capital=10000
    )
    
    results = backtester.execute_backtest()
    print("Trading Results:")
    print(results)
