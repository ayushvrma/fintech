import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import locale
import time
# Set locale for currency formatting (e.g., USD)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price of a European call option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_delta(S, K, T, r, sigma):
    """
    Calculate the delta of a European call option using the Black-Scholes model.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def black_scholes_vega(S, K, T, r, sigma):
    """
    Calculate the Vega of a European call option using the Black-Scholes model.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def implied_volatility(S, K, T, r, market_price, initial_vol=0.2, tol=1e-5, max_iter=100):
    """
    Calculate implied volatility using the Newton-Raphson method.
    """
    sigma = initial_vol
    for i in range(max_iter):
        model_price = black_scholes_call(S, K, T, r, sigma)
        vega = black_scholes_vega(S, K, T, r, sigma)
        price_diff = model_price - market_price
        
        if abs(price_diff) < tol:
            return sigma  # Converged
        
        sigma -= price_diff / vega  # Newton-Raphson step
    
    raise ValueError("Implied volatility did not converge")

def black_scholes_gamma(S, K, T, r, sigma):
    """
    Calculate the gamma of a European call option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_gamma_hedge(portfolio_gamma, option_gamma, option_price):
    """
    Calculate the number of additional options needed to neutralize gamma.
    """
    return -portfolio_gamma / option_gamma  # Add enough options to offset portfolio gamma

# Example usage

def format_currency(value):
    """Helper function to format values as currency"""
    return locale.currency(value, grouping=True)

def get_real_time_data(symbol):
    """
    Fetch real-time stock data using yfinance
    
    Parameters:
        symbol (str): Stock symbol (e.g., 'AAPL')
    
    Returns:
        tuple: Current price, historical volatility
    """
    stock = yf.Ticker(symbol)
    current_price = stock.info['currentPrice']
    
    # Calculate historical volatility using past 252 trading days
    hist_data = stock.history(period='1y')
    returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    hist_volatility = returns.std() * np.sqrt(252)
    
    return current_price, hist_volatility

def simulate_real_time_hedging(symbol, K, T, r, rebalance_interval=5):
    """
    Simulate dynamic hedging with real-time market data
    
    Parameters:
        symbol (str): Stock symbol
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free rate (annualized)
        rebalance_interval (int): Seconds between rebalancing
    """
    while True:
        try:
            S0, sigma = get_real_time_data(symbol)
            option_price = black_scholes_call(S0, K, T, r, sigma)
            delta = black_scholes_delta(S0, K, T, r, sigma)
            
            print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Stock Price: {format_currency(S0)}")
            print(f"Option Price: {format_currency(option_price)}")
            print(f"Delta: {delta:.4f}")
            print(f"Hedge Position Value: {format_currency(delta * S0)}")
            
            time.sleep(rebalance_interval)
            
        except KeyboardInterrupt:
            print("\nStopping real-time simulation...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(rebalance_interval)

# Example usage for real-time simulation
if __name__ == "__main__":
    symbol = "AAPL"  # Example using Apple stock
    K = 170.0        # Strike price
    T = 30/365      # 30 days to expiration
    r = 0.05        # Risk-free rate (5%)
    
    simulate_real_time_hedging(symbol, K, T, r)
