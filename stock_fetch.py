import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(ticker):
    """
    Fetch stock data for a given ticker.
    
    Args:
    ticker (str): The stock ticker symbol
    
    Returns:
    dict: A dictionary containing current stock price and risk-free rate
    """
    # Fetch stock data
    stock = yf.Ticker(ticker)
    
    # Get current stock price
    current_price = stock.history(period="1d")['Close'].iloc[-1]
    
    # For simplicity, we'll use the US 10-year Treasury yield as the risk-free rate
    risk_free_rate = yf.Ticker("^TNX").info['previousClose'] / 100  # Convert to decimal
    
    return {
        "current_price": current_price,
        "risk_free_rate": risk_free_rate
    }

def fetch_option_data(ticker, time_to_expiry):
    """
    Fetch option chain data for a given ticker and approximate time to expiry.
    
    Args:
    ticker (str): The stock ticker symbol
    time_to_expiry (float): Time to expiration in years
    
    Returns:
    tuple: Two DataFrames containing call and put option data
    """
    stock = yf.Ticker(ticker)
    
    # Get all available expiration dates
    expirations = stock.options
    
    # Find the expiration date closest to the given time to expiry
    target_date = datetime.now() + timedelta(days=int(time_to_expiry * 365))
    closest_expiry = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
    
    # Fetch option chain for the closest expiration date
    options = stock.option_chain(closest_expiry)
    return options.calls, options.puts, closest_expiry