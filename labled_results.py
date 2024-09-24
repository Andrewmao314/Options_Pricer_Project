from main import *
from stock_fetch import fetch_stock_data, fetch_option_data
from datetime import datetime
# Given a function name and the results from inputs, prints the results with labels so
# that they are easier to understand
def print_labeled_results(function_name, result):
    print(f"\n{function_name} Results:")
    print(f"Value: {result[0]:.8f}")
    print(f"Delta: {result[1]:.8f}")
    print(f"Gamma: {result[2]:.8f}")
    print(f"Theta: {result[3]:.8f}")
    print(f"Vega: {result[4]:.8f}")
    print(f"Rho: {result[5]:.8f}")

def labeled_black_scholes(option_type, fs, x, t, r, v):
    result = black_scholes(option_type, fs, x, t, r, v)
    print_labeled_results("Black-Scholes", result)

def labeled_merton(option_type, fs, x, t, r, q, v):
    result = merton(option_type, fs, x, t, r, q, v)
    print_labeled_results("Merton", result)

def labeled_black_76(option_type, fs, x, t, r, v):
    result = black_76(option_type, fs, x, t, r, v)
    print_labeled_results("Black '76", result)

def labeled_garman_kohlhagen(option_type, fs, x, t, r, rf, v):
    result = garman_kohlhagen(option_type, fs, x, t, r, rf, v)
    print_labeled_results("Garman-Kohlhagen", result)

def labeled_asian_76(option_type, fs, x, t, t_a, r, v):
    result = asian_76(option_type, fs, x, t, t_a, r, v)
    print_labeled_results("Asian '76", result)

def labeled_kirks_76(option_type, f1, f2, x, t, r, v1, v2, corr):
    result = kirks_76(option_type, f1, f2, x, t, r, v1, v2, corr)
    print_labeled_results("Kirk's 76", result)

def labeled_american(option_type, fs, x, t, r, q, v):
    result = american(option_type, fs, x, t, r, q, v)
    print_labeled_results("American", result)

def labeled_american_76(option_type, fs, x, t, r, v):
    result = american_76(option_type, fs, x, t, r, v)
    print_labeled_results("American '76", result)

def get_closest_option(options, target_strike):
    """Find the option with the strike price closest to the target strike."""
    return options.iloc[(options['strike'] - target_strike).abs().argsort()[:1]]

def labeled_black_scholes_real_time(ticker, option_type, strike, time_to_expiry):
    """
    Price an option using real-time data and the Black-Scholes model.
    
    Args:
    ticker (str): The stock ticker symbol
    option_type (str): 'c' for call, 'p' for put
    strike (float): The option strike price
    time_to_expiry (float): Time to expiration in years
    """
    stock_data = fetch_stock_data(ticker)
    calls, puts, closest_expiry = fetch_option_data(ticker, time_to_expiry)
    
    # Get the closest option to the target strike
    if option_type == 'c':
        option = get_closest_option(calls, strike)
    else:
        option = get_closest_option(puts, strike)
    
    # Calculate implied volatility
    implied_vol = euro_implied_vol(
        option_type,
        fs=stock_data['current_price'],
        x=option['strike'].values[0],
        t=time_to_expiry,
        r=stock_data['risk_free_rate'],
        q=0,  # Assuming no dividend for simplicity
        cp=option['lastPrice'].values[0]
    )
    
    # Price the option using Black-Scholes
    result = black_scholes(
        option_type,
        fs=stock_data['current_price'],
        x=strike,
        t=time_to_expiry,
        r=stock_data['risk_free_rate'],
        v=implied_vol
    )
    
    print(f"\nReal-time Black-Scholes Results for {ticker}:")
    print(f"Current Stock Price: ${stock_data['current_price']:.2f}")
    print(f"Risk-free Rate: {stock_data['risk_free_rate']:.4f}")
    print(f"Implied Volatility: {implied_vol:.4f}")
    print(f"Time to Expiry: {time_to_expiry:.4f} years")
    print(f"Closest Available Expiration Date: {closest_expiry}")
    print_labeled_results("Black-Scholes", result)

# Example usage
if __name__ == "__main__":
    print("Example usage of labeled results:")
    labeled_black_scholes('c', fs=60, x=65, t=0.25, r=0.08, v=0.30)
    labeled_merton('p', fs=100, x=95, t=0.5, r=0.10, q=0.05, v=0.20)
    labeled_black_76('c', fs=19, x=19, t=0.75, r=0.10, v=0.28)
    labeled_garman_kohlhagen('c', fs=1.56, x=1.60, t=0.5, r=0.06, rf=0.08, v=0.12)
    print("Example usage of labeled results with real-time data:")
    labeled_black_scholes_real_time('AAPL', 'c', 150, 0.5)  # Example: AAPL call option, strike $150, 6 months to expiry