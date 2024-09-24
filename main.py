from __future__ import division
import math
from helpers import GBS_InputError
from implementations import _gbs, _GBS_Limits, _american_option, _gbs_implied_vol, _american_implied_vol

# Inputs: 
#    option_type = "p" or "c"
#    fs          = price of underlying
#    x           = strike
#    t           = time to expiration
#    v           = implied volatility
#    r           = risk free rate
#    q           = dividend payment
#    b           = cost of carry
# Outputs: 
#    value       = price of the option
#    delta       = first derivative of value with respect to price of underlying
#    gamma       = second derivative of value w.r.t price of underlying
#    theta       = first derivative of value w.r.t. time to expiration
#    vega        = first derivative of value w.r.t. implied volatility
#    rho         = first derivative of value w.r.t. risk free rates

# ---------------------------
# Black Scholes: stock Options (no dividend yield)
def black_scholes(option_type, fs, x, t, r, v):
    b = r
    return _gbs(option_type, fs, x, t, r, b, v)

# ---------------------------
# Merton Model: Stocks Index, stocks with a continuous dividend yields
def merton(option_type, fs, x, t, r, q, v):
    b = r - q
    return _gbs(option_type, fs, x, t, r, b, v)

# ---------------------------
# Commodities
def black_76(option_type, fs, x, t, r, v):
    b = 0
    return _gbs(option_type, fs, x, t, r, b, v)

# ---------------------------
# FX Options
def garman_kohlhagen(option_type, fs, x, t, r, rf, v):
    b = r - rf
    return _gbs(option_type, fs, x, t, r, b, v)

# ---------------------------
# Average Price option on commodities
def asian_76(option_type, fs, x, t, t_a, r, v):
    # Check that TA is reasonable
    if (t_a < _GBS_Limits.MIN_TA) or (t_a > t):
        raise GBS_InputError(
            "Invalid Input Averaging Time (TA = {0}). Acceptable range for inputs is {1} to <T".format(t_a,
                                                                                                       _GBS_Limits.MIN_TA))

    # Approximation to value Asian options on commodities
    b = 0
    if t_a == t:
        # if there is no averaging period, this is just Black Scholes
        v_a = v
    else:
        # Approximate the volatility
        m = (2 * math.exp((v ** 2) * t) - 2 * math.exp((v ** 2) * t_a) * (1 + (v ** 2) * (t - t_a))) / (
            (v ** 4) * ((t - t_a) ** 2))
        v_a = math.sqrt(math.log(m) / t)

    # Finally, have the GBS function do the calculation
    return _gbs(option_type, fs, x, t, r, b, v_a)

# ---------------------------
# Spread Option formula
def kirks_76(option_type, f1, f2, x, t, r, v1, v2, corr):
    # create the modifications to the GBS formula to handle spread options
    b = 0
    fs = f1 / (f2 + x)
    f_temp = f2 / (f2 + x)
    v = math.sqrt((v1 ** 2) + ((v2 * f_temp) ** 2) - (2 * corr * v1 * v2 * f_temp))
    my_values = _gbs(option_type, fs, 1.0, t, r, b, v)

    # Have the GBS function return a value
    return my_values[0] * (f2 + x), 0, 0, 0, 0, 0

# ---------------------------
# American Options (stock style, set q=0 for non-dividend paying options)
def american(option_type, fs, x, t, r, q, v):
    b = r - q
    return _american_option(option_type, fs, x, t, r, b, v)

# ---------------------------
# Commodities
def american_76(option_type, fs, x, t, r, v):
    b = 0
    return _american_option(option_type, fs, x, t, r, b, v)

# Inputs: 
#    option_type = "p" or "c"
#    fs          = price of underlying
#    x           = strike
#    t           = time to expiration
#    v           = implied volatility
#    r           = risk free rate
#    q           = dividend payment
#    b           = cost of carry
# Outputs: 
#    value       = price of the option
#    delta       = first derivative of value with respect to price of underlying
#    gamma       = second derivative of value w.r.t price of underlying
#    theta       = first derivative of value w.r.t. time to expiration
#    vega        = first derivative of value w.r.t. implied volatility
#    rho         = first derivative of value w.r.t. risk free rates

def euro_implied_vol(option_type, fs, x, t, r, q, cp):
    b = r - q
    return _gbs_implied_vol(option_type, fs, x, t, r, b, cp)

def euro_implied_vol_76(option_type, fs, x, t, r, cp):
    b = 0
    return _gbs_implied_vol(option_type, fs, x, t, r, b, cp)

def amer_implied_vol(option_type, fs, x, t, r, q, cp):
    b = r - q
    return _american_implied_vol(option_type, fs, x, t, r, b, cp)

def amer_implied_vol_76(option_type, fs, x, t, r, cp):
    b = 0
    return _american_implied_vol(option_type, fs, x, t, r, b, cp)