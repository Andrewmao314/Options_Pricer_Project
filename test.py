from helpers import assert_close
from main import *
from implementations import _approx_implied_vol, _american_implied_vol, _gbs_implied_vol

print("=====================================")
print("Comparison to 3rd party options pricing provided from ‘Haug, Esper. The Complete Guide to Option Pricing Formulas.’")
print("=====================================")

print("Testing GBS.BlackScholes")
assert_close(black_scholes('c', fs=60, x=65, t=0.25, r=0.08, v=0.30)[0], 2.13336844492)

print("Testing GBS.Merton")
assert_close(merton('p', fs=100, x=95, t=0.5, r=0.10, q=0.05, v=0.20)[0], 2.46478764676)

print("Testing GBS.Black76")
assert_close(black_76('c', fs=19, x=19, t=0.75, r=0.10, v=0.28)[0], 1.70105072524)

print("Testing GBS.garman_kohlhagen")
assert_close(garman_kohlhagen('c', fs=1.56, x=1.60, t=0.5, r=0.06, rf=0.08, v=0.12)[0], 0.0290992531494)

print("Testing Delta")
assert_close(black_76('c', fs=105, x=100, t=0.5, r=0.10, v=0.36)[1], 0.5946287)
assert_close(black_76('p', fs=105, x=100, t=0.5, r=0.10, v=0.36)[1], -0.356601)

print("Testing Gamma")
assert_close(black_scholes('c', fs=55, x=60, t=0.75, r=0.10, v=0.30)[2], 0.0278211604769)
assert_close(black_scholes('p', fs=55, x=60, t=0.75, r=0.10, v=0.30)[2], 0.0278211604769)

print("Testing Theta")
assert_close(merton('p', fs=430, x=405, t=0.0833, r=0.07, q=0.05, v=0.20)[3], -31.1923670565)

print("Testing Vega")
assert_close(black_scholes('c', fs=55, x=60, t=0.75, r=0.10, v=0.30)[4], 18.9357773496)
assert_close(black_scholes('p', fs=55, x=60, t=0.75, r=0.10, v=0.30)[4], 18.9357773496)

print("Testing Rho")
assert_close(black_scholes('c', fs=72, x=75, t=1, r=0.09, v=0.19)[5], 38.7325050173)

print ("testing at-the-money approximation")
assert_close(_approx_implied_vol(option_type="c", fs=100, x=100, t=1, r=.05, b=0, cp=5),0.131757)
assert_close(_approx_implied_vol(option_type="c", fs=59, x=60, t=0.25, r=.067, b=0.067, cp=2.82),0.239753)

# tests for implied volatility
print("Implied Volatilty tests")

print("testing GBS Implied Vol")
assert_close(_gbs_implied_vol('c', 92.45, 107.5, 0.0876712328767123, 0.00192960198828152, 0, 0.162619795863781),0.3)
assert_close(_gbs_implied_vol('c', 93.0766666666667, 107.75, 0.164383561643836, 0.00266390125346286, 0, 0.584588840095316),0.2878)
assert_close(_gbs_implied_vol('c', 93.5333333333333, 107.75, 0.249315068493151, 0.00319934651984034, 0, 1.27026849732877),0.2907)
assert_close(_gbs_implied_vol('c', 93.8733333333333, 107.75, 0.331506849315069, 0.00350934592318849, 0, 1.97015685523537),0.2929)
assert_close(_gbs_implied_vol('c', 94.1166666666667, 107.75, 0.416438356164384, 0.00367360967852615, 0, 2.61731599547608),0.2919)
assert_close(_gbs_implied_vol('p', 94.2666666666667, 107.75, 0.498630136986301, 0.00372609838856132, 0, 16.6074587545269),0.2888)
assert_close(_gbs_implied_vol('p', 94.3666666666667, 107.75, 0.583561643835616, 0.00370681407974257, 0, 17.1686196701434),0.2923)
assert_close(_gbs_implied_vol('p', 94.44, 107.75, 0.668493150684932, 0.00364163303865433, 0, 17.6038273793172),0.2908)
assert_close(_gbs_implied_vol('p', 94.4933333333333, 107.75, 0.750684931506849, 0.00355604221290591, 0, 18.0870982577296),0.2919)
assert_close(_gbs_implied_vol('p', 94.39, 107.75, 0.917808219178082, 0.00337464630758452, 0, 18.9397688539483),0.2876)

print("Testing that GBS implied vol works for integer inputs")
assert_close(_gbs_implied_vol('c', fs=100, x=95, t=1, r=1, b=0, cp=14.6711476484), 1)
assert_close(_gbs_implied_vol('p', fs=100, x=95, t=1, r=1, b=0, cp=12.8317504425), 1)

print("testing American Option implied volatility")
assert_close(_american_implied_vol("p", fs=90, x=100, t=0.5, r=0.1, b=0, cp=10.54), 0.15, tolerance=0.01)
assert_close(_american_implied_vol("p", fs=100, x=100, t=0.5, r=0.1, b=0, cp=6.7661), 0.25, tolerance=0.0001)
assert_close(_american_implied_vol("p", fs=110, x=100, t=0.5, r=0.1, b=0, cp=5.8374), 0.35, tolerance=0.0001)
assert_close(_american_implied_vol('c', fs=42, x=40, t=0.75, r=0.04, b=-0.04, cp=5.28), 0.35, tolerance=0.01)
assert_close(_american_implied_vol('c', fs=90, x=100, t=0.1, r=0.10, b=0, cp=0.02), 0.15, tolerance=0.01)

print("Testing that American implied volatility works for integer inputs")
assert_close(_american_implied_vol('c', fs=100, x=100, t=1, r=0, b=0, cp=13.892), 0.35, tolerance=0.01)
assert_close(_american_implied_vol('p', fs=100, x=100, t=1, r=0, b=0, cp=13.892), 0.35, tolerance=0.01)