import numpy as np
import scipy.stats as st
from scipy.optimize import newton


@np.vectorize
def implied_vol(forward_price, time_to_maturity, strike, option_price, call=True, discount_factor=1.0):
    x = np.log(forward_price / strike)
    p = option_price / (discount_factor * np.sqrt(forward_price * strike))

    if np.isclose(x, 0):
        return -2 * st.norm.ppf((1 - p) / 2) / np.sqrt(time_to_maturity)

    theta = 1 if call else -1
    lower_bound = theta * (np.exp(x / 2) - np.exp(-x / 2)) if theta * x > 0 else 0.0
    upper_bound = np.exp(theta * x / 2)

    if p <= lower_bound or p >= upper_bound:
        return np.nan

    sigma0 = np.sqrt(2 * abs(x))

    def equation(sigma):
        return theta * (
            np.exp(x / 2) * st.norm.cdf(theta * (x / sigma + sigma / 2))
            - np.exp(-x / 2) * st.norm.cdf(theta * (x / sigma - sigma / 2))
        ) - p

    try:
        sigma = newton(equation, sigma0)
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        return np.nan

    return sigma / np.sqrt(time_to_maturity)


def normalized_black_price(T, moneyness, sigma, call):
    total_volatility = sigma * np.sqrt(T)
    d1 = -np.log(moneyness) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility

    call_price = st.norm.cdf(d1) - moneyness * st.norm.cdf(d2)
    put_price = moneyness * st.norm.cdf(-d2) - st.norm.cdf(-d1)

    return np.where(call, call_price, put_price)


def normalized_call_from_total_variance(log_moneyness, total_variance):
    sqrt_w = np.sqrt(total_variance)
    d1 = -log_moneyness / sqrt_w + 0.5 * sqrt_w
    d2 = d1 - sqrt_w

    return st.norm.cdf(d1) - np.exp(log_moneyness) * st.norm.cdf(d2)