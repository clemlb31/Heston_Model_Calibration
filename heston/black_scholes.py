import numpy as np
from scipy.stats import norm


class BlackScholes:
    """Formules de Black-Scholes pour calls europeens."""

    @staticmethod
    def price(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def vega(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

    @staticmethod
    def implied_vol(price, S, K, T, r, q, tol=1e-8, max_iter=100):
        """Volatilite implicite par Newton-Raphson."""
        sigma = 0.3
        for _ in range(max_iter):
            bs_price = BlackScholes.price(S, K, T, r, q, sigma)
            v = BlackScholes.vega(S, K, T, r, q, sigma)
            if v < 1e-12:
                break
            sigma -= (bs_price - price) / v
            sigma = max(sigma, 1e-6)
            if abs(bs_price - price) < tol:
                break
        return sigma
