"""
Analytical Merton jump diffusion option pricing.

Merton (1976) showed that European option prices under jump diffusion
can be expressed as an infinite series of Black-Scholes prices:

    V = sum_{n=0}^{inf} P(N=n) * BS(S, K, T, r_n, sigma_n)

where:
    P(N=n) = exp(-lambda'*T) * (lambda'*T)^n / n!
    lambda' = lambda * (1 + k)
    r_n = r - lambda*k + n*ln(1+k)/T
    sigma_n = sqrt(sigma^2 + n*sigma_J^2/T)

This converges rapidly and serves as ground truth for validating the PINN.
"""

import math
from typing import Optional

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes European call price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Call option price
    """
    if T <= 1e-12:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K * math.exp(-r * T), 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call


def black_scholes_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Black-Scholes European put price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Put option price
    """
    if T <= 1e-12:
        return max(K - S, 0.0)
    if sigma <= 1e-12:
        return max(K * math.exp(-r * T) - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put


def merton_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    n_terms: int = 50,
) -> float:
    """
    Merton jump diffusion European call price via series expansion.

    The price is an infinite sum of Black-Scholes prices weighted by
    Poisson probabilities:

        V = sum_{n=0}^{N} P(n) * BS(S, K, T, r_n, sigma_n)

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Diffusion volatility
        lambda_j: Jump intensity (jumps/year)
        mu_j: Mean log-jump size
        sigma_j: Jump size volatility
        n_terms: Number of terms in the series (default 50)

    Returns:
        Call option price under Merton's jump diffusion
    """
    if T <= 1e-12:
        return max(S - K, 0.0)

    # Expected percentage jump size
    k = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    # Adjusted intensity
    lambda_prime = lambda_j * (1.0 + k)

    price = 0.0
    log_poisson = -lambda_prime * T  # Start with log of P(N=0)

    for n in range(n_terms):
        # Poisson weight: P(N=n)
        if n > 0:
            log_poisson += math.log(lambda_prime * T) - math.log(n)

        poisson_weight = math.exp(log_poisson)

        # Skip negligible terms
        if poisson_weight < 1e-15 and n > 5:
            break

        # Adjusted rate and volatility for n jumps
        if abs(1.0 + k) > 1e-10:
            r_n = r - lambda_j * k + n * math.log(1.0 + k) / T
        else:
            r_n = r - lambda_j * k

        sigma_n_sq = sigma ** 2 + n * sigma_j ** 2 / T
        sigma_n = math.sqrt(max(sigma_n_sq, 1e-10))

        # Black-Scholes call with adjusted parameters
        bs_price = black_scholes_call(S, K, T, r_n, sigma_n)
        price += poisson_weight * bs_price

    return price


def merton_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    n_terms: int = 50,
) -> float:
    """
    Merton jump diffusion European put price via series expansion.

    Uses the same series approach as the call, but with BS put prices.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Diffusion volatility
        lambda_j: Jump intensity
        mu_j: Mean log-jump size
        sigma_j: Jump size volatility
        n_terms: Number of terms

    Returns:
        Put option price under Merton's jump diffusion
    """
    if T <= 1e-12:
        return max(K - S, 0.0)

    k = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    lambda_prime = lambda_j * (1.0 + k)

    price = 0.0
    log_poisson = -lambda_prime * T

    for n in range(n_terms):
        if n > 0:
            log_poisson += math.log(lambda_prime * T) - math.log(n)

        poisson_weight = math.exp(log_poisson)
        if poisson_weight < 1e-15 and n > 5:
            break

        if abs(1.0 + k) > 1e-10:
            r_n = r - lambda_j * k + n * math.log(1.0 + k) / T
        else:
            r_n = r - lambda_j * k

        sigma_n_sq = sigma ** 2 + n * sigma_j ** 2 / T
        sigma_n = math.sqrt(max(sigma_n_sq, 1e-10))

        bs_price = black_scholes_put(S, K, T, r_n, sigma_n)
        price += poisson_weight * bs_price

    return price


def merton_price_grid(
    S_values: np.ndarray,
    t_values: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    option_type: str = "call",
    n_terms: int = 50,
) -> np.ndarray:
    """
    Compute Merton prices on a grid of (S, t) values.

    Args:
        S_values: Array of spot prices
        t_values: Array of times
        K: Strike price
        T: Maturity
        r: Risk-free rate
        sigma: Diffusion volatility
        lambda_j: Jump intensity
        mu_j: Mean log-jump size
        sigma_j: Jump size volatility
        option_type: "call" or "put"
        n_terms: Number of series terms

    Returns:
        Price grid, shape (len(S_values), len(t_values))
    """
    pricer = merton_call_price if option_type == "call" else merton_put_price

    grid = np.zeros((len(S_values), len(t_values)))
    for i, S in enumerate(S_values):
        for j, t in enumerate(t_values):
            tau = T - t  # time to maturity
            if tau <= 0:
                if option_type == "call":
                    grid[i, j] = max(S - K, 0.0)
                else:
                    grid[i, j] = max(K - S, 0.0)
            else:
                grid[i, j] = pricer(S, K, tau, r, sigma, lambda_j, mu_j, sigma_j, n_terms)

    return grid


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Compute Black-Scholes implied volatility via Newton-Raphson.

    Args:
        market_price: Observed option price
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        option_type: "call" or "put"
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Implied volatility, or None if not found
    """
    if T <= 1e-12:
        return None

    # Check intrinsic value
    if option_type == "call":
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - 1e-10:
        return None

    # Initial guess from Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T) * market_price / S

    # Clamp initial guess
    sigma = max(min(sigma, 5.0), 0.01)

    pricer = black_scholes_call if option_type == "call" else black_scholes_put

    for _ in range(max_iter):
        price = pricer(S, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega (same for call and put)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)

        if abs(vega) < 1e-15:
            break

        sigma -= diff / vega
        sigma = max(min(sigma, 5.0), 1e-6)

    return sigma


def merton_implied_vol_smile(
    S: float,
    K_values: np.ndarray,
    T: float,
    r: float,
    sigma: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    option_type: str = "call",
) -> np.ndarray:
    """
    Compute the implied volatility smile under Merton's model.

    For each strike, computes the Merton price and inverts Black-Scholes
    to get implied volatility.

    Args:
        S: Spot price
        K_values: Array of strikes
        T: Time to maturity
        r: Risk-free rate
        sigma: Diffusion volatility
        lambda_j: Jump intensity
        mu_j: Mean log-jump size
        sigma_j: Jump size volatility
        option_type: "call" or "put"

    Returns:
        Array of implied volatilities
    """
    pricer = merton_call_price if option_type == "call" else merton_put_price

    iv_values = np.full(len(K_values), np.nan)
    for i, K in enumerate(K_values):
        price = pricer(S, K, T, r, sigma, lambda_j, mu_j, sigma_j)
        iv = implied_volatility(price, S, K, T, r, option_type)
        if iv is not None:
            iv_values[i] = iv

    return iv_values
