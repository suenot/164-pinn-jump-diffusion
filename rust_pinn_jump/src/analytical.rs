//! Analytical Merton jump diffusion option pricing.
//!
//! Merton (1976) showed that European options under jump diffusion can be priced
//! as an infinite series of Black-Scholes prices weighted by Poisson probabilities:
//!
//! ```text
//! V = sum_{n=0}^{inf} P(N=n) * BS(S, K, T, r_n, sigma_n)
//! ```
//!
//! This module also provides Black-Scholes pricing and implied volatility computation.

use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::PI;

use crate::params::MertonParams;

/// Standard normal CDF.
fn norm_cdf(x: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(x)
}

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-Scholes European call price.
pub fn bs_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 1e-12 {
        return (s - k).max(0.0);
    }
    if sigma <= 1e-12 {
        return (s - k * (-r * t).exp()).max(0.0);
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma.powi(2)) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Black-Scholes European put price.
pub fn bs_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 1e-12 {
        return (k - s).max(0.0);
    }
    if sigma <= 1e-12 {
        return (k * (-r * t).exp() - s).max(0.0);
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma.powi(2)) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
}

/// Black-Scholes Vega (same for call and put).
pub fn bs_vega(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 1e-12 || sigma <= 1e-12 {
        return 0.0;
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma.powi(2)) * t) / (sigma * sqrt_t);
    s * norm_pdf(d1) * sqrt_t
}

/// Merton jump diffusion European call price.
///
/// Computes the price as a series of Black-Scholes prices:
///
/// ```text
/// V = sum_{n=0}^{N} P(n) * BS(S, K, T, r_n, sigma_n)
/// ```
///
/// where:
/// - P(n) = Poisson(lambda' * T, n)
/// - lambda' = lambda * (1 + k)
/// - r_n = r - lambda*k + n*ln(1+k)/T
/// - sigma_n = sqrt(sigma^2 + n*sigma_j^2/T)
pub fn merton_call_price(params: &MertonParams, spot: f64) -> f64 {
    merton_price_internal(params, spot, true)
}

/// Merton jump diffusion European put price.
pub fn merton_put_price(params: &MertonParams, spot: f64) -> f64 {
    merton_price_internal(params, spot, false)
}

fn merton_price_internal(params: &MertonParams, spot: f64, is_call: bool) -> f64 {
    let t = params.maturity;
    if t <= 1e-12 {
        return if is_call {
            (spot - params.strike).max(0.0)
        } else {
            (params.strike - spot).max(0.0)
        };
    }

    let k = params.k();
    let lambda_prime = params.lambda_j * (1.0 + k);
    let n_terms = 50;

    let mut price = 0.0;
    let mut log_poisson = -lambda_prime * t;

    for n in 0..n_terms {
        if n > 0 {
            log_poisson += (lambda_prime * t).ln() - (n as f64).ln();
        }

        let poisson_weight = log_poisson.exp();
        if poisson_weight < 1e-15 && n > 5 {
            break;
        }

        // Adjusted rate
        let r_n = if (1.0 + k).abs() > 1e-10 {
            params.r - params.lambda_j * k + (n as f64) * (1.0 + k).ln() / t
        } else {
            params.r - params.lambda_j * k
        };

        // Adjusted volatility
        let sigma_n_sq = params.sigma.powi(2) + (n as f64) * params.sigma_j.powi(2) / t;
        let sigma_n = sigma_n_sq.max(1e-10).sqrt();

        let bs_price = if is_call {
            bs_call(spot, params.strike, t, r_n, sigma_n)
        } else {
            bs_put(spot, params.strike, t, r_n, sigma_n)
        };

        price += poisson_weight * bs_price;
    }

    price
}

/// Compute Black-Scholes implied volatility via Newton-Raphson.
///
/// # Arguments
/// * `market_price` - Observed option price
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to maturity
/// * `r` - Risk-free rate
/// * `is_call` - true for call, false for put
///
/// # Returns
/// Implied volatility, or None if not found
pub fn implied_volatility(
    market_price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    is_call: bool,
) -> Option<f64> {
    if t <= 1e-12 {
        return None;
    }

    // Initial guess (Brenner-Subrahmanyam)
    let mut sigma = (2.0 * PI / t).sqrt() * market_price / s;
    sigma = sigma.clamp(0.01, 5.0);

    let tol = 1e-8;
    let max_iter = 100;

    for _ in 0..max_iter {
        let price = if is_call {
            bs_call(s, k, t, r, sigma)
        } else {
            bs_put(s, k, t, r, sigma)
        };

        let diff = price - market_price;
        if diff.abs() < tol {
            return Some(sigma);
        }

        let vega = bs_vega(s, k, t, r, sigma);
        if vega.abs() < 1e-15 {
            break;
        }

        sigma -= diff / vega;
        sigma = sigma.clamp(1e-6, 5.0);
    }

    // Fall back to bisection if Newton failed
    bisection_iv(market_price, s, k, t, r, is_call)
}

fn bisection_iv(
    target: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    is_call: bool,
) -> Option<f64> {
    let mut lo = 0.001;
    let mut hi = 5.0;
    let tol = 1e-8;

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let price = if is_call {
            bs_call(s, k, t, r, mid)
        } else {
            bs_put(s, k, t, r, mid)
        };

        if (price - target).abs() < tol {
            return Some(mid);
        }

        if price > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Some(0.5 * (lo + hi))
}

/// Compute implied volatility smile under Merton's model.
///
/// For each strike, computes the Merton price and inverts BS to get IV.
pub fn merton_iv_smile(
    params: &MertonParams,
    spot: f64,
    strikes: &[f64],
    is_call: bool,
) -> Vec<Option<f64>> {
    strikes
        .iter()
        .map(|&k| {
            let mut p = params.clone();
            p.strike = k;
            let merton_price = if is_call {
                merton_call_price(&p, spot)
            } else {
                merton_put_price(&p, spot)
            };
            implied_volatility(merton_price, spot, k, params.maturity, params.r, is_call)
        })
        .collect()
}

/// Price options on a grid of spot prices.
pub fn merton_price_grid(
    params: &MertonParams,
    spots: &[f64],
    is_call: bool,
) -> Vec<f64> {
    spots
        .iter()
        .map(|&s| {
            if is_call {
                merton_call_price(params, s)
            } else {
                merton_put_price(params, s)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bs_call_atm() {
        // ATM call: S=K=100, T=1, r=0.05, sigma=0.20
        let price = bs_call(100.0, 100.0, 1.0, 0.05, 0.20);
        // Expected: approximately 10.45
        assert!(price > 9.0 && price < 12.0, "BS call ATM = {}", price);
    }

    #[test]
    fn test_bs_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let call = bs_call(s, k, t, r, sigma);
        let put = bs_put(s, k, t, r, sigma);

        // Put-call parity: C - P = S - K*exp(-rT)
        let parity_rhs = s - k * (-r * t).exp();
        assert_relative_eq!(call - put, parity_rhs, epsilon = 1e-6);
    }

    #[test]
    fn test_merton_call_no_jumps() {
        // With lambda=0, Merton should equal BS
        let params = MertonParams {
            sigma: 0.20,
            r: 0.05,
            lambda_j: 0.0,
            mu_j: 0.0,
            sigma_j: 0.0,
            strike: 100.0,
            maturity: 1.0,
            s_min: 20.0,
            s_max: 300.0,
        };

        let merton = merton_call_price(&params, 100.0);
        let bs = bs_call(100.0, 100.0, 1.0, 0.05, 0.20);
        assert_relative_eq!(merton, bs, epsilon = 0.01);
    }

    #[test]
    fn test_merton_call_with_jumps() {
        let params = MertonParams::default_equity();
        let price = merton_call_price(&params, 100.0);
        // With jumps, option should be worth more than BS
        let bs = bs_call(100.0, 100.0, 1.0, 0.05, 0.20);
        assert!(price > bs * 0.9, "Merton = {}, BS = {}", price, bs);
    }

    #[test]
    fn test_implied_volatility() {
        let price = bs_call(100.0, 100.0, 1.0, 0.05, 0.25);
        let iv = implied_volatility(price, 100.0, 100.0, 1.0, 0.05, true);
        assert!(iv.is_some());
        assert_relative_eq!(iv.unwrap(), 0.25, epsilon = 1e-4);
    }
}
