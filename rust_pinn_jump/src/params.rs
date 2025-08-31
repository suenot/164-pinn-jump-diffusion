//! Merton jump diffusion model parameters.

use serde::{Deserialize, Serialize};

/// Parameters for Merton's jump diffusion model.
///
/// The model extends geometric Brownian motion with a compound Poisson jump process:
///
/// ```text
/// dS/S = (mu - lambda*k) dt + sigma dW + J dN
/// ```
///
/// where:
/// - `sigma`: diffusion volatility
/// - `lambda_j`: jump intensity (average jumps per year)
/// - `mu_j`: mean log-jump size
/// - `sigma_j`: jump size volatility
/// - `k = exp(mu_j + sigma_j^2/2) - 1`: expected percentage jump
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MertonParams {
    /// Diffusion volatility (annualized)
    pub sigma: f64,
    /// Risk-free interest rate
    pub r: f64,
    /// Jump intensity (expected number of jumps per year)
    pub lambda_j: f64,
    /// Mean log-jump size
    pub mu_j: f64,
    /// Jump size volatility (std of log-jump)
    pub sigma_j: f64,
    /// Strike price
    pub strike: f64,
    /// Time to maturity (years)
    pub maturity: f64,
    /// Lower bound for spot price domain
    pub s_min: f64,
    /// Upper bound for spot price domain
    pub s_max: f64,
}

impl MertonParams {
    /// Create default equity parameters.
    pub fn default_equity() -> Self {
        Self {
            sigma: 0.20,
            r: 0.05,
            lambda_j: 5.0,
            mu_j: -0.05,
            sigma_j: 0.15,
            strike: 100.0,
            maturity: 1.0,
            s_min: 20.0,
            s_max: 300.0,
        }
    }

    /// Create default BTC/crypto parameters.
    pub fn default_btc() -> Self {
        Self {
            sigma: 0.50,
            r: 0.05,
            lambda_j: 12.0,
            mu_j: -0.08,
            sigma_j: 0.15,
            strike: 50_000.0,
            maturity: 1.0,
            s_min: 10_000.0,
            s_max: 150_000.0,
        }
    }

    /// Expected percentage jump size: E[J] = exp(mu_j + sigma_j^2 / 2) - 1
    pub fn k(&self) -> f64 {
        (self.mu_j + 0.5 * self.sigma_j.powi(2)).exp() - 1.0
    }
}

impl Default for MertonParams {
    fn default() -> Self {
        Self::default_equity()
    }
}

/// Estimated jump parameters from historical data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedParams {
    /// Annualized diffusion volatility (non-jump days)
    pub sigma: f64,
    /// Jump intensity (jumps per year)
    pub lambda_j: f64,
    /// Mean log-jump size
    pub mu_j: f64,
    /// Jump size volatility
    pub sigma_j: f64,
    /// Number of detected jumps
    pub n_jumps: usize,
    /// Total number of observations
    pub n_observations: usize,
    /// Jump detection threshold (number of sigmas)
    pub threshold: f64,
}

impl std::fmt::Display for EstimatedParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EstimatedParams(sigma={:.4}, lambda_j={:.2}, mu_j={:.4}, sigma_j={:.4}, \
             n_jumps={}/{}, threshold={:.1} sigma)",
            self.sigma, self.lambda_j, self.mu_j, self.sigma_j,
            self.n_jumps, self.n_observations, self.threshold,
        )
    }
}

/// Estimate Merton parameters from historical log-returns.
///
/// Method:
/// 1. Estimate robust volatility via MAD
/// 2. Identify jumps as returns exceeding threshold * sigma
/// 3. Estimate diffusion vol from non-jump days
/// 4. Estimate jump intensity and distribution from jump days
pub fn estimate_jump_params(
    log_returns: &[f64],
    threshold_sigma: f64,
    annualization_factor: f64,
) -> EstimatedParams {
    let n = log_returns.len();
    if n < 10 {
        return EstimatedParams {
            sigma: 0.20,
            lambda_j: 5.0,
            mu_j: -0.05,
            sigma_j: 0.10,
            n_jumps: 0,
            n_observations: n,
            threshold: threshold_sigma,
        };
    }

    // Median
    let mut sorted = log_returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2];

    // MAD (Median Absolute Deviation)
    let mut abs_devs: Vec<f64> = log_returns.iter().map(|r| (r - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[n / 2];
    let sigma_robust = mad * 1.4826;

    // Identify jumps
    let threshold = threshold_sigma * sigma_robust;
    let mut jump_returns = Vec::new();
    let mut non_jump_returns = Vec::new();

    for &r in log_returns {
        if (r - median).abs() > threshold {
            jump_returns.push(r);
        } else {
            non_jump_returns.push(r);
        }
    }

    let n_jumps = jump_returns.len();

    // Diffusion volatility (from non-jump returns, annualized)
    let sigma = if non_jump_returns.len() > 1 {
        let mean_nj: f64 = non_jump_returns.iter().sum::<f64>() / non_jump_returns.len() as f64;
        let var_nj: f64 = non_jump_returns.iter()
            .map(|r| (r - mean_nj).powi(2))
            .sum::<f64>() / (non_jump_returns.len() - 1) as f64;
        var_nj.sqrt() * annualization_factor.sqrt()
    } else {
        sigma_robust * annualization_factor.sqrt()
    };

    // Jump intensity (annualized)
    let lambda_j = (n_jumps as f64 / n as f64 * annualization_factor).max(0.1);

    // Jump distribution parameters
    let (mu_j, sigma_j) = if n_jumps > 0 {
        let mean_j: f64 = jump_returns.iter().sum::<f64>() / n_jumps as f64;
        let var_j: f64 = if n_jumps > 1 {
            jump_returns.iter()
                .map(|r| (r - mean_j).powi(2))
                .sum::<f64>() / (n_jumps - 1) as f64
        } else {
            0.01
        };
        (mean_j, var_j.sqrt().max(0.01))
    } else {
        (-0.05, 0.10)
    };

    EstimatedParams {
        sigma,
        lambda_j,
        mu_j,
        sigma_j,
        n_jumps,
        n_observations: n,
        threshold: threshold_sigma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merton_params_k() {
        let params = MertonParams::default_equity();
        let k = params.k();
        // k = exp(-0.05 + 0.5 * 0.15^2) - 1 = exp(-0.03875) - 1 ≈ -0.038
        assert!((k - (-0.038)).abs() < 0.01, "k = {}", k);
    }

    #[test]
    fn test_estimate_params() {
        let returns: Vec<f64> = (0..500)
            .map(|_| {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen::<f64>() * 0.04 - 0.02
            })
            .collect();

        let estimated = estimate_jump_params(&returns, 3.0, 252.0);
        assert!(estimated.sigma > 0.0);
        assert!(estimated.lambda_j >= 0.1);
    }
}
