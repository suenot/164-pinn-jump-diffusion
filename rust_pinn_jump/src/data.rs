//! Data fetching from Bybit (crypto) and synthetic data generation.
//!
//! Provides:
//! - Bybit REST API client for OHLCV kline data
//! - Synthetic data generation with jump diffusion dynamics
//! - Return statistics computation

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API kline response.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<serde_json::Value>>,
}

/// Fetch OHLCV kline data from Bybit REST API.
///
/// # Arguments
/// * `symbol` - Trading pair (e.g., "BTCUSDT")
/// * `interval` - Kline interval ("1", "5", "15", "60", "D", etc.)
/// * `limit` - Number of candles (max 1000)
/// * `category` - Market category ("spot", "linear")
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
    category: &str,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category={}&symbol={}&interval={}&limit={}",
        category, symbol, interval, limit.min(1000)
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?
        .json()?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    let candles: Vec<Candle> = response
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].as_str()?.parse().ok()?,
                    open: row[1].as_str()?.parse().ok()?,
                    high: row[2].as_str()?.parse().ok()?,
                    low: row[3].as_str()?.parse().ok()?,
                    close: row[4].as_str()?.parse().ok()?,
                    volume: row[5].as_str()?.parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    let mut sorted = candles;
    sorted.sort_by_key(|c| c.timestamp);
    Ok(sorted)
}

/// Generate synthetic data with jump diffusion dynamics.
///
/// Simulates the process:
/// ```text
/// dS/S = mu*dt + sigma*dW + J*dN
/// ```
pub fn generate_synthetic_data(
    initial_price: f64,
    n_days: usize,
    sigma: f64,
    mu: f64,
    lambda_j: f64,
    mu_j: f64,
    sigma_j: f64,
    seed: u64,
) -> Vec<Candle> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut price = initial_price;
    let dt = 1.0 / 252.0; // Daily
    let lambda_daily = lambda_j / 252.0;

    let base_ts = 1_704_067_200_000_i64; // 2024-01-01 in millis

    let mut candles = Vec::with_capacity(n_days);

    for i in 0..n_days {
        let old_price = price;

        // Diffusion component
        let z: f64 = rng.sample(StandardNormal);
        let mut log_return = (mu - 0.5 * sigma.powi(2)) * dt + sigma * dt.sqrt() * z;

        // Jump component
        if rng.gen::<f64>() < lambda_daily {
            let jump: f64 = rng.sample(StandardNormal);
            log_return += mu_j + sigma_j * jump;
        }

        price *= log_return.exp();

        // Generate OHLCV from the price path
        let noise: f64 = rng.gen::<f64>() * 0.01 + 0.003;
        let open_price = old_price * (1.0 + rng.sample::<f64, _>(StandardNormal) * 0.003);

        candles.push(Candle {
            timestamp: base_ts + (i as i64 * 86_400_000),
            open: open_price,
            high: price.max(open_price) * (1.0 + noise),
            low: price.min(open_price) * (1.0 - noise),
            close: price,
            volume: (rng.gen::<f64>() * 1000.0 + 100.0) * initial_price,
        });
    }

    candles
}

/// Compute log-returns from candle close prices.
pub fn log_returns(candles: &[Candle]) -> Vec<f64> {
    candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Return distribution statistics.
#[derive(Debug, Serialize)]
pub struct ReturnStats {
    pub n_observations: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub annualized_vol: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub min_return: f64,
    pub max_return: f64,
    pub pct_above_3sigma: f64,
}

/// Compute return statistics from log-returns.
pub fn compute_return_stats(returns: &[f64]) -> ReturnStats {
    let n = returns.len();
    if n < 2 {
        return ReturnStats {
            n_observations: n,
            mean: 0.0,
            std_dev: 0.0,
            annualized_vol: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            min_return: 0.0,
            max_return: 0.0,
            pct_above_3sigma: 0.0,
        };
    }

    let mean = returns.iter().sum::<f64>() / n as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    let skewness = if std_dev > 0.0 {
        let m3 = returns.iter().map(|r| ((r - mean) / std_dev).powi(3)).sum::<f64>();
        m3 / n as f64
    } else {
        0.0
    };

    let kurtosis = if std_dev > 0.0 {
        let m4 = returns.iter().map(|r| ((r - mean) / std_dev).powi(4)).sum::<f64>();
        m4 / n as f64 - 3.0 // Excess kurtosis
    } else {
        0.0
    };

    let min_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let threshold_3sigma = 3.0 * std_dev;
    let n_above = returns.iter().filter(|r| r.abs() > threshold_3sigma).count();
    let pct_above_3sigma = n_above as f64 / n as f64 * 100.0;

    ReturnStats {
        n_observations: n,
        mean,
        std_dev,
        annualized_vol: std_dev * 252.0_f64.sqrt(),
        skewness,
        kurtosis,
        min_return,
        max_return,
        pct_above_3sigma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let candles = generate_synthetic_data(100.0, 252, 0.20, 0.05, 5.0, -0.05, 0.10, 42);
        assert_eq!(candles.len(), 252);
        assert!(candles[0].close > 0.0);
    }

    #[test]
    fn test_log_returns() {
        let candles = generate_synthetic_data(100.0, 100, 0.20, 0.05, 5.0, -0.05, 0.10, 42);
        let returns = log_returns(&candles);
        assert_eq!(returns.len(), 99);
    }

    #[test]
    fn test_return_stats() {
        let candles = generate_synthetic_data(100.0, 500, 0.20, 0.05, 5.0, -0.05, 0.10, 42);
        let returns = log_returns(&candles);
        let stats = compute_return_stats(&returns);
        assert!(stats.annualized_vol > 0.0);
        assert!(stats.n_observations == 499);
    }
}
