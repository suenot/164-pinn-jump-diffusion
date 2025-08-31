//! Backtesting framework for the jump mispricing strategy.
//!
//! Strategy: compare Merton (jump-aware) option prices with Black-Scholes
//! (no-jump) prices. When the difference exceeds a threshold, trade
//! the mispricing.

use crate::analytical::{bs_call, merton_call_price};
use crate::data::Candle;
use crate::params::{estimate_jump_params, MertonParams};

/// A single trade record.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_day: usize,
    pub exit_day: usize,
    pub direction: Direction,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub mispricing_pct: f64,
}

/// Trade direction.
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Long,
    Short,
}

/// Backtest result summary.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub trades: Vec<Trade>,
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub n_trades: usize,
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BacktestResult(\n  n_trades={},\n  total_pnl={:.2},\n  sharpe={:.2},\n  \
             max_dd={:.2}%,\n  win_rate={:.1}%,\n  profit_factor={:.2}\n)",
            self.n_trades,
            self.total_pnl,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
        )
    }
}

/// Strategy configuration.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Rolling window for parameter estimation (in days)
    pub calibration_window: usize,
    /// Number of days to hold each trade
    pub holding_period: usize,
    /// Minimum mispricing to trigger a trade (as fraction)
    pub mispricing_threshold: f64,
    /// Maximum number of simultaneous positions
    pub max_positions: usize,
    /// Option maturity in years for pricing
    pub option_maturity: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Notional per trade
    pub position_size: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            calibration_window: 60,
            holding_period: 5,
            mispricing_threshold: 0.05,
            max_positions: 5,
            option_maturity: 30.0 / 365.0,
            risk_free_rate: 0.05,
            position_size: 10_000.0,
        }
    }
}

/// Run the jump mispricing backtest.
///
/// Strategy:
/// 1. Estimate Merton parameters from a rolling window
/// 2. Price ATM option with Merton (accounting for jumps)
/// 3. Price same option with Black-Scholes (no jumps)
/// 4. If Merton price >> BS price, buy (market underestimates jumps)
/// 5. Hold for `holding_period` days, then close
pub fn run_backtest(candles: &[Candle], config: &StrategyConfig) -> BacktestResult {
    let n = candles.len();
    let start_idx = config.calibration_window + 1;

    if n <= start_idx {
        return BacktestResult {
            trades: vec![],
            total_pnl: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            n_trades: 0,
        };
    }

    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let mut trades: Vec<Trade> = Vec::new();
    let mut open_positions: Vec<(usize, f64, f64, Direction, usize)> = Vec::new();
    // (entry_day, entry_option_price, strike, direction, exit_day)

    let mut daily_pnl: Vec<f64> = Vec::new();

    for i in start_idx..n {
        let mut day_pnl = 0.0;

        // Close expired positions
        let mut to_close = Vec::new();
        for (idx, &(entry_day, entry_opt_price, strike, direction, exit_day)) in
            open_positions.iter().enumerate()
        {
            if i >= exit_day {
                // Compute approximate exit value
                let spot = prices[i];
                let tau_remaining = (config.option_maturity
                    - config.holding_period as f64 / 252.0)
                    .max(1.0 / 365.0);

                // Use BS for quick approximate exit pricing
                let exit_opt_price = bs_call(
                    spot,
                    strike,
                    tau_remaining,
                    config.risk_free_rate,
                    compute_realized_vol(&prices, i, config.calibration_window),
                );

                let trade_pnl = match direction {
                    Direction::Long => {
                        (exit_opt_price - entry_opt_price) / entry_opt_price * config.position_size
                    }
                    Direction::Short => {
                        (entry_opt_price - exit_opt_price) / entry_opt_price * config.position_size
                    }
                };

                let mispricing = (entry_opt_price - exit_opt_price).abs() / entry_opt_price;

                trades.push(Trade {
                    entry_day,
                    exit_day: i,
                    direction,
                    entry_price: entry_opt_price,
                    exit_price: exit_opt_price,
                    pnl: trade_pnl,
                    mispricing_pct: mispricing * 100.0,
                });

                day_pnl += trade_pnl;
                to_close.push(idx);
            }
        }

        // Remove closed positions (reverse order)
        for idx in to_close.into_iter().rev() {
            open_positions.remove(idx);
        }

        // Check for new entries
        if open_positions.len() < config.max_positions {
            let window_start = i.saturating_sub(config.calibration_window);
            let window = &prices[window_start..=i];

            let log_rets: Vec<f64> = window
                .windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();

            if log_rets.len() >= 10 {
                let estimated = estimate_jump_params(&log_rets, 3.0, 252.0);

                let spot = prices[i];
                let strike = spot; // ATM

                // Merton price (with jumps)
                let merton_params = MertonParams {
                    sigma: estimated.sigma.max(0.05),
                    r: config.risk_free_rate,
                    lambda_j: estimated.lambda_j,
                    mu_j: estimated.mu_j,
                    sigma_j: estimated.sigma_j.max(0.01),
                    strike,
                    maturity: config.option_maturity,
                    s_min: spot * 0.2,
                    s_max: spot * 3.0,
                };
                let merton_price = merton_call_price(&merton_params, spot);

                // BS price (no jumps)
                let bs_price = bs_call(
                    spot,
                    strike,
                    config.option_maturity,
                    config.risk_free_rate,
                    estimated.sigma.max(0.05),
                );

                if bs_price > 0.01 {
                    let mispricing = (merton_price - bs_price) / bs_price;

                    if mispricing > config.mispricing_threshold {
                        // Market underestimates jump risk -> buy option
                        let exit_day = (i + config.holding_period).min(n - 1);
                        open_positions.push((
                            i,
                            bs_price,
                            strike,
                            Direction::Long,
                            exit_day,
                        ));
                    } else if mispricing < -config.mispricing_threshold {
                        // Market overestimates -> sell option (rare)
                        let exit_day = (i + config.holding_period).min(n - 1);
                        open_positions.push((
                            i,
                            bs_price,
                            strike,
                            Direction::Short,
                            exit_day,
                        ));
                    }
                }
            }
        }

        daily_pnl.push(day_pnl);
    }

    // Compute summary statistics
    let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
    let n_trades = trades.len();
    let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = if n_trades > 0 {
        wins as f64 / n_trades as f64
    } else {
        0.0
    };

    let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_loss: f64 = trades
        .iter()
        .filter(|t| t.pnl < 0.0)
        .map(|t| t.pnl.abs())
        .sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Sharpe ratio from daily PnL
    let sharpe_ratio = if daily_pnl.len() > 1 {
        let mean_pnl = daily_pnl.iter().sum::<f64>() / daily_pnl.len() as f64;
        let var = daily_pnl
            .iter()
            .map(|p| (p - mean_pnl).powi(2))
            .sum::<f64>()
            / (daily_pnl.len() - 1) as f64;
        let std_pnl = var.sqrt();
        if std_pnl > 0.0 {
            mean_pnl / std_pnl * 252.0_f64.sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Max drawdown
    let mut equity = Vec::with_capacity(daily_pnl.len() + 1);
    equity.push(config.position_size * config.max_positions as f64);
    for &pnl in &daily_pnl {
        equity.push(*equity.last().unwrap() + pnl);
    }
    let max_drawdown = compute_max_drawdown(&equity);

    BacktestResult {
        trades,
        total_pnl,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        profit_factor,
        n_trades,
    }
}

/// Compute realized volatility from a price window.
fn compute_realized_vol(prices: &[f64], current_idx: usize, window: usize) -> f64 {
    let start = current_idx.saturating_sub(window);
    let slice = &prices[start..=current_idx.min(prices.len() - 1)];

    if slice.len() < 3 {
        return 0.20; // Default
    }

    let log_rets: Vec<f64> = slice.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean: f64 = log_rets.iter().sum::<f64>() / log_rets.len() as f64;
    let var: f64 = log_rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
        / (log_rets.len() - 1).max(1) as f64;

    (var * 252.0).sqrt().max(0.05)
}

/// Compute maximum drawdown from an equity curve.
fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut max_dd = 0.0_f64;
    let mut peak = equity[0];

    for &val in equity {
        if val > peak {
            peak = val;
        }
        let dd = (val - peak) / peak;
        if dd < max_dd {
            max_dd = dd;
        }
    }

    max_dd.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::generate_synthetic_data;

    #[test]
    fn test_backtest_runs() {
        let candles = generate_synthetic_data(100.0, 500, 0.20, 0.05, 5.0, -0.05, 0.10, 42);
        let config = StrategyConfig {
            calibration_window: 30,
            holding_period: 3,
            mispricing_threshold: 0.02,
            ..Default::default()
        };

        let result = run_backtest(&candles, &config);
        // Should complete without panicking
        assert!(result.n_trades >= 0);
        println!("{}", result);
    }
}
