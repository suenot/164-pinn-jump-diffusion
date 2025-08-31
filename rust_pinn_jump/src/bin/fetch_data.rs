//! Fetch market data from Bybit and compute return statistics.
//!
//! Usage:
//!     cargo run --bin fetch_data -- --symbol BTCUSDT --interval D --limit 500

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use pinn_jump_diffusion::data::{
    compute_return_stats, fetch_bybit_klines, generate_synthetic_data, log_returns,
};
use pinn_jump_diffusion::params::estimate_jump_params;

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch market data and estimate jump parameters")]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1, 5, 15, 60, D, W)
    #[arg(short, long, default_value = "D")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value_t = 500)]
    limit: usize,

    /// Market category (spot, linear)
    #[arg(short, long, default_value = "spot")]
    category: String,

    /// Use synthetic data instead of live API
    #[arg(long, default_value_t = false)]
    synthetic: bool,

    /// Jump detection threshold in sigmas
    #[arg(long, default_value_t = 3.0)]
    threshold: f64,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!(
        "{}",
        format!("Fetching {} data (interval={}, limit={})", args.symbol, args.interval, args.limit)
            .blue()
            .bold()
    );
    println!("{}", "=".repeat(60).blue());

    let candles = if args.synthetic {
        println!("{}", "Using synthetic data with jump diffusion dynamics".yellow());
        let initial_price = if args.symbol.contains("BTC") {
            50_000.0
        } else if args.symbol.contains("ETH") {
            3_000.0
        } else {
            100.0
        };
        generate_synthetic_data(initial_price, args.limit, 0.50, 0.05, 12.0, -0.08, 0.15, 42)
    } else {
        println!("Fetching from Bybit API...");
        match fetch_bybit_klines(&args.symbol, &args.interval, args.limit, &args.category) {
            Ok(candles) => candles,
            Err(e) => {
                println!(
                    "{}",
                    format!("API error: {}. Falling back to synthetic data.", e).yellow()
                );
                let initial_price = if args.symbol.contains("BTC") {
                    50_000.0
                } else {
                    100.0
                };
                generate_synthetic_data(
                    initial_price, args.limit, 0.50, 0.05, 12.0, -0.08, 0.15, 42,
                )
            }
        }
    };

    println!("Fetched {} candles", candles.len());

    // Show last 5 candles
    println!("\n{}", "Last 5 candles:".green().bold());
    for candle in candles.iter().rev().take(5) {
        println!(
            "  ts={}, O={:.2}, H={:.2}, L={:.2}, C={:.2}, V={:.0}",
            candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume,
        );
    }

    // Compute returns
    let returns = log_returns(&candles);
    if returns.is_empty() {
        println!("Not enough data to compute returns.");
        return Ok(());
    }

    // Return statistics
    println!("\n{}", "Return Statistics:".green().bold());
    let stats = compute_return_stats(&returns);
    println!("  Observations:     {}", stats.n_observations);
    println!("  Mean daily return: {:.6}", stats.mean);
    println!("  Std daily return:  {:.6}", stats.std_dev);
    println!("  Annualized vol:    {:.4}", stats.annualized_vol);
    println!("  Skewness:          {:.4}", stats.skewness);
    println!("  Excess kurtosis:   {:.4}", stats.kurtosis);
    println!("  Min return:        {:.4}", stats.min_return);
    println!("  Max return:        {:.4}", stats.max_return);
    println!("  % above 3-sigma:   {:.2}%", stats.pct_above_3sigma);

    // Estimate jump parameters
    println!("\n{}", "Estimated Merton Parameters:".green().bold());
    let params = estimate_jump_params(&returns, args.threshold, 252.0);
    println!("  {}", params);

    // Normal distribution comparison
    let normal_expected_3sigma = 0.27; // Expected % above 3-sigma for normal
    println!("\n{}", "Fat Tail Analysis:".green().bold());
    println!(
        "  3-sigma exceedances: {:.2}% (normal expects {:.2}%)",
        stats.pct_above_3sigma, normal_expected_3sigma
    );
    let fat_tail_ratio = stats.pct_above_3sigma / normal_expected_3sigma;
    if fat_tail_ratio > 2.0 {
        println!(
            "  {}",
            format!(
                "  => {:.1}x more extreme events than normal -- strong evidence of jumps!",
                fat_tail_ratio
            )
            .red()
            .bold()
        );
    } else {
        println!("  => Returns are close to normal (limited jump evidence)");
    }

    // Save to CSV
    let csv_path = format!("{}_klines.csv", args.symbol.to_lowercase());
    let mut writer = csv::Writer::from_path(&csv_path)?;
    writer.write_record(["timestamp", "open", "high", "low", "close", "volume"])?;
    for candle in &candles {
        writer.write_record([
            candle.timestamp.to_string(),
            format!("{:.2}", candle.open),
            format!("{:.2}", candle.high),
            format!("{:.2}", candle.low),
            format!("{:.2}", candle.close),
            format!("{:.0}", candle.volume),
        ])?;
    }
    writer.flush()?;
    println!("\n{}", format!("Data saved to {}", csv_path).green());

    Ok(())
}
