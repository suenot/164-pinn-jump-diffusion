//! Price options using the analytical Merton model and compute volatility smile.
//!
//! This binary demonstrates:
//! - Merton vs Black-Scholes pricing comparison
//! - Volatility smile computation
//! - Greeks via finite differences
//! - Impact of jump parameters
//!
//! Usage:
//!     cargo run --bin price_options
//!     cargo run --bin price_options -- --asset BTC --strikes 40000,45000,50000,55000,60000

use anyhow::Result;
use clap::Parser;
use colored::Colorize;

use pinn_jump_diffusion::analytical::{
    bs_call, bs_put, implied_volatility, merton_call_price, merton_iv_smile, merton_put_price,
};
use pinn_jump_diffusion::params::MertonParams;

#[derive(Parser, Debug)]
#[command(name = "price_options", about = "Price options under Merton jump diffusion")]
struct Args {
    /// Asset type (SPY, BTC, ETH)
    #[arg(short, long, default_value = "SPY")]
    asset: String,

    /// Comma-separated list of strikes (auto-generated if not provided)
    #[arg(short, long)]
    strikes: Option<String>,

    /// Spot price (default: auto based on asset)
    #[arg(long)]
    spot: Option<f64>,

    /// Time to maturity in years
    #[arg(short, long, default_value_t = 0.5)]
    maturity: f64,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Setup parameters
    let mut params = match args.asset.to_uppercase().as_str() {
        "BTC" | "BTCUSDT" => MertonParams::default_btc(),
        _ => MertonParams::default_equity(),
    };
    params.maturity = args.maturity;

    let spot = args.spot.unwrap_or(params.strike);

    // Generate strikes if not provided
    let strikes: Vec<f64> = if let Some(s) = &args.strikes {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    } else {
        let k = params.strike;
        (0..11)
            .map(|i| k * (0.7 + 0.06 * i as f64))
            .collect()
    };

    println!("{}", "=".repeat(70).blue());
    println!(
        "{}",
        format!(
            "Merton Jump Diffusion Option Pricing: {}",
            args.asset
        )
        .blue()
        .bold()
    );
    println!("{}", "=".repeat(70).blue());

    println!("\n{}", "Model Parameters:".green().bold());
    println!("  Spot       = {:.2}", spot);
    println!("  sigma      = {:.4}", params.sigma);
    println!("  r          = {:.4}", params.r);
    println!("  lambda_j   = {:.2}", params.lambda_j);
    println!("  mu_j       = {:.4}", params.mu_j);
    println!("  sigma_j    = {:.4}", params.sigma_j);
    println!("  maturity   = {:.2}y", params.maturity);
    println!("  k (E[J])   = {:.4}", params.k());

    // ═══════════════════════════════════════════════════════════════
    // Price comparison table
    // ═══════════════════════════════════════════════════════════════
    println!("\n{}", "Call Option Prices:".green().bold());
    println!(
        "{:>12} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "Strike", "Moneyness", "BS Price", "Merton", "Jump Prem", "IV (%)"
    );
    println!("{}", "-".repeat(70));

    for &k in &strikes {
        let mut p = params.clone();
        p.strike = k;

        let bs_price = bs_call(spot, k, params.maturity, params.r, params.sigma);
        let merton_price = merton_call_price(&p, spot);
        let jump_premium = merton_price - bs_price;
        let moneyness = k / spot;

        let iv = implied_volatility(merton_price, spot, k, params.maturity, params.r, true);
        let iv_str = iv
            .map(|v| format!("{:.2}", v * 100.0))
            .unwrap_or_else(|| "N/A".to_string());

        let premium_color = if jump_premium > 0.0 {
            format!("{:+.4}", jump_premium).green()
        } else {
            format!("{:+.4}", jump_premium).red()
        };

        println!(
            "{:12.2} {:10.4} {:12.4} {:12.4} {} {:>10}",
            k, moneyness, bs_price, merton_price, premium_color, iv_str,
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Volatility smile
    // ═══════════════════════════════════════════════════════════════
    println!("\n{}", "Volatility Smile (Implied Vol vs Moneyness):".green().bold());

    let smile = merton_iv_smile(&params, spot, &strikes, true);

    println!("{:>10} {:>10} {:>30}", "K/S", "IV (%)", "Visual");
    println!("{}", "-".repeat(55));

    let iv_values: Vec<f64> = smile.iter().filter_map(|&v| v).collect();
    let iv_min = iv_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let iv_max = iv_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for (&k, iv) in strikes.iter().zip(smile.iter()) {
        let moneyness = k / spot;
        if let Some(iv_val) = iv {
            let bar_len = if iv_max > iv_min {
                ((iv_val - iv_min) / (iv_max - iv_min) * 25.0) as usize
            } else {
                12
            };
            let bar = "#".repeat(bar_len.max(1));
            println!(
                "{:10.4} {:9.2}% {}",
                moneyness,
                iv_val * 100.0,
                bar.cyan(),
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Impact of jump parameters
    // ═══════════════════════════════════════════════════════════════
    println!(
        "\n{}",
        "Impact of Jump Intensity on ATM Call:".green().bold()
    );
    println!("{:>12} {:>12} {:>12} {:>12}", "lambda_j", "Merton", "BS", "Premium");
    println!("{}", "-".repeat(50));

    for &lam in &[0.0, 2.0, 5.0, 10.0, 20.0, 50.0] {
        let mut p = params.clone();
        p.lambda_j = lam;
        let m_price = merton_call_price(&p, spot);
        let b_price = bs_call(spot, params.strike, params.maturity, params.r, params.sigma);
        println!(
            "{:12.1} {:12.4} {:12.4} {:+12.4}",
            lam,
            m_price,
            b_price,
            m_price - b_price,
        );
    }

    println!(
        "\n{}",
        "Impact of Mean Jump Size on ATM Call:".green().bold()
    );
    println!("{:>12} {:>12} {:>12}", "mu_j", "Merton", "IV (%)");
    println!("{}", "-".repeat(38));

    for &mu in &[-0.20, -0.15, -0.10, -0.05, 0.0, 0.05] {
        let mut p = params.clone();
        p.mu_j = mu;
        let m_price = merton_call_price(&p, spot);
        let iv = implied_volatility(m_price, spot, params.strike, params.maturity, params.r, true);
        let iv_str = iv
            .map(|v| format!("{:.2}", v * 100.0))
            .unwrap_or_else(|| "N/A".to_string());
        println!("{:12.4} {:12.4} {:>12}", mu, m_price, iv_str);
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
