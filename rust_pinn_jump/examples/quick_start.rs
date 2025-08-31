//! Quick start example for the Jump Diffusion PINN library.
//!
//! Demonstrates:
//! 1. Setting up Merton parameters (equity and crypto)
//! 2. Analytical Merton pricing
//! 3. Black-Scholes comparison
//! 4. Implied volatility smile
//! 5. PINN creation and evaluation
//! 6. Greeks computation
//!
//! Run with:
//!     cargo run --example quick_start

use pinn_jump_diffusion::analytical::{
    bs_call, implied_volatility, merton_call_price, merton_iv_smile, merton_put_price,
};
use pinn_jump_diffusion::data::{compute_return_stats, generate_synthetic_data, log_returns};
use pinn_jump_diffusion::params::{estimate_jump_params, MertonParams};
use pinn_jump_diffusion::pinn::JumpDiffusionPINN;

fn main() {
    println!("====================================================");
    println!("  PINN Jump Diffusion -- Quick Start Example");
    println!("====================================================\n");

    // ═══════════════════════════════════════════════════════════
    // Part 1: Analytical Merton Pricing
    // ═══════════════════════════════════════════════════════════
    println!("--- Part 1: Analytical Merton Pricing ---\n");

    let params = MertonParams::default_equity();
    let spot = 100.0;

    let merton_call = merton_call_price(&params, spot);
    let merton_put = merton_put_price(&params, spot);
    let bs_call_price = bs_call(spot, params.strike, params.maturity, params.r, params.sigma);

    println!("Equity Parameters:");
    println!("  sigma={}, r={}, lambda={}, mu_j={}, sigma_j={}",
             params.sigma, params.r, params.lambda_j, params.mu_j, params.sigma_j);
    println!("  Spot={}, Strike={}, T={}", spot, params.strike, params.maturity);
    println!();
    println!("Merton Call:  {:.4}", merton_call);
    println!("Merton Put:   {:.4}", merton_put);
    println!("BS Call:       {:.4}", bs_call_price);
    println!("Jump Premium: {:+.4} ({:.1}%)",
             merton_call - bs_call_price,
             (merton_call - bs_call_price) / bs_call_price * 100.0);

    // Put-call parity check
    let parity_lhs = merton_call - merton_put;
    let parity_rhs = spot - params.strike * (-params.r * params.maturity).exp();
    println!("\nPut-Call Parity Check:");
    println!("  C - P   = {:.4}", parity_lhs);
    println!("  S - K*df = {:.4}", parity_rhs);
    println!("  Diff     = {:.6}", (parity_lhs - parity_rhs).abs());

    // ═══════════════════════════════════════════════════════════
    // Part 2: Volatility Smile
    // ═══════════════════════════════════════════════════════════
    println!("\n--- Part 2: Volatility Smile ---\n");

    let strikes: Vec<f64> = (0..9).map(|i| 80.0 + 5.0 * i as f64).collect();
    let smile = merton_iv_smile(&params, spot, &strikes, true);

    println!("{:>8} {:>10}", "Strike", "IV (%)");
    println!("{}", "-".repeat(20));
    for (&k, iv) in strikes.iter().zip(smile.iter()) {
        if let Some(v) = iv {
            println!("{:8.1} {:9.2}%", k, v * 100.0);
        }
    }
    println!("\nNote: The U-shape (higher IV for OTM puts and calls) is the volatility smile.");

    // ═══════════════════════════════════════════════════════════
    // Part 3: Crypto Parameters (BTC)
    // ═══════════════════════════════════════════════════════════
    println!("\n--- Part 3: BTC Jump Diffusion ---\n");

    let btc_params = MertonParams::default_btc();
    let btc_spot = 50_000.0;

    let btc_call = merton_call_price(&btc_params, btc_spot);
    let btc_bs = bs_call(
        btc_spot,
        btc_params.strike,
        btc_params.maturity,
        btc_params.r,
        btc_params.sigma,
    );

    println!("BTC Parameters:");
    println!("  sigma={}, lambda={}, mu_j={}, sigma_j={}",
             btc_params.sigma, btc_params.lambda_j, btc_params.mu_j, btc_params.sigma_j);
    println!();
    println!("BTC ATM Call (Merton):  {:.2}", btc_call);
    println!("BTC ATM Call (BS):      {:.2}", btc_bs);
    println!("Jump Premium:           {:+.2} ({:.1}%)",
             btc_call - btc_bs,
             (btc_call - btc_bs) / btc_bs * 100.0);

    // ═══════════════════════════════════════════════════════════
    // Part 4: Parameter Estimation from Synthetic Data
    // ═══════════════════════════════════════════════════════════
    println!("\n--- Part 4: Parameter Estimation ---\n");

    let candles = generate_synthetic_data(100.0, 500, 0.20, 0.05, 5.0, -0.05, 0.10, 42);
    let returns = log_returns(&candles);
    let stats = compute_return_stats(&returns);
    let estimated = estimate_jump_params(&returns, 3.0, 252.0);

    println!("Synthetic data (500 days, true: sigma=0.20, lambda=5, mu_j=-0.05):");
    println!("  Annualized vol: {:.4}", stats.annualized_vol);
    println!("  Skewness:       {:.4}", stats.skewness);
    println!("  Kurtosis:       {:.4}", stats.kurtosis);
    println!();
    println!("Estimated: {}", estimated);

    // ═══════════════════════════════════════════════════════════
    // Part 5: PINN (untrained demo)
    // ═══════════════════════════════════════════════════════════
    println!("\n--- Part 5: PINN Structure (untrained) ---\n");

    let pinn = JumpDiffusionPINN::new(
        MertonParams::default_equity(),
        64,   // hidden_dim
        3,    // n_hidden
        8,    // n_quadrature
        true, // is_call
    );

    println!("PINN Architecture:");
    println!("  Total parameters: {}", pinn.n_params());
    println!("  Quadrature nodes: 8");

    // Evaluate at a few points (untrained, so values will be random)
    println!("\nUntrained PINN prices (for structure verification):");
    for &s in &[80.0, 90.0, 100.0, 110.0, 120.0] {
        let v = pinn.price(s, 0.0);
        let greeks = pinn.greeks(s, 0.0);
        println!("  S={:6.1}: V={:.4}, delta={:.4}, gamma={:.6}",
                 s, v, greeks.delta, greeks.gamma);
    }

    // PIDE residual
    let residual = pinn.pide_residual(100.0, 0.5);
    println!("\nPIDE residual at (S=100, t=0.5): {:.6}", residual);
    println!("(Non-zero because the network is untrained)");

    println!("\n====================================================");
    println!("  Quick Start Complete!");
    println!("====================================================");
    println!("\nNext steps:");
    println!("  1. Train the PINN: cargo run --release --bin train");
    println!("  2. Price options:  cargo run --bin price_options");
    println!("  3. Fetch data:     cargo run --bin fetch_data -- --synthetic");
}
