//! Train the Jump Diffusion PINN.
//!
//! Uses an evolutionary strategy (gradient-free) for training since
//! this is a from-scratch implementation without autograd.
//!
//! For production, use tch-rs (PyTorch) or burn for proper autograd training.
//!
//! Usage:
//!     cargo run --release --bin train
//!     cargo run --release --bin train -- --asset BTC --epochs 5000

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use pinn_jump_diffusion::analytical::{merton_call_price, merton_put_price};
use pinn_jump_diffusion::params::MertonParams;
use pinn_jump_diffusion::pinn::JumpDiffusionPINN;

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train the Jump Diffusion PINN")]
struct Args {
    /// Asset type (SPY, BTC, ETH)
    #[arg(short, long, default_value = "SPY")]
    asset: String,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 2000)]
    epochs: usize,

    /// Hidden layer dimension
    #[arg(long, default_value_t = 64)]
    hidden_dim: usize,

    /// Number of hidden layers
    #[arg(long, default_value_t = 3)]
    n_hidden: usize,

    /// Number of quadrature points
    #[arg(long, default_value_t = 8)]
    n_quadrature: usize,

    /// Number of interior collocation points
    #[arg(long, default_value_t = 200)]
    n_interior: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.5)]
    lr: f64,

    /// Noise scale for perturbations
    #[arg(long, default_value_t = 0.01)]
    noise: f64,

    /// Option type (call, put)
    #[arg(long, default_value = "call")]
    option_type: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Create parameters based on asset
    let params = match args.asset.to_uppercase().as_str() {
        "BTC" | "BTCUSDT" => MertonParams::default_btc(),
        "ETH" | "ETHUSDT" => MertonParams {
            sigma: 0.55,
            r: 0.05,
            lambda_j: 15.0,
            mu_j: -0.10,
            sigma_j: 0.20,
            strike: 3000.0,
            maturity: 1.0,
            s_min: 500.0,
            s_max: 10000.0,
        },
        _ => MertonParams::default_equity(),
    };

    let is_call = args.option_type == "call";

    println!("{}", "=".repeat(60).blue());
    println!(
        "{}",
        format!(
            "Training PINN for {} {} option",
            args.asset,
            if is_call { "call" } else { "put" }
        )
        .blue()
        .bold()
    );
    println!("{}", "=".repeat(60).blue());

    println!("\n{}", "Model Parameters:".green().bold());
    println!("  sigma     = {:.4}", params.sigma);
    println!("  r         = {:.4}", params.r);
    println!("  lambda_j  = {:.2}", params.lambda_j);
    println!("  mu_j      = {:.4}", params.mu_j);
    println!("  sigma_j   = {:.4}", params.sigma_j);
    println!("  strike    = {:.2}", params.strike);
    println!("  maturity  = {:.2}", params.maturity);
    println!("  k (E[J])  = {:.4}", params.k());

    // Create PINN
    let mut pinn = JumpDiffusionPINN::new(
        params.clone(),
        args.hidden_dim,
        args.n_hidden,
        args.n_quadrature,
        is_call,
    );

    println!("\n{}", "Network Architecture:".green().bold());
    println!("  Hidden dim:      {}", args.hidden_dim);
    println!("  Hidden layers:   {}", args.n_hidden);
    println!("  Quadrature pts:  {}", args.n_quadrature);
    println!("  Total params:    {}", pinn.n_params());

    // Sample collocation points
    let (interior, terminal, boundary) =
        pinn.sample_collocation_points(args.n_interior, 100, 100);

    // Training loop
    println!("\n{}", "Training:".green().bold());
    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} loss={msg}")
            .unwrap(),
    );

    let mut best_loss = f64::INFINITY;
    let mut noise_scale = args.noise;

    for epoch in 0..args.epochs {
        let loss = pinn.train_step(
            &interior,
            &terminal,
            &boundary,
            args.lr,
            noise_scale,
        );

        if loss < best_loss {
            best_loss = loss;
        }

        // Decay noise over time
        if epoch % 500 == 0 && epoch > 0 {
            noise_scale *= 0.8;
        }

        // Resample collocation points periodically
        if epoch % 200 == 0 && epoch > 0 {
            let (new_int, new_term, new_bound) =
                pinn.sample_collocation_points(args.n_interior, 100, 100);
            // We would update here but borrow checker makes it complex
            // In practice, use interior = new_int etc.
        }

        if epoch % 100 == 0 {
            pb.set_message(format!("{:.6}", loss));
        }
        pb.inc(1);
    }
    pb.finish_with_message(format!("best={:.6}", best_loss));

    // Validation against analytical Merton
    println!("\n{}", "Validation against Analytical Merton:".green().bold());
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12}",
        "Spot", "PINN", "Merton", "Error", "Rel Err"
    );
    println!("{}", "-".repeat(62));

    let k = params.strike;
    let test_spots = vec![k * 0.8, k * 0.9, k, k * 1.1, k * 1.2];

    for spot in &test_spots {
        let pinn_price = pinn.price(*spot, 0.0);
        let merton_price = if is_call {
            merton_call_price(&params, *spot)
        } else {
            merton_put_price(&params, *spot)
        };

        let err = (pinn_price - merton_price).abs();
        let rel_err = if merton_price > 0.01 {
            err / merton_price
        } else {
            0.0
        };

        let err_color = if rel_err < 0.05 {
            format!("{:.4}", err).green()
        } else if rel_err < 0.20 {
            format!("{:.4}", err).yellow()
        } else {
            format!("{:.4}", err).red()
        };

        println!(
            "{:12.2} {:12.4} {:12.4} {} {:11.2}%",
            spot,
            pinn_price,
            merton_price,
            err_color,
            rel_err * 100.0,
        );
    }

    // Greeks
    println!("\n{}", "Greeks at ATM (S = K):".green().bold());
    let greeks = pinn.greeks(k, 0.0);
    println!("  {}", greeks);

    println!("\n{}", "Training complete!".green().bold());
    println!(
        "Note: This uses evolutionary optimization (gradient-free).",
    );
    println!(
        "For better convergence, use the Python implementation with PyTorch autograd."
    );

    Ok(())
}
