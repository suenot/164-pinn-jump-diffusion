//! # PINN for Jump Diffusion Option Pricing
//!
//! Physics-Informed Neural Network implementation for solving Merton's
//! Jump Diffusion Partial Integro-Differential Equation (PIDE).
//!
//! ## PIDE
//!
//! ```text
//! dV/dt + 0.5*sigma^2*S^2 * d2V/dS2 + (r - lambda*k)*S * dV/dS
//! - (r + lambda)*V + lambda * integral[V(S*e^y, t) * f(y) dy] = 0
//! ```
//!
//! The integral is computed via Gauss-Hermite quadrature.
//!
//! ## Modules
//!
//! - `params`: Merton jump diffusion parameters
//! - `analytical`: Analytical Merton pricing (series of BS prices)
//! - `pinn`: Neural network and PIDE residual computation
//! - `quadrature`: Gauss-Hermite quadrature nodes and weights
//! - `data`: Data fetching from Bybit and Yahoo Finance
//! - `backtest`: Strategy backtesting

pub mod analytical;
pub mod backtest;
pub mod data;
pub mod params;
pub mod pinn;
pub mod quadrature;

pub use params::MertonParams;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::analytical::{merton_call_price, merton_put_price};
    pub use crate::params::MertonParams;
    pub use crate::pinn::JumpDiffusionPINN;
    pub use crate::quadrature::gauss_hermite;
}
