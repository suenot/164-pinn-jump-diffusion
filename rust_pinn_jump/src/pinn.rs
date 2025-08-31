//! Physics-Informed Neural Network for Merton's Jump Diffusion PIDE.
//!
//! Implements a simple feed-forward neural network with:
//! - Input preprocessing (log-moneyness, normalized time)
//! - Hidden layers with tanh activation
//! - Softplus output to ensure V >= 0
//! - PIDE residual computation with Gauss-Hermite quadrature for the integral
//!
//! Note: This is a from-scratch implementation for educational purposes.
//! For production use, consider using `tch-rs` (PyTorch bindings) or `burn`.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::params::MertonParams;
use crate::quadrature::{compute_jump_integral, gauss_hermite, GaussHermite};

/// A single dense layer: y = activation(W*x + b)
#[derive(Clone)]
pub struct DenseLayer {
    /// Weight matrix: (out_features, in_features)
    pub weights: Array2<f64>,
    /// Bias vector: (out_features,)
    pub bias: Array1<f64>,
    /// Activation function type
    pub activation: Activation,
}

/// Activation function options.
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Tanh,
    Softplus,
    Linear,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    pub fn new(in_features: usize, out_features: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();

        let weights = Array2::from_shape_fn((out_features, in_features), |_| {
            rng.sample::<f64, _>(StandardNormal) * scale
        });
        let bias = Array1::zeros(out_features);

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Forward pass: y = activation(W*x + b)
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(input) + &self.bias;
        match self.activation {
            Activation::Tanh => z.mapv(|x| x.tanh()),
            Activation::Softplus => z.mapv(|x| (1.0 + x.exp()).ln().max(0.0)),
            Activation::Linear => z,
        }
    }

    /// Number of parameters in this layer.
    pub fn n_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Physics-Informed Neural Network for jump diffusion option pricing.
///
/// Architecture:
/// ```text
/// Input: (log_moneyness, normalized_tau) -- 2D
///   -> Dense(2, hidden_dim) + tanh
///   -> Dense(hidden_dim, hidden_dim) + tanh  (x n_hidden)
///   -> Dense(hidden_dim, 1) + softplus
/// ```
pub struct JumpDiffusionPINN {
    /// Model parameters
    pub params: MertonParams,
    /// Network layers
    pub layers: Vec<DenseLayer>,
    /// Gauss-Hermite quadrature for jump integral
    pub quadrature: GaussHermite,
    /// Option type: true = call, false = put
    pub is_call: bool,
}

impl JumpDiffusionPINN {
    /// Create a new PINN with the given architecture.
    ///
    /// # Arguments
    /// * `params` - Merton model parameters
    /// * `hidden_dim` - Width of hidden layers
    /// * `n_hidden` - Number of hidden layers
    /// * `n_quadrature` - Number of Gauss-Hermite quadrature points
    /// * `is_call` - true for call, false for put
    pub fn new(
        params: MertonParams,
        hidden_dim: usize,
        n_hidden: usize,
        n_quadrature: usize,
        is_call: bool,
    ) -> Self {
        let mut layers = Vec::new();

        // Input layer: 2 -> hidden_dim
        layers.push(DenseLayer::new(2, hidden_dim, Activation::Tanh));

        // Hidden layers: hidden_dim -> hidden_dim
        for _ in 0..n_hidden {
            layers.push(DenseLayer::new(hidden_dim, hidden_dim, Activation::Tanh));
        }

        // Output layer: hidden_dim -> 1 with softplus
        layers.push(DenseLayer::new(hidden_dim, 1, Activation::Softplus));

        let quadrature = gauss_hermite(n_quadrature);

        Self {
            params,
            layers,
            quadrature,
            is_call,
        }
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Preprocess raw inputs (S, t) to network inputs (log_moneyness, normalized_tau).
    fn preprocess(&self, spot: f64, time: f64) -> Array1<f64> {
        let log_moneyness = (spot / self.params.strike).ln();
        let normalized_tau = (self.params.maturity - time) / self.params.maturity.max(1e-10);
        Array1::from(vec![log_moneyness, normalized_tau])
    }

    /// Forward pass: compute V(S, t).
    pub fn forward(&self, spot: f64, time: f64) -> f64 {
        let mut x = self.preprocess(spot, time);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x[0] // Single output
    }

    /// Price an option.
    pub fn price(&self, spot: f64, time: f64) -> f64 {
        self.forward(spot, time)
    }

    /// Compute numerical derivatives dV/dS and d2V/dS2 via finite differences.
    ///
    /// Returns (dV/dS, d2V/dS2, dV/dt)
    pub fn compute_derivatives(&self, spot: f64, time: f64, h: f64) -> (f64, f64, f64) {
        let v = self.forward(spot, time);

        // dV/dS via central differences
        let v_up = self.forward(spot + h, time);
        let v_down = self.forward(spot - h, time);
        let dv_ds = (v_up - v_down) / (2.0 * h);

        // d2V/dS2 via central differences
        let d2v_ds2 = (v_up - 2.0 * v + v_down) / (h * h);

        // dV/dt via central differences
        let dt = h * 0.01; // Smaller step for time
        let v_t_up = self.forward(spot, (time + dt).min(self.params.maturity));
        let v_t_down = self.forward(spot, (time - dt).max(0.0));
        let dv_dt = (v_t_up - v_t_down) / (2.0 * dt);

        (dv_ds, d2v_ds2, dv_dt)
    }

    /// Compute the PIDE residual at point (S, t).
    ///
    /// ```text
    /// R = dV/dt + 0.5*sigma^2*S^2 * d2V/dS2 + (r - lambda*k)*S * dV/dS
    ///     - (r + lambda)*V + lambda * Integral
    /// ```
    pub fn pide_residual(&self, spot: f64, time: f64) -> f64 {
        let p = &self.params;
        let k = p.k();
        let h = spot * 1e-4; // Step size for finite differences

        let v = self.forward(spot, time);
        let (dv_ds, d2v_ds2, dv_dt) = self.compute_derivatives(spot, time, h);

        // Jump integral
        let integral = compute_jump_integral(
            spot,
            p.mu_j,
            p.sigma_j,
            &self.quadrature,
            |s_jumped| {
                let s_clamped = s_jumped.clamp(p.s_min, p.s_max);
                self.forward(s_clamped, time)
            },
        );

        // PIDE residual
        let diffusion = 0.5 * p.sigma.powi(2) * spot.powi(2) * d2v_ds2;
        let drift = (p.r - p.lambda_j * k) * spot * dv_ds;
        let discount = -(p.r + p.lambda_j) * v;
        let jump = p.lambda_j * integral;

        dv_dt + diffusion + drift + discount + jump
    }

    /// Compute terminal condition error.
    pub fn terminal_error(&self, spot: f64) -> f64 {
        let v_pred = self.forward(spot, self.params.maturity);
        let payoff = if self.is_call {
            (spot - self.params.strike).max(0.0)
        } else {
            (self.params.strike - spot).max(0.0)
        };
        v_pred - payoff
    }

    /// Compute boundary condition value.
    pub fn boundary_value(&self, spot: f64, time: f64) -> f64 {
        let p = &self.params;
        let tau = p.maturity - time;

        if self.is_call {
            if spot <= p.s_min {
                0.0
            } else {
                spot - p.strike * (-p.r * tau).exp()
            }
        } else {
            if spot <= p.s_min {
                p.strike * (-p.r * tau).exp()
            } else {
                0.0
            }
        }
    }

    /// Simple training step using gradient-free optimization (evolutionary strategy).
    ///
    /// Perturbs weights randomly and keeps changes that reduce the loss.
    /// This is a simplified training approach for demonstration.
    /// For production, use proper autograd (e.g., via tch-rs or burn).
    pub fn train_step(
        &mut self,
        collocation_points: &[(f64, f64)],
        terminal_points: &[f64],
        boundary_points: &[(f64, f64)],
        learning_rate: f64,
        noise_scale: f64,
    ) -> f64 {
        // Compute current loss
        let current_loss = self.compute_loss(collocation_points, terminal_points, boundary_points);

        // Random perturbation
        let mut rng = rand::thread_rng();
        let mut best_loss = current_loss;

        // Try several random perturbations
        let n_trials = 5;
        let mut best_layers: Option<Vec<DenseLayer>> = None;

        for _ in 0..n_trials {
            // Save current state
            let saved_layers = self.layers.clone();

            // Perturb all weights
            for layer in &mut self.layers {
                for w in layer.weights.iter_mut() {
                    *w += rng.sample::<f64, _>(StandardNormal) * noise_scale;
                }
                for b in layer.bias.iter_mut() {
                    *b += rng.sample::<f64, _>(StandardNormal) * noise_scale;
                }
            }

            // Compute new loss
            let new_loss = self.compute_loss(collocation_points, terminal_points, boundary_points);

            if new_loss < best_loss {
                best_loss = new_loss;
                best_layers = Some(self.layers.clone());
            }

            // Restore state
            self.layers = saved_layers;
        }

        // Apply best perturbation if it improved
        if let Some(better) = best_layers {
            // Interpolate toward the better point (like a learning rate)
            for (current, better_layer) in self.layers.iter_mut().zip(better.iter()) {
                for (cw, bw) in current.weights.iter_mut().zip(better_layer.weights.iter()) {
                    *cw += learning_rate * (bw - *cw);
                }
                for (cb, bb) in current.bias.iter_mut().zip(better_layer.bias.iter()) {
                    *cb += learning_rate * (bb - *cb);
                }
            }
        }

        best_loss
    }

    /// Compute total loss across all training points.
    fn compute_loss(
        &self,
        collocation_points: &[(f64, f64)],
        terminal_points: &[f64],
        boundary_points: &[(f64, f64)],
    ) -> f64 {
        let n_pide = collocation_points.len().max(1) as f64;
        let n_ic = terminal_points.len().max(1) as f64;
        let n_bc = boundary_points.len().max(1) as f64;

        // PIDE residual loss
        let l_pide: f64 = collocation_points
            .iter()
            .map(|&(s, t)| self.pide_residual(s, t).powi(2))
            .sum::<f64>()
            / n_pide;

        // Terminal condition loss
        let l_ic: f64 = terminal_points
            .iter()
            .map(|&s| self.terminal_error(s).powi(2))
            .sum::<f64>()
            / n_ic;

        // Boundary condition loss
        let l_bc: f64 = boundary_points
            .iter()
            .map(|&(s, t)| {
                let v_pred = self.forward(s, t);
                let v_exact = self.boundary_value(s, t);
                (v_pred - v_exact).powi(2)
            })
            .sum::<f64>()
            / n_bc;

        // Weighted sum (adaptive weights would be better)
        1.0 * l_pide + 10.0 * l_ic + 10.0 * l_bc
    }

    /// Generate random collocation points for training.
    pub fn sample_collocation_points(
        &self,
        n_interior: usize,
        n_terminal: usize,
        n_boundary: usize,
    ) -> (Vec<(f64, f64)>, Vec<f64>, Vec<(f64, f64)>) {
        let mut rng = rand::thread_rng();
        let p = &self.params;

        let log_s_min = p.s_min.ln();
        let log_s_max = p.s_max.ln();

        // Interior points
        let interior: Vec<(f64, f64)> = (0..n_interior)
            .map(|_| {
                let log_s = rng.gen::<f64>() * (log_s_max - log_s_min) + log_s_min;
                let s = log_s.exp();
                let t = rng.gen::<f64>() * p.maturity * 0.999;
                (s, t)
            })
            .collect();

        // Terminal points (t = T)
        let terminal: Vec<f64> = (0..n_terminal)
            .map(|_| {
                let log_s = rng.gen::<f64>() * (log_s_max - log_s_min) + log_s_min;
                log_s.exp()
            })
            .collect();

        // Boundary points (S = S_min or S = S_max)
        let boundary: Vec<(f64, f64)> = (0..n_boundary)
            .map(|i| {
                let t = rng.gen::<f64>() * p.maturity;
                let s = if i < n_boundary / 2 { p.s_min } else { p.s_max };
                (s, t)
            })
            .collect();

        (interior, terminal, boundary)
    }

    /// Compute Greeks via finite differences.
    ///
    /// Returns a struct with delta, gamma, theta.
    pub fn greeks(&self, spot: f64, time: f64) -> Greeks {
        let h = spot * 1e-4;
        let (dv_ds, d2v_ds2, dv_dt) = self.compute_derivatives(spot, time, h);

        Greeks {
            value: self.forward(spot, time),
            delta: dv_ds,
            gamma: d2v_ds2,
            theta: dv_dt,
        }
    }
}

/// Option Greeks computed by the PINN.
#[derive(Debug, Clone)]
pub struct Greeks {
    pub value: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
}

impl std::fmt::Display for Greeks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Greeks(value={:.4}, delta={:.4}, gamma={:.6}, theta={:.4})",
            self.value, self.delta, self.gamma, self.theta
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinn_creation() {
        let params = MertonParams::default_equity();
        let pinn = JumpDiffusionPINN::new(params, 32, 2, 8, true);
        assert!(pinn.n_params() > 0);
    }

    #[test]
    fn test_pinn_forward() {
        let params = MertonParams::default_equity();
        let pinn = JumpDiffusionPINN::new(params, 32, 2, 8, true);

        let v = pinn.forward(100.0, 0.0);
        // Should be non-negative (softplus output)
        assert!(v >= 0.0, "V = {}", v);
    }

    #[test]
    fn test_pinn_residual() {
        let params = MertonParams::default_equity();
        let pinn = JumpDiffusionPINN::new(params, 32, 2, 8, true);

        // Residual should be a finite number (network is untrained so it won't be zero)
        let r = pinn.pide_residual(100.0, 0.5);
        assert!(r.is_finite(), "Residual = {}", r);
    }

    #[test]
    fn test_sample_collocation() {
        let params = MertonParams::default_equity();
        let pinn = JumpDiffusionPINN::new(params, 32, 2, 8, true);

        let (interior, terminal, boundary) = pinn.sample_collocation_points(100, 50, 50);
        assert_eq!(interior.len(), 100);
        assert_eq!(terminal.len(), 50);
        assert_eq!(boundary.len(), 50);
    }
}
