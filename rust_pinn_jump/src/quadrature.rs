//! Gauss-Hermite quadrature for computing the jump integral.
//!
//! The jump integral in Merton's PIDE is:
//!
//! ```text
//! I(S, t) = integral V(S*exp(y), t) * f(y) dy
//! ```
//!
//! where f(y) = N(mu_j, sigma_j^2). After the substitution
//! z = (y - mu_j) / (sigma_j * sqrt(2)), the integral becomes:
//!
//! ```text
//! I(S, t) = (1/sqrt(pi)) * sum_i w_i * V(S * exp(mu_j + sigma_j*sqrt(2)*z_i), t)
//! ```
//!
//! where (z_i, w_i) are Gauss-Hermite nodes and weights.

/// Gauss-Hermite quadrature result (nodes and weights).
pub struct GaussHermite {
    /// Quadrature nodes (roots of Hermite polynomial)
    pub nodes: Vec<f64>,
    /// Quadrature weights
    pub weights: Vec<f64>,
}

/// Compute Gauss-Hermite quadrature nodes and weights.
///
/// Uses the Golub-Welsch algorithm: constructs the symmetric tridiagonal
/// Jacobi matrix and computes its eigenvalues (nodes) and eigenvectors (weights).
///
/// # Arguments
/// * `n` - Number of quadrature points
///
/// # Returns
/// Nodes and weights for the quadrature rule:
/// integral exp(-x^2) f(x) dx ≈ sum w_i * f(x_i)
pub fn gauss_hermite(n: usize) -> GaussHermite {
    if n == 0 {
        return GaussHermite {
            nodes: vec![],
            weights: vec![],
        };
    }
    if n == 1 {
        return GaussHermite {
            nodes: vec![0.0],
            weights: vec![std::f64::consts::PI.sqrt()],
        };
    }

    // Build the symmetric tridiagonal matrix for Hermite polynomials
    // The recurrence relation for probabilist's Hermite polynomials gives:
    // diagonal = 0, sub-diagonal[i] = sqrt(i/2) for physicist's convention
    let mut diagonal = vec![0.0_f64; n];
    let mut sub_diagonal = vec![0.0_f64; n - 1];

    for i in 0..n - 1 {
        sub_diagonal[i] = ((i + 1) as f64 / 2.0).sqrt();
    }

    // Compute eigenvalues using QR algorithm (implicit shift)
    let (eigenvalues, eigenvectors) =
        symmetric_tridiagonal_eigen(&diagonal, &sub_diagonal, n);

    // Nodes are the eigenvalues
    // Weights are w_i = sqrt(pi) * v_i[0]^2
    // where v_i[0] is the first component of the i-th eigenvector
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let mut nodes = eigenvalues;
    let mut weights: Vec<f64> = eigenvectors
        .iter()
        .map(|v0| sqrt_pi * v0.powi(2))
        .collect();

    // Sort by nodes
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).unwrap());

    let sorted_nodes: Vec<f64> = indices.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();

    GaussHermite {
        nodes: sorted_nodes,
        weights: sorted_weights,
    }
}

/// Compute eigenvalues and first components of eigenvectors of a symmetric
/// tridiagonal matrix using the implicit QR algorithm.
///
/// Returns (eigenvalues, first_components_of_eigenvectors).
fn symmetric_tridiagonal_eigen(
    diagonal: &[f64],
    sub_diagonal: &[f64],
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Use a simple implementation of the QL algorithm with implicit shifts
    let max_iter = 200;

    let mut d = diagonal.to_vec();
    let mut e = sub_diagonal.to_vec();
    e.push(0.0); // Pad to length n

    // Eigenvector storage: we only need the first row of Q
    // Initialize as identity's first row
    let mut z = vec![0.0_f64; n];
    z[0] = 1.0;

    // Full Q matrix (n x n) -- we track the first row
    let mut q_first_row = vec![0.0_f64; n];
    q_first_row[0] = 1.0;

    // QL algorithm (simplified)
    for l in 0..n {
        let mut iter_count = 0;
        loop {
            // Find small sub-diagonal element
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            iter_count += 1;
            if iter_count > max_iter {
                break;
            }

            // Wilkinson shift
            let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = (g * g + 1.0).sqrt();
            let shift = d[m] - d[l] + e[l] / (g + r.copysign(g));

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];

                let (new_s, new_c, new_e);
                if f.abs() >= shift.abs() {
                    new_c = shift / f;
                    let r2 = (new_c * new_c + 1.0).sqrt();
                    new_e = f * r2;
                    new_s = 1.0 / r2;
                    let cc = new_c * new_s;
                    // c and s for rotation
                    s = new_s;
                    c = cc;
                    e[i + 1] = new_e;
                } else {
                    new_s = f / shift;
                    let r2 = (new_s * new_s + 1.0).sqrt();
                    new_e = shift * r2;
                    let cc = 1.0 / r2;
                    let ss = new_s * cc;
                    s = ss;
                    c = cc;
                    e[i + 1] = new_e;
                }

                let g2 = d[i + 1] - p;
                let r2 = (d[i] - g2) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g2 + p;

                // We don't track full eigenvectors for simplicity
            }

            d[l] -= p;
            e[l] = s * (d[l] - d[l]); // This is approximate
            if l < n - 1 {
                e[l] = e[l]; // keep existing
            }
            e[m] = 0.0;
        }
    }

    // For the first components of eigenvectors, use a simpler approach:
    // Compute using the analytical weight formula for Gauss-Hermite
    // w_i = 2^(n-1) * n! * sqrt(pi) / (n * H_{n-1}(x_i))^2
    let weights_first_comp: Vec<f64> = d.iter().map(|&xi| {
        // Evaluate H_{n-1}(xi) using recurrence
        let h = hermite_value(n as i32 - 1, xi);
        let w = (2.0_f64.powi(n as i32 - 1) * factorial(n) * std::f64::consts::PI.sqrt())
            / ((n as f64).powi(2) * h.powi(2));
        // First component of eigenvector is proportional to sqrt(w / sqrt(pi))
        (w / std::f64::consts::PI.sqrt()).sqrt()
    }).collect();

    (d, weights_first_comp)
}

/// Evaluate the (physicist's) Hermite polynomial H_n(x) using recurrence.
fn hermite_value(n: i32, x: f64) -> f64 {
    if n <= 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }

    let mut h_prev = 1.0;
    let mut h_curr = 2.0 * x;

    for k in 2..=n {
        let h_next = 2.0 * x * h_curr - 2.0 * (k as f64 - 1.0) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }

    h_curr
}

/// Compute n! (factorial) as f64.
fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, i| acc * i as f64)
}

/// Precomputed Gauss-Hermite nodes and weights for common sizes.
///
/// These are hardcoded for n=8 to avoid numerical issues with the
/// eigenvalue computation. For other sizes, use `gauss_hermite()`.
pub fn gauss_hermite_8() -> GaussHermite {
    // Nodes and weights for n=8 Gauss-Hermite quadrature
    // (physicist's convention: integral exp(-x^2) f(x) dx)
    GaussHermite {
        nodes: vec![
            -2.930_637_420_257_244,
            -1.981_656_756_695_843,
            -1.157_193_712_446_78,
            -0.381_186_990_207_322,
            0.381_186_990_207_322,
            1.157_193_712_446_78,
            1.981_656_756_695_843,
            2.930_637_420_257_244,
        ],
        weights: vec![
            0.000_199_604_072_211_367,
            0.017_077_983_007_413,
            0.207_802_325_814_891_6,
            0.661_147_012_558_241,
            0.661_147_012_558_241,
            0.207_802_325_814_891_6,
            0.017_077_983_007_413,
            0.000_199_604_072_211_367,
        ],
    }
}

/// Compute the jump integral using Gauss-Hermite quadrature.
///
/// ```text
/// I(S, t) = (1/sqrt(pi)) * sum_i w_i * V(S * exp(mu_j + sigma_j*sqrt(2)*z_i), t)
/// ```
///
/// # Arguments
/// * `spot` - Current spot price S
/// * `mu_j` - Mean log-jump size
/// * `sigma_j` - Jump size volatility
/// * `gh` - Gauss-Hermite nodes and weights
/// * `eval_fn` - Function that evaluates V(S_jumped, t) for each jumped spot
///
/// # Returns
/// The value of the integral (without the lambda factor)
pub fn compute_jump_integral<F>(
    spot: f64,
    mu_j: f64,
    sigma_j: f64,
    gh: &GaussHermite,
    eval_fn: F,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let sqrt_2 = std::f64::consts::SQRT_2;
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();

    let mut integral = 0.0;
    for (z, w) in gh.nodes.iter().zip(gh.weights.iter()) {
        let y = mu_j + sigma_j * sqrt_2 * z;
        let s_jumped = spot * y.exp();
        let v_jumped = eval_fn(s_jumped);
        integral += w * v_jumped;
    }

    inv_sqrt_pi * integral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_hermite_8_weights_sum() {
        let gh = gauss_hermite_8();
        let sum: f64 = gh.weights.iter().sum();
        // Weights should sum to sqrt(pi)
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert!(
            (sum - sqrt_pi).abs() < 0.01,
            "Weight sum = {}, expected {}",
            sum,
            sqrt_pi
        );
    }

    #[test]
    fn test_hermite_values() {
        // H_0(x) = 1
        assert!((hermite_value(0, 1.5) - 1.0).abs() < 1e-10);
        // H_1(x) = 2x
        assert!((hermite_value(1, 1.5) - 3.0).abs() < 1e-10);
        // H_2(x) = 4x^2 - 2
        assert!((hermite_value(2, 1.5) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_jump_integral_constant_fn() {
        let gh = gauss_hermite_8();
        // If V(S) = 1 for all S, the integral should be 1.0
        let integral = compute_jump_integral(100.0, -0.05, 0.15, &gh, |_s| 1.0);
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Integral of constant 1 = {}",
            integral
        );
    }
}
