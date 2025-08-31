"""
Greeks computation for the Jump Diffusion PINN via automatic differentiation.

PINNs provide Greeks "for free" since the network is differentiable:
    Delta = dV/dS
    Gamma = d2V/dS2
    Theta = dV/dt
    Vega  = dV/dsigma (requires sigma as input or finite differences)
    Jump-specific: dV/d(lambda), dV/d(mu_J), dV/d(sigma_J)

Usage:
    model = JumpDiffusionPINN(params)
    greeks = compute_greeks(model, S=100.0, t=0.0)
    print(greeks)
"""

from typing import Dict, Optional

import numpy as np
import torch

from jump_diffusion_pinn import JumpDiffusionPINN, MertonParams


def compute_greeks(
    model: JumpDiffusionPINN,
    S: float,
    t: float = 0.0,
    dS: float = 0.01,
) -> Dict[str, float]:
    """
    Compute option Greeks using automatic differentiation.

    First and second order derivatives are computed via torch.autograd.
    This is exact (no finite difference errors) for dV/dS, d2V/dS2, dV/dt.

    Args:
        model: Trained PINN model
        S: Spot price
        t: Current time
        dS: Not used for autograd Greeks (kept for API compatibility)

    Returns:
        Dictionary with Greeks: delta, gamma, theta, speed, charm
    """
    model.eval()

    S_tensor = torch.tensor([S], dtype=torch.float32, requires_grad=True)
    t_tensor = torch.tensor([t], dtype=torch.float32, requires_grad=True)

    # Forward pass
    V = model.forward(S_tensor, t_tensor)
    value = V.item()

    # Delta = dV/dS
    dV_dS = torch.autograd.grad(
        V, S_tensor,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]
    delta = dV_dS.item()

    # Gamma = d2V/dS2
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S_tensor,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=True,
        retain_graph=True,
    )[0]
    gamma = d2V_dS2.item()

    # Theta = dV/dt (note: convention is often -dV/dt for time decay)
    dV_dt = torch.autograd.grad(
        V, t_tensor,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]
    theta = dV_dt.item()

    # Speed = d3V/dS3 (third derivative)
    try:
        d3V_dS3 = torch.autograd.grad(
            d2V_dS2, S_tensor,
            grad_outputs=torch.ones_like(d2V_dS2),
            create_graph=True,
            retain_graph=True,
        )[0]
        speed = d3V_dS3.item()
    except RuntimeError:
        speed = 0.0

    # Charm = d2V/(dS dt)
    try:
        d2V_dSdt = torch.autograd.grad(
            dV_dS, t_tensor,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=False,
            retain_graph=True,
        )[0]
        charm = d2V_dSdt.item()
    except RuntimeError:
        charm = 0.0

    return {
        "value": value,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "speed": speed,
        "charm": charm,
    }


def compute_greeks_grid(
    model: JumpDiffusionPINN,
    S_values: np.ndarray,
    t: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Compute Greeks for an array of spot prices (vectorized).

    Args:
        model: Trained PINN model
        S_values: Array of spot prices
        t: Current time

    Returns:
        Dictionary mapping Greek names to arrays
    """
    model.eval()

    n = len(S_values)
    S_tensor = torch.tensor(S_values, dtype=torch.float32, requires_grad=True)
    t_tensor = torch.full((n,), t, dtype=torch.float32, requires_grad=True)

    # Forward
    V = model.forward(S_tensor, t_tensor)

    # Delta
    dV_dS = torch.autograd.grad(
        V, S_tensor,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gamma
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S_tensor,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Theta
    dV_dt = torch.autograd.grad(
        V, t_tensor,
        grad_outputs=torch.ones_like(V),
        create_graph=False,
        retain_graph=True,
    )[0]

    return {
        "value": V.detach().numpy(),
        "delta": dV_dS.detach().numpy(),
        "gamma": d2V_dS2.detach().numpy(),
        "theta": dV_dt.detach().numpy(),
    }


def compute_jump_greeks(
    model: JumpDiffusionPINN,
    S: float,
    t: float = 0.0,
    d_lambda: float = 0.01,
    d_mu_j: float = 0.001,
    d_sigma_j: float = 0.001,
) -> Dict[str, float]:
    """
    Compute jump-specific Greeks via finite differences.

    These sensitivities measure how the option value changes with
    jump parameters -- critical for jump risk hedging.

    Args:
        model: Trained PINN model
        S: Spot price
        t: Current time
        d_lambda: Perturbation for lambda_j
        d_mu_j: Perturbation for mu_j
        d_sigma_j: Perturbation for sigma_j

    Returns:
        Dictionary with jump Greeks: dV_dlambda, dV_dmu_j, dV_dsigma_j
    """
    base_price = model.price(S, t)
    original_params = model.params

    results = {}

    # dV/d(lambda_j) via central finite differences
    original_lambda = original_params.lambda_j
    original_params.lambda_j = original_lambda + d_lambda
    price_up = model.price(S, t)
    original_params.lambda_j = original_lambda - d_lambda
    price_down = model.price(S, t)
    original_params.lambda_j = original_lambda
    results["dV_dlambda"] = (price_up - price_down) / (2 * d_lambda)

    # dV/d(mu_j)
    original_mu = original_params.mu_j
    original_params.mu_j = original_mu + d_mu_j
    price_up = model.price(S, t)
    original_params.mu_j = original_mu - d_mu_j
    price_down = model.price(S, t)
    original_params.mu_j = original_mu
    results["dV_dmu_j"] = (price_up - price_down) / (2 * d_mu_j)

    # dV/d(sigma_j)
    original_sigma_j = original_params.sigma_j
    original_params.sigma_j = original_sigma_j + d_sigma_j
    price_up = model.price(S, t)
    original_params.sigma_j = original_sigma_j - d_sigma_j
    price_down = model.price(S, t)
    original_params.sigma_j = original_sigma_j
    results["dV_dsigma_j"] = (price_up - price_down) / (2 * d_sigma_j)

    results["value"] = base_price
    return results


def greek_surface(
    model: JumpDiffusionPINN,
    S_range: tuple = (60.0, 150.0),
    t_range: tuple = (0.0, 0.9),
    n_S: int = 50,
    n_t: int = 20,
    greek: str = "delta",
) -> tuple:
    """
    Compute a Greek surface over (S, t) grid.

    Args:
        model: Trained PINN model
        S_range: (S_min, S_max) for the grid
        t_range: (t_min, t_max) for the grid
        n_S: Number of spot points
        n_t: Number of time points
        greek: Which Greek to compute ("delta", "gamma", "theta")

    Returns:
        Tuple of (S_grid, t_grid, greek_grid) -- all 2D arrays
    """
    S_values = np.linspace(S_range[0], S_range[1], n_S)
    t_values = np.linspace(t_range[0], t_range[1], n_t)

    greek_grid = np.zeros((n_S, n_t))

    for j, t in enumerate(t_values):
        greeks = compute_greeks_grid(model, S_values, t)
        greek_grid[:, j] = greeks[greek]

    SS, TT = np.meshgrid(S_values, t_values, indexing="ij")
    return SS, TT, greek_grid


def main():
    """Demo: compute Greeks for a sample PINN."""
    params = MertonParams(sigma=0.20, r=0.05, lambda_j=5.0, mu_j=-0.05, sigma_j=0.15)
    model = JumpDiffusionPINN(params)

    print("Greeks at S=100 (untrained model -- for structure demo):")
    greeks = compute_greeks(model, S=100.0, t=0.0)
    for name, value in greeks.items():
        print(f"  {name:>10}: {value:.6f}")

    print("\nJump Greeks:")
    jump_greeks = compute_jump_greeks(model, S=100.0, t=0.0)
    for name, value in jump_greeks.items():
        print(f"  {name:>15}: {value:.6f}")


if __name__ == "__main__":
    main()
