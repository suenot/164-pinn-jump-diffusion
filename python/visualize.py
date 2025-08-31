"""
Visualization module for Jump Diffusion PINN results.

Generates:
    - Volatility smile plots
    - Option value surfaces
    - PINN vs Analytical comparison
    - Greek surfaces
    - Training loss curves
    - Jump impact analysis
    - Return distribution with fitted model

Usage:
    python visualize.py --plot smile
    python visualize.py --plot surface
    python visualize.py --plot all
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jump_diffusion_pinn import JumpDiffusionPINN, MertonParams
from merton_analytical import (
    implied_volatility,
    merton_call_price,
    merton_implied_vol_smile,
    merton_price_grid,
)
from greeks import compute_greeks_grid, greek_surface


# Configure matplotlib
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
})


def plot_volatility_smile(
    params: MertonParams,
    T_values: Optional[list] = None,
    save_path: Optional[str] = None,
):
    """
    Plot the implied volatility smile under Merton's jump diffusion.

    Shows how IV varies across strikes for different maturities,
    demonstrating the smile/skew produced by jump risk.

    Args:
        params: Merton model parameters
        T_values: List of maturities to plot
        save_path: Optional file path to save the figure
    """
    if T_values is None:
        T_values = [0.1, 0.25, 0.5, 1.0]

    S = params.K
    K_range = np.linspace(params.K * 0.7, params.K * 1.3, 80)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(T_values)))

    for T, color in zip(T_values, colors):
        iv_smile = merton_implied_vol_smile(
            S, K_range, T, params.r, params.sigma,
            params.lambda_j, params.mu_j, params.sigma_j,
        )
        moneyness = K_range / S
        valid = ~np.isnan(iv_smile)
        ax.plot(
            moneyness[valid], iv_smile[valid] * 100,
            color=color, label=f"T = {T:.2f}y",
        )

    # Add Black-Scholes flat line for reference
    ax.axhline(
        y=params.sigma * 100, color="gray", linestyle="--",
        alpha=0.5, label=f"BS vol = {params.sigma*100:.0f}%",
    )

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("Volatility Smile under Merton Jump Diffusion")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_option_surface(
    params: MertonParams,
    option_type: str = "call",
    save_path: Optional[str] = None,
):
    """
    Plot the 3D option value surface V(S, t).

    Args:
        params: Merton model parameters
        option_type: "call" or "put"
        save_path: Optional file path to save the figure
    """
    S_values = np.linspace(params.S_min * 1.2, params.S_max * 0.8, 60)
    t_values = np.linspace(0.0, params.T * 0.95, 40)

    V_grid = merton_price_grid(
        S_values, t_values, params.K, params.T, params.r,
        params.sigma, params.lambda_j, params.mu_j, params.sigma_j,
        option_type=option_type,
    )

    SS, TT = np.meshgrid(S_values, t_values, indexing="ij")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        SS, TT, V_grid,
        cmap="viridis", alpha=0.8,
        linewidth=0, antialiased=True,
    )

    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Time (t)")
    ax.set_zlabel(f"{option_type.capitalize()} Value")
    ax.set_title(f"Merton Jump Diffusion {option_type.capitalize()} Surface")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Option Value")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_pinn_vs_analytical(
    model: JumpDiffusionPINN,
    params: MertonParams,
    option_type: str = "call",
    save_path: Optional[str] = None,
):
    """
    Compare PINN prices against analytical Merton prices.

    Args:
        model: Trained PINN model
        params: Merton parameters
        option_type: "call" or "put"
        save_path: Optional file path to save the figure
    """
    pricer = merton_call_price if option_type == "call" else None

    S_values = np.linspace(params.K * 0.5, params.K * 1.5, 100)
    t = 0.0
    tau = params.T - t

    # Analytical prices
    analytical = np.array([
        merton_call_price(S, params.K, tau, params.r, params.sigma,
                          params.lambda_j, params.mu_j, params.sigma_j)
        for S in S_values
    ])

    # PINN prices
    model.eval()
    pinn_prices = np.array([model.price(S, t) for S in S_values])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Price comparison
    axes[0].plot(S_values, analytical, "b-", label="Merton Analytical", linewidth=2)
    axes[0].plot(S_values, pinn_prices, "r--", label="PINN", linewidth=2)
    axes[0].axvline(x=params.K, color="gray", linestyle=":", alpha=0.5, label=f"Strike K={params.K}")
    axes[0].set_xlabel("Spot Price (S)")
    axes[0].set_ylabel("Option Value")
    axes[0].set_title(f"PINN vs Analytical Merton {option_type.capitalize()} Prices")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error plot
    error = pinn_prices - analytical
    relative_error = np.where(analytical > 0.01, error / analytical * 100, 0.0)
    axes[1].plot(S_values, relative_error, "g-", linewidth=1.5)
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[1].set_xlabel("Spot Price (S)")
    axes[1].set_ylabel("Relative Error (%)")
    axes[1].set_title("PINN Pricing Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_greeks(
    model: JumpDiffusionPINN,
    params: MertonParams,
    save_path: Optional[str] = None,
):
    """
    Plot Greeks (Delta, Gamma, Theta) across spot prices.

    Args:
        model: Trained PINN model
        params: Merton parameters
        save_path: Optional file path to save the figure
    """
    S_values = np.linspace(params.K * 0.5, params.K * 1.5, 100)
    greeks = compute_greeks_grid(model, S_values, t=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Value
    axes[0, 0].plot(S_values, greeks["value"], "b-")
    axes[0, 0].set_title("Option Value V(S)")
    axes[0, 0].set_xlabel("Spot Price (S)")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(True, alpha=0.3)

    # Delta
    axes[0, 1].plot(S_values, greeks["delta"], "r-")
    axes[0, 1].set_title("Delta = dV/dS")
    axes[0, 1].set_xlabel("Spot Price (S)")
    axes[0, 1].set_ylabel("Delta")
    axes[0, 1].grid(True, alpha=0.3)

    # Gamma
    axes[1, 0].plot(S_values, greeks["gamma"], "g-")
    axes[1, 0].set_title("Gamma = d2V/dS2")
    axes[1, 0].set_xlabel("Spot Price (S)")
    axes[1, 0].set_ylabel("Gamma")
    axes[1, 0].grid(True, alpha=0.3)

    # Theta
    axes[1, 1].plot(S_values, greeks["theta"], "m-")
    axes[1, 1].set_title("Theta = dV/dt")
    axes[1, 1].set_xlabel("Spot Price (S)")
    axes[1, 1].set_ylabel("Theta")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.axvline(x=params.K, color="gray", linestyle=":", alpha=0.3)

    plt.suptitle("Greeks under Merton Jump Diffusion (PINN)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_jump_impact(
    params: MertonParams,
    save_path: Optional[str] = None,
):
    """
    Visualize the impact of jump parameters on option prices.

    Shows how lambda_j, mu_j, and sigma_j affect the volatility smile.

    Args:
        params: Base Merton parameters
        save_path: Optional file path to save the figure
    """
    S = params.K
    T = 0.5
    K_range = np.linspace(params.K * 0.75, params.K * 1.25, 60)
    moneyness = K_range / S

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Vary lambda_j
    for lam in [0.0, 2.0, 5.0, 10.0, 20.0]:
        iv = merton_implied_vol_smile(
            S, K_range, T, params.r, params.sigma,
            lam, params.mu_j, params.sigma_j,
        )
        valid = ~np.isnan(iv)
        label = f"lambda={lam:.0f}" if lam > 0 else "BS (no jumps)"
        axes[0].plot(moneyness[valid], iv[valid] * 100, label=label)
    axes[0].set_title("Effect of Jump Intensity (lambda)")
    axes[0].set_xlabel("Moneyness (K/S)")
    axes[0].set_ylabel("Implied Volatility (%)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Vary mu_j
    for mu in [-0.15, -0.10, -0.05, 0.0, 0.05]:
        iv = merton_implied_vol_smile(
            S, K_range, T, params.r, params.sigma,
            params.lambda_j, mu, params.sigma_j,
        )
        valid = ~np.isnan(iv)
        axes[1].plot(moneyness[valid], iv[valid] * 100, label=f"mu_J={mu:.2f}")
    axes[1].set_title("Effect of Mean Jump Size (mu_J)")
    axes[1].set_xlabel("Moneyness (K/S)")
    axes[1].set_ylabel("Implied Volatility (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Vary sigma_j
    for sig in [0.05, 0.10, 0.15, 0.25, 0.40]:
        iv = merton_implied_vol_smile(
            S, K_range, T, params.r, params.sigma,
            params.lambda_j, params.mu_j, sig,
        )
        valid = ~np.isnan(iv)
        axes[2].plot(moneyness[valid], iv[valid] * 100, label=f"sigma_J={sig:.2f}")
    axes[2].set_title("Effect of Jump Volatility (sigma_J)")
    axes[2].set_xlabel("Moneyness (K/S)")
    axes[2].set_ylabel("Implied Volatility (%)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Impact of Jump Parameters on Volatility Smile", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_loss(
    history: dict,
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.

    Args:
        history: Training history dictionary from train.py
        save_path: Optional file path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total loss
    axes[0].semilogy(history["total"], alpha=0.7, label="Total")
    axes[0].semilogy(history["pide"], alpha=0.7, label="PIDE")
    axes[0].semilogy(history["ic"], alpha=0.7, label="IC")
    axes[0].semilogy(history["bc"], alpha=0.7, label="BC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training Loss Components")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation error
    if history.get("val_mean_rel_error"):
        val_epochs = np.arange(
            len(history["val_mean_rel_error"])
        ) * 2000  # Assuming validate_every=2000
        axes[1].plot(val_epochs, np.array(history["val_mean_rel_error"]) * 100, "ro-")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean Relative Error (%)")
        axes[1].set_title("Validation Error vs Analytical Merton")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No validation data", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_return_distribution(
    returns: np.ndarray,
    params: MertonParams,
    save_path: Optional[str] = None,
):
    """
    Plot empirical return distribution vs Merton model fit.

    Args:
        returns: Array of log-returns
        params: Estimated Merton parameters
        save_path: Optional file path to save the figure
    """
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical histogram
    ax.hist(returns, bins=80, density=True, alpha=0.5, color="steelblue", label="Empirical")

    # Normal fit (BS equivalent)
    x = np.linspace(returns.min(), returns.max(), 300)
    mu_n = np.mean(returns)
    sigma_n = np.std(returns)
    ax.plot(x, norm.pdf(x, mu_n, sigma_n), "r--", linewidth=2, label="Normal (BS)")

    # Merton mixture density (approximate with n_terms)
    from scipy.stats import poisson
    daily_lambda = params.lambda_j / 252.0
    density = np.zeros_like(x)
    for n in range(20):
        p_n = poisson.pmf(n, daily_lambda)
        if p_n < 1e-15:
            break
        mu_mix = mu_n + n * params.mu_j  # Simplified daily
        sigma_mix = np.sqrt(sigma_n ** 2 + n * params.sigma_j ** 2)
        density += p_n * norm.pdf(x, mu_mix, sigma_mix)

    ax.plot(x, density, "g-", linewidth=2, label="Merton JD")

    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution: Empirical vs Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inset: log-scale to show tails
    ax_inset = ax.inset_axes([0.6, 0.5, 0.35, 0.4])
    ax_inset.hist(returns, bins=80, density=True, alpha=0.5, color="steelblue")
    ax_inset.plot(x, norm.pdf(x, mu_n, sigma_n), "r--", linewidth=1)
    ax_inset.plot(x, density, "g-", linewidth=1)
    ax_inset.set_yscale("log")
    ax_inset.set_title("Log Scale (Tails)", fontsize=9)
    ax_inset.set_ylim(bottom=1e-4)
    ax_inset.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Jump Diffusion PINN results")
    parser.add_argument(
        "--plot",
        type=str,
        default="smile",
        choices=["smile", "surface", "jump_impact", "all"],
        help="What to plot",
    )
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Default parameters for visualization
    params = MertonParams(sigma=0.20, r=0.05, lambda_j=5.0, mu_j=-0.08, sigma_j=0.15)

    if args.plot in ("smile", "all"):
        plot_volatility_smile(
            params, save_path=os.path.join(args.save_dir, "vol_smile.png")
        )

    if args.plot in ("surface", "all"):
        plot_option_surface(
            params, save_path=os.path.join(args.save_dir, "option_surface.png")
        )

    if args.plot in ("jump_impact", "all"):
        plot_jump_impact(
            params, save_path=os.path.join(args.save_dir, "jump_impact.png")
        )

    # BTC parameters for crypto visualization
    if args.plot == "all":
        btc_params = MertonParams(
            sigma=0.50, r=0.05, lambda_j=12.0, mu_j=-0.10, sigma_j=0.20,
            K=50000.0, S_min=10000.0, S_max=150000.0,
        )
        plot_volatility_smile(
            btc_params,
            T_values=[0.05, 0.1, 0.25, 0.5],
            save_path=os.path.join(args.save_dir, "btc_vol_smile.png"),
        )


if __name__ == "__main__":
    main()
