"""
Training pipeline for the Jump Diffusion PINN.

Handles:
    - Collocation point sampling (resampled periodically)
    - PIDE residual + terminal + boundary condition losses
    - Adaptive loss weighting (learnable)
    - Learning rate scheduling
    - Validation against analytical Merton solution
    - Checkpointing and logging

Usage:
    python train.py --asset BTC --exchange bybit
    python train.py --sigma 0.2 --lambda_j 5 --mu_j -0.05 --sigma_j 0.15
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from jump_diffusion_pinn import JumpDiffusionPINN, MertonParams
from merton_analytical import merton_call_price, merton_put_price


def create_default_params(asset: str = "SPY", exchange: str = "yahoo") -> MertonParams:
    """
    Create default Merton parameters based on asset type.

    Args:
        asset: Asset name (e.g., "SPY", "BTC", "ETH")
        exchange: Exchange name (e.g., "yahoo", "bybit")

    Returns:
        MertonParams with appropriate defaults
    """
    if asset.upper() in ("BTC", "BTCUSDT", "ETH", "ETHUSDT"):
        # Crypto: higher vol, more frequent jumps
        return MertonParams(
            sigma=0.50,
            r=0.05,
            lambda_j=12.0,
            mu_j=-0.08,
            sigma_j=0.15,
            K=50000.0 if "BTC" in asset.upper() else 3000.0,
            T=1.0,
            S_min=10000.0 if "BTC" in asset.upper() else 500.0,
            S_max=150000.0 if "BTC" in asset.upper() else 10000.0,
        )
    else:
        # Equity: typical parameters
        return MertonParams(
            sigma=0.20,
            r=0.05,
            lambda_j=5.0,
            mu_j=-0.05,
            sigma_j=0.15,
            K=100.0,
            T=1.0,
            S_min=20.0,
            S_max=300.0,
        )


def validate_against_analytical(
    model: JumpDiffusionPINN,
    params: MertonParams,
    n_points: int = 200,
    option_type: str = "call",
) -> dict:
    """
    Validate the PINN against analytical Merton prices.

    Args:
        model: Trained PINN model
        params: Merton parameters
        n_points: Number of validation points
        option_type: "call" or "put"

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    pricer = merton_call_price if option_type == "call" else merton_put_price

    # Generate validation points
    S_values = np.linspace(params.S_min * 1.1, params.S_max * 0.9, n_points)
    t_values = np.array([0.0, 0.25 * params.T, 0.5 * params.T, 0.75 * params.T])

    errors = []
    relative_errors = []

    for t in t_values:
        tau = params.T - t
        for S in S_values:
            # Analytical price
            exact = pricer(
                S, params.K, tau, params.r, params.sigma,
                params.lambda_j, params.mu_j, params.sigma_j,
            )

            # PINN price
            pinn_price = model.price(S, t)

            err = abs(pinn_price - exact)
            errors.append(err)
            if exact > 0.01:
                relative_errors.append(err / exact)

    errors = np.array(errors)
    relative_errors = np.array(relative_errors)

    return {
        "mean_abs_error": float(np.mean(errors)),
        "max_abs_error": float(np.max(errors)),
        "mean_rel_error": float(np.mean(relative_errors)),
        "max_rel_error": float(np.max(relative_errors)),
        "median_rel_error": float(np.median(relative_errors)),
    }


def train_pinn(
    params: Optional[MertonParams] = None,
    option_type: str = "call",
    n_epochs: int = 20000,
    lr: float = 1e-3,
    n_pide: int = 2000,
    n_ic: int = 500,
    n_bc: int = 500,
    resample_every: int = 1000,
    validate_every: int = 2000,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    verbose: bool = True,
) -> JumpDiffusionPINN:
    """
    Train the Jump Diffusion PINN.

    Args:
        params: Merton model parameters
        option_type: "call" or "put"
        n_epochs: Number of training epochs
        lr: Initial learning rate
        n_pide: Number of PIDE collocation points
        n_ic: Number of terminal condition points
        n_bc: Number of boundary condition points
        resample_every: Resample collocation points every N epochs
        validate_every: Run validation every N epochs
        device: Torch device ("cpu" or "cuda")
        save_dir: Directory for checkpoints
        verbose: Print progress

    Returns:
        Trained JumpDiffusionPINN model
    """
    if params is None:
        params = MertonParams()

    # Create model
    model = JumpDiffusionPINN(
        params=params,
        hidden_dim=128,
        n_residual_blocks=4,
        n_fourier=64,
        sigma_ff=1.0,
        n_quadrature=32,
        option_type=option_type,
    ).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Merton params: {params.to_dict()}")
        print(f"Device: {device}")
        print(f"Training for {n_epochs} epochs")

    # Optimizer: separate lr for network and adaptive weights
    net_params = [p for name, p in model.named_parameters() if "log_w" not in name]
    weight_params = [p for name, p in model.named_parameters() if "log_w" in name]

    optimizer = optim.Adam(
        [
            {"params": net_params, "lr": lr},
            {"params": weight_params, "lr": 0.1 * lr},
        ]
    )

    # Learning rate scheduler: cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-6
    )

    # Training log
    history = {
        "total": [],
        "pide": [],
        "ic": [],
        "bc": [],
        "val_mean_rel_error": [],
    }

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initial collocation points
    points = model.sample_collocation_points(n_pide, n_ic, n_bc, device)

    best_val_error = float("inf")
    start_time = time.time()

    progress = tqdm(range(n_epochs), desc="Training", disable=not verbose)
    for epoch in progress:
        model.train()

        # Resample collocation points periodically
        if epoch > 0 and epoch % resample_every == 0:
            points = model.sample_collocation_points(n_pide, n_ic, n_bc, device)

        # Forward pass and loss computation
        optimizer.zero_grad()
        total_loss, loss_dict = model.compute_total_loss(
            S_pide=points["S_pide"],
            t_pide=points["t_pide"],
            S_ic=points["S_ic"],
            t_ic=points["t_ic"],
            S_bc=points["S_bc"],
            t_bc=points["t_bc"],
            V_bc=points["V_bc"],
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Log
        history["total"].append(loss_dict["total"])
        history["pide"].append(loss_dict["pide"])
        history["ic"].append(loss_dict["ic"])
        history["bc"].append(loss_dict["bc"])

        # Update progress bar
        if epoch % 100 == 0:
            progress.set_postfix({
                "loss": f"{loss_dict['total']:.2e}",
                "pide": f"{loss_dict['pide']:.2e}",
                "ic": f"{loss_dict['ic']:.2e}",
                "bc": f"{loss_dict['bc']:.2e}",
            })

        # Validation
        if epoch > 0 and epoch % validate_every == 0:
            val_metrics = validate_against_analytical(model, params, option_type=option_type)
            history["val_mean_rel_error"].append(val_metrics["mean_rel_error"])

            if verbose:
                elapsed = time.time() - start_time
                print(f"\n  Epoch {epoch}: val_mean_rel_error={val_metrics['mean_rel_error']:.4f}, "
                      f"val_max_rel_error={val_metrics['max_rel_error']:.4f}, "
                      f"elapsed={elapsed:.1f}s")

            # Save best model
            if val_metrics["mean_rel_error"] < best_val_error:
                best_val_error = val_metrics["mean_rel_error"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "params": params.to_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    os.path.join(save_dir, "best_model.pt"),
                )

    # Final validation
    final_metrics = validate_against_analytical(model, params, option_type=option_type)
    elapsed = time.time() - start_time

    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Final metrics: {json.dumps(final_metrics, indent=2)}")

    # Save final model and training history
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": params.to_dict(),
            "history": history,
            "final_metrics": final_metrics,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Jump Diffusion PINN")
    parser.add_argument("--asset", type=str, default="SPY", help="Asset name")
    parser.add_argument("--exchange", type=str, default="yahoo", help="Exchange")
    parser.add_argument("--option_type", type=str, default="call", choices=["call", "put"])
    parser.add_argument("--n_epochs", type=int, default=20000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=None, help="Diffusion vol")
    parser.add_argument("--lambda_j", type=float, default=None, help="Jump intensity")
    parser.add_argument("--mu_j", type=float, default=None, help="Mean log-jump")
    parser.add_argument("--sigma_j", type=float, default=None, help="Jump vol")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Create parameters
    params = create_default_params(args.asset, args.exchange)

    # Override with command-line arguments if provided
    if args.sigma is not None:
        params.sigma = args.sigma
    if args.lambda_j is not None:
        params.lambda_j = args.lambda_j
    if args.mu_j is not None:
        params.mu_j = args.mu_j
    if args.sigma_j is not None:
        params.sigma_j = args.sigma_j

    # Detect GPU
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        print("CUDA detected, using GPU")

    # Train
    model = train_pinn(
        params=params,
        option_type=args.option_type,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
    )

    # Demo pricing
    print("\n--- Demo Pricing ---")
    K = params.K
    test_spots = [K * 0.8, K * 0.9, K, K * 1.1, K * 1.2]
    pricer = merton_call_price if args.option_type == "call" else merton_put_price

    print(f"{'Spot':>12} {'PINN':>12} {'Merton':>12} {'Error':>12} {'Rel Err':>12}")
    print("-" * 62)
    for S in test_spots:
        pinn_price = model.price(S)
        exact = pricer(
            S, K, params.T, params.r, params.sigma,
            params.lambda_j, params.mu_j, params.sigma_j,
        )
        err = abs(pinn_price - exact)
        rel = err / exact if exact > 0.01 else 0.0
        print(f"{S:12.2f} {pinn_price:12.4f} {exact:12.4f} {err:12.4f} {rel:12.4%}")


if __name__ == "__main__":
    main()
