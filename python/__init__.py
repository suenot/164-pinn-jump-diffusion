"""
Chapter 143: PINN for Jump Diffusion Option Pricing

Physics-Informed Neural Networks for solving Merton's Jump Diffusion
Partial Integro-Differential Equation (PIDE) for option pricing.

Modules:
    jump_diffusion_pinn: Core PINN model with integral loss handling
    train: Training pipeline with adaptive loss weighting
    data_loader: Market data from Bybit (crypto) and Yahoo Finance (stocks)
    merton_analytical: Analytical Merton solution for validation
    greeks: Greeks computation via automatic differentiation
    visualize: Volatility smile, option surfaces, and comparison plots
    backtest: Trading strategy backtesting framework
"""

from .jump_diffusion_pinn import JumpDiffusionPINN, MertonParams
from .merton_analytical import merton_call_price, merton_put_price
from .greeks import compute_greeks

__version__ = "0.1.0"
__all__ = [
    "JumpDiffusionPINN",
    "MertonParams",
    "merton_call_price",
    "merton_put_price",
    "compute_greeks",
]
