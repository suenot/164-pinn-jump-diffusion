"""
Core PINN model for Merton's Jump Diffusion PIDE.

The network approximates V(S, t) -- the European option value under jump diffusion.
It is trained by minimizing the PIDE residual, including the integral term computed
via Gauss-Hermite quadrature.

The PIDE:
    dV/dt + 0.5*sigma^2*S^2 * d2V/dS2 + (r - lambda*k)*S * dV/dS
    - (r + lambda)*V + lambda * integral[V(S*e^y, t) * f(y) dy] = 0

where f(y) = N(mu_J, sigma_J^2) is the log-normal jump size distribution.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MertonParams:
    """Parameters for Merton's jump diffusion model."""

    sigma: float = 0.20       # Diffusion volatility
    r: float = 0.05           # Risk-free rate
    lambda_j: float = 5.0     # Jump intensity (jumps/year)
    mu_j: float = -0.05       # Mean log-jump size
    sigma_j: float = 0.15     # Jump size volatility
    K: float = 100.0          # Strike price
    T: float = 1.0            # Time to maturity (years)
    S_min: float = 20.0       # Lower spot boundary
    S_max: float = 300.0      # Upper spot boundary

    @property
    def k(self) -> float:
        """Expected percentage jump size: E[J] = exp(mu_J + sigma_J^2/2) - 1."""
        return math.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0

    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return {
            "sigma": self.sigma,
            "r": self.r,
            "lambda_j": self.lambda_j,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
            "K": self.K,
            "T": self.T,
            "S_min": self.S_min,
            "S_max": self.S_max,
            "k": self.k,
        }


class FourierFeatureLayer(nn.Module):
    """
    Random Fourier feature encoding to help the network learn
    high-frequency patterns in the option surface.

    Maps input x to [sin(B*x), cos(B*x)] where B is a frozen random matrix.
    """

    def __init__(self, in_features: int, n_frequencies: int = 64, sigma_ff: float = 1.0):
        super().__init__()
        self.n_frequencies = n_frequencies
        # Random frequency matrix (frozen, not trained)
        B = torch.randn(in_features, n_frequencies) * sigma_ff
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding."""
        projected = x @ self.B  # (batch, n_frequencies)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    @property
    def out_features(self) -> int:
        return 2 * self.n_frequencies


class ResidualBlock(nn.Module):
    """Residual block with tanh activation."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.linear(x))


class JumpDiffusionPINN(nn.Module):
    """
    Physics-Informed Neural Network for Merton's Jump Diffusion PIDE.

    Architecture:
        Input: (log-moneyness, normalized_tau) -- 2D
        -> Fourier Feature Encoding
        -> Dense layers with residual connections
        -> Softplus output (ensures V >= 0)

    The network is trained to satisfy:
        1. PIDE residual at interior collocation points
        2. Terminal condition at t = T
        3. Boundary conditions at S_min and S_max

    The integral term in the PIDE is computed via Gauss-Hermite quadrature.
    """

    def __init__(
        self,
        params: MertonParams,
        hidden_dim: int = 128,
        n_residual_blocks: int = 4,
        n_fourier: int = 64,
        sigma_ff: float = 1.0,
        n_quadrature: int = 32,
        option_type: str = "call",
    ):
        super().__init__()
        self.params = params
        self.option_type = option_type
        self.hidden_dim = hidden_dim
        self.n_quadrature = n_quadrature

        # Gauss-Hermite quadrature nodes and weights
        nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature)
        self.register_buffer("gh_nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer("gh_weights", torch.tensor(weights, dtype=torch.float32))

        # Fourier feature layer
        self.fourier = FourierFeatureLayer(2, n_fourier, sigma_ff)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.fourier.out_features, hidden_dim),
            nn.Tanh(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus(beta=1.0)

        # Adaptive loss weights (learnable log-weights)
        self.log_w_pide = nn.Parameter(torch.tensor(0.0))
        self.log_w_ic = nn.Parameter(torch.tensor(0.0))
        self.log_w_bc = nn.Parameter(torch.tensor(0.0))
        self.log_w_data = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _preprocess(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Preprocess inputs to log-moneyness and normalized time.

        Args:
            S: Spot prices, shape (batch,)
            t: Current times, shape (batch,)

        Returns:
            Preprocessed input tensor, shape (batch, 2)
        """
        K = self.params.K
        T = self.params.T
        x1 = torch.log(S / K)          # log-moneyness
        x2 = (T - t) / T               # normalized time-to-maturity in [0, 1]
        return torch.stack([x1, x2], dim=-1)

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute option value V(S, t).

        Args:
            S: Spot prices, shape (batch,)
            t: Current times, shape (batch,)

        Returns:
            Option values, shape (batch,)
        """
        x = self._preprocess(S, t)
        x = self.fourier(x)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        x = self.softplus(x)  # Ensure V >= 0
        return x.squeeze(-1)

    def compute_pide_residual(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the PIDE residual at collocation points (S, t).

        The PIDE is:
            dV/dt + 0.5*sigma^2*S^2 * d2V/dS2 + (r - lambda*k)*S * dV/dS
            - (r + lambda)*V + lambda * Integral = 0

        All derivatives are computed via autograd.
        The integral is computed via Gauss-Hermite quadrature.

        Args:
            S: Spot prices, shape (N,), requires_grad=True
            t: Times, shape (N,), requires_grad=True

        Returns:
            PIDE residual, shape (N,)
        """
        p = self.params

        # Ensure inputs require grad for autograd
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)

        V = self.forward(S, t)

        # First derivatives via autograd
        # dV/dS
        dV_dS = torch.autograd.grad(
            V, S,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True,
        )[0]

        # dV/dt
        dV_dt = torch.autograd.grad(
            V, t,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True,
        )[0]

        # d2V/dS2
        d2V_dS2 = torch.autograd.grad(
            dV_dS, S,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute the integral term via Gauss-Hermite quadrature
        # I(S, t) = lambda / sqrt(pi) * sum_i w_i * V(S * exp(mu_J + sigma_J*sqrt(2)*z_i), t)
        integral = self._compute_jump_integral(S, t)

        # PIDE residual
        diffusion = 0.5 * p.sigma ** 2 * S ** 2 * d2V_dS2
        drift = (p.r - p.lambda_j * p.k) * S * dV_dS
        discount = -(p.r + p.lambda_j) * V
        jump = p.lambda_j * integral

        residual = dV_dt + diffusion + drift + discount + jump

        return residual

    def _compute_jump_integral(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the jump integral via Gauss-Hermite quadrature.

        I(S, t) = integral V(S*e^y, t) * f(y) dy
                 = (1/sqrt(pi)) * sum_i w_i * V(S * exp(mu_J + sigma_J*sqrt(2)*z_i), t)

        This is the key innovation: we evaluate the network at shifted spot prices
        corresponding to each quadrature node, then take a weighted sum.

        Args:
            S: Spot prices, shape (N,)
            t: Times, shape (N,)

        Returns:
            Integral values, shape (N,)
        """
        p = self.params
        N = S.shape[0]
        n_q = self.n_quadrature

        # Compute shifted log-jump sizes for each quadrature node
        # y_i = mu_J + sigma_J * sqrt(2) * z_i
        y_nodes = p.mu_j + p.sigma_j * math.sqrt(2) * self.gh_nodes  # (n_q,)

        # Compute jumped spot prices: S_jumped[j, i] = S[j] * exp(y_i)
        # S shape: (N,), y_nodes shape: (n_q,)
        S_jumped = S.unsqueeze(1) * torch.exp(y_nodes.unsqueeze(0))  # (N, n_q)

        # Flatten for batch evaluation
        S_flat = S_jumped.reshape(-1)  # (N * n_q,)
        t_flat = t.unsqueeze(1).expand(-1, n_q).reshape(-1)  # (N * n_q,)

        # Clamp spot prices to domain
        S_flat = torch.clamp(S_flat, min=p.S_min, max=p.S_max)

        # Evaluate network at all jumped prices
        with torch.no_grad() if not self.training else torch.enable_grad():
            V_jumped = self.forward(S_flat, t_flat)  # (N * n_q,)

        V_jumped = V_jumped.reshape(N, n_q)  # (N, n_q)

        # Weighted sum: (1/sqrt(pi)) * sum_i w_i * V_jumped[:, i]
        integral = (1.0 / math.sqrt(math.pi)) * torch.sum(
            self.gh_weights.unsqueeze(0) * V_jumped, dim=1
        )  # (N,)

        return integral

    def compute_total_loss(
        self,
        S_pide: torch.Tensor,
        t_pide: torch.Tensor,
        S_ic: torch.Tensor,
        t_ic: torch.Tensor,
        S_bc: torch.Tensor,
        t_bc: torch.Tensor,
        V_bc: torch.Tensor,
        S_data: Optional[torch.Tensor] = None,
        t_data: Optional[torch.Tensor] = None,
        V_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the total training loss with adaptive weighting.

        L_total = w_pide * L_pide + w_ic * L_ic + w_bc * L_bc + w_data * L_data

        Args:
            S_pide, t_pide: Interior collocation points for PIDE residual
            S_ic, t_ic: Terminal condition points (t_ic should be T)
            S_bc, t_bc, V_bc: Boundary condition points and exact values
            S_data, t_data, V_data: Optional market data points

        Returns:
            Tuple of (total_loss, loss_dict with individual components)
        """
        # PIDE residual loss
        residual = self.compute_pide_residual(S_pide, t_pide)
        L_pide = torch.mean(residual ** 2)

        # Terminal condition loss
        V_ic_pred = self.forward(S_ic, t_ic)
        if self.option_type == "call":
            payoff = torch.relu(S_ic - self.params.K)
        else:
            payoff = torch.relu(self.params.K - S_ic)
        L_ic = torch.mean((V_ic_pred - payoff) ** 2)

        # Boundary condition loss
        V_bc_pred = self.forward(S_bc, t_bc)
        L_bc = torch.mean((V_bc_pred - V_bc) ** 2)

        # Adaptive weights
        w_pide = torch.exp(self.log_w_pide)
        w_ic = torch.exp(self.log_w_ic)
        w_bc = torch.exp(self.log_w_bc)

        total_loss = w_pide * L_pide + w_ic * L_ic + w_bc * L_bc

        # Optional market data loss
        L_data = torch.tensor(0.0, device=S_pide.device)
        if S_data is not None and t_data is not None and V_data is not None:
            V_data_pred = self.forward(S_data, t_data)
            L_data = torch.mean((V_data_pred - V_data) ** 2)
            w_data = torch.exp(self.log_w_data)
            total_loss = total_loss + w_data * L_data

        # Regularization on weights to prevent them from becoming too large
        weight_reg = 0.01 * (
            self.log_w_pide ** 2
            + self.log_w_ic ** 2
            + self.log_w_bc ** 2
            + self.log_w_data ** 2
        )
        total_loss = total_loss + weight_reg

        loss_dict = {
            "total": total_loss.item(),
            "pide": L_pide.item(),
            "ic": L_ic.item(),
            "bc": L_bc.item(),
            "data": L_data.item(),
            "w_pide": w_pide.item(),
            "w_ic": w_ic.item(),
            "w_bc": w_bc.item(),
        }

        return total_loss, loss_dict

    def sample_collocation_points(
        self,
        n_pide: int = 2000,
        n_ic: int = 500,
        n_bc: int = 500,
        device: str = "cpu",
    ) -> dict:
        """
        Sample collocation points for training.

        Interior points are sampled using Latin Hypercube Sampling (LHS)
        for better space coverage.

        Args:
            n_pide: Number of interior PIDE collocation points
            n_ic: Number of terminal condition points
            n_bc: Number of boundary condition points
            device: Torch device

        Returns:
            Dictionary with all collocation point tensors
        """
        p = self.params

        # Interior points (S, t) for PIDE residual
        # Use log-uniform distribution for S to better cover the range
        log_S_min = math.log(p.S_min)
        log_S_max = math.log(p.S_max)
        log_S_pide = torch.rand(n_pide, device=device) * (log_S_max - log_S_min) + log_S_min
        S_pide = torch.exp(log_S_pide)
        t_pide = torch.rand(n_pide, device=device) * p.T * 0.999  # Avoid t=T exactly

        # Terminal condition points at t = T
        log_S_ic = torch.rand(n_ic, device=device) * (log_S_max - log_S_min) + log_S_min
        S_ic = torch.exp(log_S_ic)
        t_ic = torch.full((n_ic,), p.T, device=device)

        # Boundary condition points
        n_bc_half = n_bc // 2
        t_bc = torch.rand(n_bc, device=device) * p.T

        # Lower boundary: S = S_min
        S_bc_low = torch.full((n_bc_half,), p.S_min, device=device)
        # Upper boundary: S = S_max
        S_bc_high = torch.full((n_bc - n_bc_half,), p.S_max, device=device)
        S_bc = torch.cat([S_bc_low, S_bc_high])

        # Exact boundary values
        if self.option_type == "call":
            V_bc_low = torch.zeros(n_bc_half, device=device)
            tau_high = p.T - t_bc[n_bc_half:]
            V_bc_high = p.S_max - p.K * torch.exp(-p.r * tau_high)
        else:
            tau_low = p.T - t_bc[:n_bc_half]
            V_bc_low = p.K * torch.exp(-p.r * tau_low)
            V_bc_high = torch.zeros(n_bc - n_bc_half, device=device)

        V_bc = torch.cat([V_bc_low, V_bc_high])

        return {
            "S_pide": S_pide,
            "t_pide": t_pide,
            "S_ic": S_ic,
            "t_ic": t_ic,
            "S_bc": S_bc,
            "t_bc": t_bc,
            "V_bc": V_bc,
        }

    def price(self, S: float, t: float = 0.0) -> float:
        """
        Price a single option.

        Args:
            S: Spot price
            t: Current time (default 0)

        Returns:
            Option value
        """
        self.eval()
        with torch.no_grad():
            S_t = torch.tensor([S], dtype=torch.float32)
            t_t = torch.tensor([t], dtype=torch.float32)
            V = self.forward(S_t, t_t)
        return V.item()

    def price_grid(
        self,
        S_values: np.ndarray,
        t_values: np.ndarray,
    ) -> np.ndarray:
        """
        Price options on a grid of (S, t) values.

        Args:
            S_values: Array of spot prices, shape (n_S,)
            t_values: Array of times, shape (n_t,)

        Returns:
            Option values on the grid, shape (n_S, n_t)
        """
        self.eval()
        n_S = len(S_values)
        n_t = len(t_values)

        # Create meshgrid
        SS, TT = np.meshgrid(S_values, t_values, indexing="ij")
        S_flat = torch.tensor(SS.flatten(), dtype=torch.float32)
        t_flat = torch.tensor(TT.flatten(), dtype=torch.float32)

        with torch.no_grad():
            V_flat = self.forward(S_flat, t_flat)

        return V_flat.numpy().reshape(n_S, n_t)
