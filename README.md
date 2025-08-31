# Chapter 143: PINN for Jump Diffusion Option Pricing

## Overview

Standard option pricing models like Black-Scholes assume that asset prices follow geometric Brownian motion -- a continuous diffusion process. This assumption produces thin-tailed return distributions and cannot explain the **volatility smile** observed in real markets. Asset prices, particularly in cryptocurrency markets, exhibit sudden large moves (crashes, rallies, liquidation cascades) that continuous models fail to capture.

**Merton's Jump Diffusion Model** (1976) extends Black-Scholes by adding a compound Poisson jump process to the diffusion, producing fat tails and a volatility smile. However, pricing options under jump diffusion requires solving a **Partial Integro-Differential Equation (PIDE)** -- a PDE with an additional integral term representing the expected change in option value due to jumps. This integral term makes the equation significantly harder to solve numerically.

**Physics-Informed Neural Networks (PINNs)** offer an elegant mesh-free approach: we train a neural network to approximate the option value function V(S,t) while enforcing the PIDE as a soft constraint in the loss function. The integral term is handled via numerical quadrature (Gauss-Hermite), making the entire pipeline differentiable and GPU-friendly.

This chapter implements PINN-based jump diffusion pricing for both equity and crypto options, with a focus on **Bybit perpetual and options markets** where jumps are especially prevalent.

---

## Table of Contents

1. [Why Black-Scholes Fails](#1-why-black-scholes-fails)
2. [Merton's Jump Diffusion Model](#2-mertons-jump-diffusion-model)
3. [The PIDE for Jump Diffusion](#3-the-pide-for-jump-diffusion)
4. [Physics-Informed Neural Networks](#4-physics-informed-neural-networks)
5. [Handling the Integral Term](#5-handling-the-integral-term)
6. [Network Architecture](#6-network-architecture)
7. [Loss Function Design](#7-loss-function-design)
8. [Gauss-Hermite Quadrature](#8-gauss-hermite-quadrature)
9. [Analytical Merton Solution](#9-analytical-merton-solution)
10. [Greeks Under Jump Diffusion](#10-greeks-under-jump-diffusion)
11. [Volatility Smile Reproduction](#11-volatility-smile-reproduction)
12. [Comparison with FFT Methods](#12-comparison-with-fft-methods)
13. [Application to Crypto Markets](#13-application-to-crypto-markets)
14. [Trading Strategy and Backtesting](#14-trading-strategy-and-backtesting)
15. [Implementation Guide](#15-implementation-guide)
16. [References](#16-references)

---

## 1. Why Black-Scholes Fails

The Black-Scholes model assumes:

```
dS/S = mu * dt + sigma * dW
```

where W is a standard Brownian motion. This implies:
- Log-returns are normally distributed
- Volatility is constant across strikes and maturities
- No sudden jumps in price

**Empirical observations contradict all three:**

| Feature | Black-Scholes Prediction | Market Reality |
|---------|------------------------|----------------|
| Return distribution | Gaussian (thin tails) | Leptokurtic (fat tails) |
| Implied volatility | Flat across strikes | Volatility smile/skew |
| Large moves | Extremely rare | Regular occurrence |
| Crypto 10%+ daily moves | ~0% probability | Happens multiple times/year |

In crypto markets (Bybit), the failure is even more pronounced:
- BTC has experienced 20%+ daily drops (March 2020, May 2021, November 2022)
- Liquidation cascades create discontinuous price movements
- Options market-makers quote significant smile, implying jump risk

---

## 2. Merton's Jump Diffusion Model

Merton (1976) proposed augmenting the diffusion with a jump component:

```
dS/S = (mu - lambda*k) * dt + sigma * dW + J * dN
```

where:
- `sigma` -- diffusion volatility
- `dW` -- Brownian motion increment
- `dN` -- Poisson process with intensity `lambda` (average number of jumps per year)
- `J` -- random jump size, where `ln(1+J) ~ N(mu_J, sigma_J^2)`
- `k = E[J] = exp(mu_J + sigma_J^2/2) - 1` -- expected percentage jump size
- The drift is compensated by `-lambda*k` to maintain risk-neutrality

### Parameter Interpretation

| Parameter | Symbol | Typical (Equity) | Typical (BTC) |
|-----------|--------|-------------------|----------------|
| Diffusion volatility | sigma | 0.15-0.25 | 0.40-0.80 |
| Jump intensity | lambda | 1-5/year | 5-20/year |
| Mean jump size (log) | mu_J | -0.05 to -0.10 | -0.10 to -0.20 |
| Jump size volatility | sigma_J | 0.10-0.20 | 0.15-0.40 |

### Return Distribution

Under Merton's model, the log-return over interval dt has a **mixture of normals**:

```
ln(S_T/S_0) = sum over n from 0 to infinity of:
  P(N=n) * Normal(
    (r - lambda*k - sigma^2/2)*T + n*mu_J,
    sigma^2*T + n*sigma_J^2
  )
```

where P(N=n) is the Poisson probability of n jumps:

```
P(N=n) = (lambda*T)^n * exp(-lambda*T) / n!
```

This mixture produces fat tails and skewness, matching market observations.

---

## 3. The PIDE for Jump Diffusion

Under risk-neutral pricing, the option value V(S,t) satisfies the **Partial Integro-Differential Equation**:

```
dV/dt + (1/2)*sigma^2*S^2 * d^2V/dS^2 + (r - lambda*k)*S * dV/dS - (r + lambda)*V
  + lambda * integral from -inf to +inf of V(S*e^y, t) * f(y) dy = 0
```

where:
- `f(y)` is the PDF of the log-jump size: `f(y) = N(mu_J, sigma_J^2)`
- The integral computes the expected option value after a jump of log-size y

### Boundary and Terminal Conditions

**European Call:**
```
V(S, T) = max(S - K, 0)                    (terminal condition)
V(0, t) = 0                                 (lower boundary)
V(S, t) -> S - K*exp(-r*(T-t))  as S -> inf (upper boundary)
```

**European Put:**
```
V(S, T) = max(K - S, 0)                    (terminal condition)
V(0, t) = K*exp(-r*(T-t))                   (lower boundary)
V(S, t) -> 0                    as S -> inf (upper boundary)
```

### Comparison with Black-Scholes PDE

| Term | Black-Scholes | Merton Jump Diffusion |
|------|--------------|----------------------|
| Time derivative | dV/dt | dV/dt |
| Diffusion | (1/2)*sigma^2*S^2 * d^2V/dS^2 | (1/2)*sigma^2*S^2 * d^2V/dS^2 |
| Drift | r*S * dV/dS | (r - lambda*k)*S * dV/dS |
| Discount | -r*V | -(r + lambda)*V |
| Jump integral | **Not present** | lambda * integral V(S*e^y) * f(y) dy |

The jump integral is what makes the equation an **integro-differential** equation and requires special numerical treatment.

---

## 4. Physics-Informed Neural Networks

A PINN approximates V(S,t) with a neural network V_nn(S,t; theta), where theta are the network parameters. The network is trained to satisfy:

1. **The PIDE** at collocation points in the interior domain
2. **Terminal conditions** at t = T
3. **Boundary conditions** at S = 0 and S -> S_max

### Key Advantages for Jump Diffusion

- **Mesh-free**: No discretization grid needed; collocation points are sampled randomly
- **Differentiable**: Automatic differentiation provides exact derivatives dV/dS, d^2V/dS^2, dV/dt
- **GPU-friendly**: Batch evaluation on GPU for both the PDE residual and the integral term
- **Flexible**: Easy to extend to stochastic volatility + jumps (Bates model)

### Architecture Overview

```
Input: (S, t)          -- spot price and time
    |
    v
[Log-transform: x = ln(S/K)]  -- improves numerical conditioning
    |
    v
[Fourier Feature Layer]       -- helps learn high-frequency features
    |
    v
[Fully Connected: 128 -> 128 -> 128 -> 128]  -- tanh activation
    |
    v
Output: V(S, t)        -- option value
```

---

## 5. Handling the Integral Term

The integral term in the PIDE is:

```
I(S, t) = lambda * integral from -inf to +inf of V_nn(S*e^y, t) * f(y) dy
```

where `f(y) = (1 / (sigma_J * sqrt(2*pi))) * exp(-(y - mu_J)^2 / (2*sigma_J^2))`.

### Substitution for Gauss-Hermite Quadrature

We substitute `z = (y - mu_J) / (sigma_J * sqrt(2))` so that:

```
y = mu_J + sigma_J * sqrt(2) * z
dy = sigma_J * sqrt(2) * dz
f(y) dy = (1/sqrt(pi)) * exp(-z^2) * dz
```

The integral becomes:

```
I(S, t) = lambda / sqrt(pi) * sum over i of w_i * V_nn(S * exp(mu_J + sigma_J*sqrt(2)*z_i), t)
```

where `(z_i, w_i)` are the Gauss-Hermite quadrature nodes and weights.

### Why Gauss-Hermite?

The Gauss-Hermite quadrature is specifically designed for integrals of the form:

```
integral from -inf to +inf of f(x) * exp(-x^2) dx ≈ sum_i w_i * f(x_i)
```

Since the jump-size distribution is log-normal, after substitution it has exactly the `exp(-x^2)` kernel. With 20-32 quadrature points, we achieve excellent accuracy for typical jump parameters.

### Practical Implementation

```python
# Gauss-Hermite nodes and weights
nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature)

# For each collocation point (S_j, t_j):
# Compute shifted spot prices for all quadrature nodes
y_values = mu_J + sigma_J * np.sqrt(2) * nodes        # shape: (n_quad,)
S_jumped = S_j * np.exp(y_values)                       # shape: (n_quad,)

# Evaluate network at all jumped prices
t_repeated = t_j * np.ones(n_quadrature)                # shape: (n_quad,)
V_jumped = network(S_jumped, t_repeated)                 # shape: (n_quad,)

# Compute integral
integral = (lambda_J / np.sqrt(np.pi)) * np.sum(weights * V_jumped)
```

---

## 6. Network Architecture

### Input Preprocessing

Raw spot price S varies over orders of magnitude. We use log-moneyness and normalized time:

```
x1 = ln(S / K)          -- log-moneyness (centered around 0 for ATM)
x2 = (T - t) / T_max    -- normalized time to maturity in [0, 1]
```

### Fourier Feature Encoding

To help the network learn high-frequency patterns in the option surface, we use random Fourier features:

```
gamma(x) = [sin(B * x), cos(B * x)]
```

where B is a matrix of random frequencies sampled from N(0, sigma_ff^2). This technique is critical for capturing the sharp kink at the strike price.

### Full Architecture

```
Layer 1:  Fourier Features    (2) -> (2 * n_fourier)
Layer 2:  Linear + Tanh       (2*n_fourier) -> 128
Layer 3:  Linear + Tanh       128 -> 128   (+ residual connection)
Layer 4:  Linear + Tanh       128 -> 128   (+ residual connection)
Layer 5:  Linear + Tanh       128 -> 128   (+ residual connection)
Layer 6:  Linear (output)     128 -> 1
Layer 7:  Softplus             ensures V >= 0
```

Total parameters: approximately 70,000 -- small enough for fast training.

---

## 7. Loss Function Design

The total loss is a weighted sum of four components:

```
L_total = w_pide * L_pide + w_ic * L_ic + w_bc * L_bc + w_data * L_data
```

### L_pide -- PIDE Residual Loss

At N_pide collocation points {(S_j, t_j)} sampled in the interior domain:

```
R_j = dV/dt + (1/2)*sigma^2*S_j^2 * d^2V/dS^2 + (r - lambda*k)*S_j * dV/dS
      - (r + lambda)*V + lambda * Integral_j

L_pide = (1/N_pide) * sum_j R_j^2
```

The derivatives dV/dt, dV/dS, d^2V/dS^2 are computed via automatic differentiation. The integral Integral_j is computed via Gauss-Hermite quadrature as described above.

### L_ic -- Terminal Condition Loss

At N_ic points sampled at t = T:

```
L_ic = (1/N_ic) * sum_j (V_nn(S_j, T) - payoff(S_j))^2
```

### L_bc -- Boundary Condition Loss

At N_bc points sampled at S = S_min and S = S_max:

```
L_bc = (1/N_bc) * sum_j (V_nn(S_j^bc, t_j) - V_exact_boundary(S_j^bc, t_j))^2
```

### L_data -- Market Data Loss (Optional)

If we have observed market option prices {V_market_i}:

```
L_data = (1/N_data) * sum_i (V_nn(S_i, t_i) - V_market_i)^2
```

This allows **calibration** to market data while respecting the PIDE physics.

### Adaptive Weighting

We use the self-adaptive weighting scheme of McClenny & Brainerd (2020):

```
w_k = exp(s_k)
```

where s_k are learnable log-weight parameters optimized alongside the network. This automatically balances the loss components during training.

---

## 8. Gauss-Hermite Quadrature

### Mathematical Foundation

The Gauss-Hermite quadrature approximates:

```
integral from -inf to +inf of f(x) * exp(-x^2) dx ≈ sum_{i=1}^{n} w_i * f(x_i)
```

The nodes x_i are the roots of the Hermite polynomial H_n(x), and the weights are:

```
w_i = (2^(n-1) * n! * sqrt(pi)) / (n^2 * [H_{n-1}(x_i)]^2)
```

### Accuracy Analysis

For our jump integral with log-normal jump distribution:
- **n=8**: Relative error < 1% for most practical parameters
- **n=16**: Relative error < 0.01%
- **n=32**: Machine-precision accuracy
- **n=64**: Overkill but used for validation

We default to **n=32** for a balance of accuracy and computational cost.

### Truncation Effects

The Gauss-Hermite quadrature implicitly handles the infinite integration limits. For extreme jump parameters (sigma_J > 0.5), we verify accuracy by comparison with a truncated Simpson's rule on [-5*sigma_J, 5*sigma_J].

---

## 9. Analytical Merton Solution

Merton showed that the European option price under jump diffusion can be written as an **infinite series of Black-Scholes prices**:

```
V_Merton = sum_{n=0}^{inf} P(N=n) * BS(S, K, T, r_n, sigma_n)
```

where:
```
P(N=n) = exp(-lambda'*T) * (lambda'*T)^n / n!
lambda' = lambda * (1 + k)
r_n = r - lambda*k + n*ln(1+k)/T
sigma_n = sqrt(sigma^2 + n*sigma_J^2/T)
```

This series converges rapidly (typically 20-30 terms suffice) and serves as our **ground truth** for validating the PINN.

### Validation Protocol

We compute the Merton analytical price at a dense grid of (S, t) points and compare:

```
Relative Error = |V_PINN - V_Merton| / V_Merton
```

Target: mean relative error < 0.5% across the domain.

---

## 10. Greeks Under Jump Diffusion

PINNs provide Greeks "for free" via automatic differentiation:

### First-Order Greeks

**Delta** (sensitivity to spot):
```
Delta = dV/dS
```

**Theta** (sensitivity to time):
```
Theta = dV/dt
```

**Rho** (sensitivity to rate):
```
Rho = dV/dr  (requires r as network input)
```

### Second-Order Greeks

**Gamma** (curvature):
```
Gamma = d^2V/dS^2
```

**Vanna** (cross-sensitivity):
```
Vanna = d^2V/(dS * dsigma)
```

### Jump-Specific Greeks

**Jump Sensitivity** (sensitivity to jump intensity):
```
dV/d(lambda) -- how does the option value change with more frequent jumps?
```

**Jump Size Sensitivity**:
```
dV/d(mu_J) -- effect of mean jump size
dV/d(sigma_J) -- effect of jump size uncertainty
```

These jump-specific Greeks are unique to jump diffusion models and critical for hedging in practice. PINNs compute them effortlessly via autograd.

---

## 11. Volatility Smile Reproduction

The key test for any jump diffusion model is reproducing the **volatility smile** -- the pattern of implied volatility across strikes.

### Implied Volatility Computation

Given the PINN price V_PINN(S, K, T), we invert Black-Scholes to find the implied volatility:

```
sigma_imp: BS(S, K, T, r, sigma_imp) = V_PINN(S, K, T)
```

This is solved via Newton-Raphson or bisection.

### Expected Smile Shape

Under Merton's jump diffusion:
- **Negative mu_J** (downward jumps): produces a **left skew** (higher IV for OTM puts)
- **Large sigma_J**: produces a **U-shaped smile** (higher IV for both OTM puts and calls)
- **Large lambda**: amplifies the smile effect
- **Short maturities**: smile is more pronounced
- **Long maturities**: smile flattens as diffusion dominates

### Crypto Smile Characteristics

BTC options on Bybit/Deribit typically show:
- Strong left skew (crash risk premium)
- Higher absolute IV levels (40-80% vs 15-25% for equities)
- Term structure inversion during stress periods
- Smile steepening before major events (ETF decisions, halvings)

---

## 12. Comparison with FFT Methods

### Carr-Madan FFT Approach

The Carr-Madan (1999) method prices options via:

```
C(K) = (exp(-alpha*ln(K))) / pi * integral from 0 to inf of
       exp(-i*v*ln(K)) * psi(v) / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v) dv
```

where psi(v) is the characteristic function of the log-price. For Merton's model:

```
psi_Merton(v) = exp(
  i*v*(ln(S) + (r - sigma^2/2 - lambda*k)*T)
  - sigma^2*v^2*T/2
  + lambda*T*(exp(i*v*mu_J - sigma_J^2*v^2/2) - 1)
)
```

### PINN vs FFT Comparison

| Aspect | FFT (Carr-Madan) | PINN |
|--------|-------------------|------|
| Speed (single price) | ~1ms | ~0.1ms (after training) |
| Training time | None | 5-30 minutes |
| Greeks | Finite differences | Exact (autograd) |
| Flexibility | Needs characteristic function | Any PIDE |
| Exotic options | Limited | Straightforward |
| Calibration | Separate optimization | Built into training |
| American options | Cannot handle directly | Natural extension |
| GPU acceleration | Moderate benefit | Massive benefit |

---

## 13. Application to Crypto Markets

### Why Jumps Matter More in Crypto

Cryptocurrency markets exhibit jump behavior far more frequently than traditional markets:

1. **Liquidation cascades**: Leveraged positions trigger chain liquidations
2. **Regulatory announcements**: China bans, SEC actions, etc.
3. **Exchange events**: Hacks, insolvency (FTX), delistings
4. **Whale movements**: Large transfers triggering panic
5. **Technical failures**: Network congestion, bridge exploits
6. **Market microstructure**: Thin order books amplify moves

### Bybit Data Pipeline

We use Bybit's REST API for:
- Historical OHLCV kline data
- Options chain data (where available)
- Funding rates (as jump risk proxy)

```python
# Fetch BTC/USDT klines from Bybit
endpoint = "https://api.bybit.com/v5/market/kline"
params = {
    "category": "spot",
    "symbol": "BTCUSDT",
    "interval": "D",
    "limit": 1000
}
```

### Calibrating Jump Parameters from Crypto Data

We estimate jump parameters from historical data:

1. **Identify jumps**: Returns exceeding 3*sigma threshold
2. **Jump intensity (lambda)**: Count of jump days / total days * 252
3. **Jump mean (mu_J)**: Average log-return on jump days
4. **Jump volatility (sigma_J)**: Std of log-returns on jump days
5. **Diffusion volatility (sigma)**: Std of log-returns on non-jump days

Typical BTC estimates:
```
sigma = 0.50  (annualized diffusion volatility)
lambda = 12   (about 12 jumps per year)
mu_J = -0.05  (mean jump is -5%)
sigma_J = 0.10 (jump sizes vary +/- 10%)
```

---

## 14. Trading Strategy and Backtesting

### Strategy: Jump Risk Mispricing

The strategy exploits the difference between PINN-implied fair value and market prices:

1. **Calibrate** Merton model parameters from recent data
2. **Price** options using the PINN
3. **Compare** PINN prices with market mid-prices
4. **Trade** when mispricing exceeds threshold:
   - Buy options underpriced by the market (market underestimates jump risk)
   - Sell options overpriced by the market
5. **Delta-hedge** to isolate the volatility/jump component

### Risk Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Sortino Ratio**: Downside-risk-adjusted return
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

---

## 15. Implementation Guide

### Python Implementation

```bash
cd python/
pip install -r requirements.txt
python train.py --asset BTC --exchange bybit
python visualize.py --plot smile
python backtest.py --start 2023-01-01 --end 2024-01-01
```

### Rust Implementation

```bash
cd rust_pinn_jump/
cargo build --release
cargo run --bin fetch_data
cargo run --bin train
cargo run --bin price_options
```

### Project Structure

```
143_pinn_jump_diffusion/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Simple explanation (English)
├── readme.simple.ru.md          # Simple explanation (Russian)
├── python/
│   ├── requirements.txt
│   ├── __init__.py
│   ├── jump_diffusion_pinn.py   # Core PINN model
│   ├── train.py                 # Training pipeline
│   ├── data_loader.py           # Bybit + stock data
│   ├── merton_analytical.py     # Analytical Merton solution
│   ├── greeks.py                # Greeks via autograd
│   ├── visualize.py             # Plots and visualization
│   └── backtest.py              # Strategy backtesting
└── rust_pinn_jump/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs               # Core library
    │   └── bin/
    │       ├── train.rs
    │       ├── price_options.rs
    │       └── fetch_data.rs
    └── examples/
        └── quick_start.rs
```

---

## 16. References

1. **Merton, R. C.** (1976). "Option pricing when underlying stock returns are discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

3. **Carr, P., & Madan, D.** (1999). "Option valuation using the fast Fourier transform." *Journal of Computational Finance*, 2(4), 61-73.

4. **Kou, S. G.** (2002). "A jump-diffusion model for option pricing." *Management Science*, 48(8), 1086-1101.

5. **Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E.** (2021). "DeepXDE: A deep learning library for solving differential equations." *SIAM Review*, 63(1), 208-228.

6. **Physics-Informed Neural Networks for Option Pricing in Illiquid Jump Markets** (2025). ACM Digital Library, doi:10.1145/3760678.3760691.

7. **Salvador, B., Oosterlee, C. W., & van der Meer, R.** (2020). "Financial option valuation by unsupervised learning with artificial neural networks." *Mathematics*, 9(1), 46.

8. **McClenny, L. D., & Brainerd, U. M.** (2020). "Self-adaptive physics-informed neural networks using a soft attention mechanism." *arXiv:2009.04544*.

---

## Summary

This chapter demonstrated how Physics-Informed Neural Networks can solve the Merton jump diffusion PIDE for option pricing. Key takeaways:

1. **Jump diffusion** captures fat tails and the volatility smile that Black-Scholes misses
2. **PINNs** handle the integral term elegantly via Gauss-Hermite quadrature
3. **Automatic differentiation** provides exact Greeks at zero additional cost
4. **Crypto markets** (Bybit) are an ideal application domain due to frequent price jumps
5. **The approach generalizes** to more complex models (Bates, SVJ, etc.)

The combination of physics-based constraints with neural network flexibility produces a pricing engine that is both accurate and computationally efficient after training, making it practical for real-time trading applications.
