"""
Backtesting framework for the Jump Diffusion PINN trading strategy.

Strategy: Jump Risk Mispricing
    1. Calibrate Merton model parameters from recent data
    2. Price options using the PINN (or analytical Merton for speed)
    3. Compare PINN fair value with market mid-prices
    4. Trade when mispricing exceeds threshold
    5. Delta-hedge to isolate vol/jump mispricing PnL

Usage:
    python backtest.py --start 2023-01-01 --end 2024-01-01
    python backtest.py --asset BTC --exchange bybit
"""

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import DataLoader
from merton_analytical import (
    implied_volatility,
    merton_call_price,
    merton_put_price,
)


@dataclass
class Trade:
    """Record of a single trade."""

    entry_date: str
    exit_date: Optional[str]
    direction: str           # "long" or "short"
    option_type: str         # "call" or "put"
    strike: float
    entry_price: float       # Price we entered at
    fair_value: float        # Our model's fair value
    exit_price: Optional[float] = None
    pnl: float = 0.0
    delta_hedge_pnl: float = 0.0
    total_pnl: float = 0.0

    @property
    def mispricing_pct(self) -> float:
        """Mispricing as percentage of fair value."""
        if self.fair_value > 0:
            return (self.fair_value - self.entry_price) / self.fair_value * 100
        return 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: List[Trade] = field(default_factory=list)
    daily_pnl: Optional[np.ndarray] = None
    equity_curve: Optional[np.ndarray] = None
    dates: Optional[list] = None

    @property
    def total_return(self) -> float:
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            return self.equity_curve[-1] / self.equity_curve[0] - 1
        return 0.0

    @property
    def sharpe_ratio(self) -> float:
        if self.daily_pnl is not None and len(self.daily_pnl) > 1:
            mean_ret = np.mean(self.daily_pnl)
            std_ret = np.std(self.daily_pnl)
            if std_ret > 0:
                return mean_ret / std_ret * math.sqrt(252)
        return 0.0

    @property
    def sortino_ratio(self) -> float:
        if self.daily_pnl is not None and len(self.daily_pnl) > 1:
            mean_ret = np.mean(self.daily_pnl)
            downside = self.daily_pnl[self.daily_pnl < 0]
            if len(downside) > 0:
                downside_std = np.std(downside)
                if downside_std > 0:
                    return mean_ret / downside_std * math.sqrt(252)
        return 0.0

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            peak = np.maximum.accumulate(self.equity_curve)
            drawdown = (self.equity_curve - peak) / peak
            return float(np.min(drawdown))
        return 0.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.total_pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.total_pnl for t in self.trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in self.trades if t.total_pnl < 0))
        if gross_loss > 0:
            return gross_profit / gross_loss
        return float("inf") if gross_profit > 0 else 0.0

    def summary(self) -> dict:
        return {
            "n_trades": len(self.trades),
            "total_return": f"{self.total_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.2f}",
        }


class JumpMispricingStrategy:
    """
    Trading strategy based on jump risk mispricing.

    The strategy:
    1. Estimates Merton jump diffusion parameters from a rolling window
    2. Computes model fair values for ATM and OTM options
    3. Compares with simulated market prices (BS-based in backtest)
    4. Enters trades when mispricing exceeds a threshold
    5. Exits after a holding period or when mispricing reverts

    In live trading, market prices would come from the exchange.
    In backtesting, we simulate market prices using Black-Scholes
    (which underestimates jumps) to create a realistic mispricing signal.
    """

    def __init__(
        self,
        calibration_window: int = 60,
        holding_period: int = 5,
        mispricing_threshold: float = 0.05,
        max_positions: int = 5,
        position_size: float = 10000.0,
        risk_free_rate: float = 0.05,
        option_maturity: float = 30 / 365,
    ):
        """
        Initialize the strategy.

        Args:
            calibration_window: Days for parameter estimation
            holding_period: Days to hold each trade
            mispricing_threshold: Minimum mispricing % to enter trade
            max_positions: Maximum simultaneous positions
            position_size: Notional per trade
            risk_free_rate: Risk-free rate for pricing
            option_maturity: Hypothetical option maturity in years
        """
        self.calibration_window = calibration_window
        self.holding_period = holding_period
        self.mispricing_threshold = mispricing_threshold
        self.max_positions = max_positions
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate
        self.option_maturity = option_maturity

    def run_backtest(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> BacktestResult:
        """
        Run the backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for close prices

        Returns:
            BacktestResult with trades, PnL, and metrics
        """
        prices = df[price_col].values
        dates = df["timestamp"].values if "timestamp" in df.columns else list(range(len(df)))
        n = len(prices)

        result = BacktestResult()
        result.dates = dates.tolist()

        equity = [self.position_size * self.max_positions]
        daily_pnl_list = []
        open_trades: List[Tuple[Trade, int]] = []  # (Trade, exit_day_index)

        start_idx = self.calibration_window + 1

        for i in range(start_idx, n):
            day_pnl = 0.0
            current_price = prices[i]
            current_date = str(dates[i])

            # Close expired trades
            trades_to_close = []
            for j, (trade, exit_idx) in enumerate(open_trades):
                if i >= exit_idx:
                    # Compute exit price using Merton model with current spot
                    remaining_tau = max(
                        self.option_maturity - self.holding_period / 252.0, 1.0 / 365
                    )
                    pricer = merton_call_price if trade.option_type == "call" else merton_put_price

                    # Re-estimate params for current window
                    window_returns = np.diff(np.log(prices[max(0, i - self.calibration_window):i + 1]))
                    sigma_est = float(np.std(window_returns) * math.sqrt(252))

                    exit_fair_value = pricer(
                        current_price, trade.strike, remaining_tau,
                        self.risk_free_rate, sigma_est,
                        5.0, -0.05, 0.10,  # Simple params for exit pricing
                    )

                    trade.exit_price = exit_fair_value
                    trade.exit_date = current_date

                    # PnL from option position
                    if trade.direction == "long":
                        trade.pnl = (exit_fair_value - trade.entry_price) * self.position_size / trade.entry_price
                    else:
                        trade.pnl = (trade.entry_price - exit_fair_value) * self.position_size / trade.entry_price

                    # Simplified delta hedge PnL
                    spot_change = (current_price - prices[exit_idx - self.holding_period]) / prices[exit_idx - self.holding_period]
                    trade.delta_hedge_pnl = -trade.pnl * 0.3 * abs(spot_change)  # Rough approximation

                    trade.total_pnl = trade.pnl + trade.delta_hedge_pnl
                    day_pnl += trade.total_pnl

                    result.trades.append(trade)
                    trades_to_close.append(j)

            # Remove closed trades (reverse order to preserve indices)
            for j in sorted(trades_to_close, reverse=True):
                open_trades.pop(j)

            # Check for new entry signals
            if len(open_trades) < self.max_positions:
                window = prices[i - self.calibration_window:i + 1]
                log_returns = np.diff(np.log(window))

                # Estimate parameters
                sigma_est = float(np.std(log_returns) * math.sqrt(252))
                sigma_est = max(sigma_est, 0.05)

                # Detect jumps in window
                mad = float(np.median(np.abs(log_returns - np.median(log_returns))))
                sigma_robust = mad * 1.4826
                is_jump = np.abs(log_returns - np.median(log_returns)) > 3 * sigma_robust
                jump_rets = log_returns[is_jump]

                n_jumps = int(np.sum(is_jump))
                lambda_est = max(n_jumps / len(log_returns) * 252, 0.5)
                mu_j_est = float(np.mean(jump_rets)) if n_jumps > 0 else -0.05
                sigma_j_est = float(np.std(jump_rets)) if n_jumps > 1 else 0.10

                # Price ATM call with Merton (our model)
                strike = current_price
                merton_price = merton_call_price(
                    current_price, strike, self.option_maturity,
                    self.risk_free_rate, sigma_est,
                    lambda_est, mu_j_est, sigma_j_est,
                )

                # "Market" price: use Black-Scholes (no jumps) as the market estimate
                from merton_analytical import black_scholes_call
                bs_price = black_scholes_call(
                    current_price, strike, self.option_maturity,
                    self.risk_free_rate, sigma_est,
                )

                # Mispricing signal
                if merton_price > 0 and bs_price > 0:
                    mispricing = (merton_price - bs_price) / bs_price

                    if mispricing > self.mispricing_threshold:
                        # Market (BS) underprices jump risk -> buy the option
                        trade = Trade(
                            entry_date=current_date,
                            exit_date=None,
                            direction="long",
                            option_type="call",
                            strike=strike,
                            entry_price=bs_price,
                            fair_value=merton_price,
                        )
                        exit_idx = min(i + self.holding_period, n - 1)
                        open_trades.append((trade, exit_idx))

                    elif mispricing < -self.mispricing_threshold:
                        # Rare: market overprices jump risk -> sell
                        trade = Trade(
                            entry_date=current_date,
                            exit_date=None,
                            direction="short",
                            option_type="call",
                            strike=strike,
                            entry_price=bs_price,
                            fair_value=merton_price,
                        )
                        exit_idx = min(i + self.holding_period, n - 1)
                        open_trades.append((trade, exit_idx))

            daily_pnl_list.append(day_pnl)
            equity.append(equity[-1] + day_pnl)

        result.daily_pnl = np.array(daily_pnl_list)
        result.equity_curve = np.array(equity)

        return result


def plot_backtest_results(result: BacktestResult, save_path: Optional[str] = None):
    """
    Plot backtest equity curve and trade statistics.

    Args:
        result: BacktestResult object
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Equity curve
    axes[0, 0].plot(result.equity_curve, "b-", linewidth=1.5)
    axes[0, 0].set_title("Equity Curve")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Portfolio Value")
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative PnL
    if result.daily_pnl is not None and len(result.daily_pnl) > 0:
        cum_pnl = np.cumsum(result.daily_pnl)
        axes[0, 1].plot(cum_pnl, "g-", linewidth=1.5)
        axes[0, 1].set_title("Cumulative PnL")
        axes[0, 1].set_xlabel("Day")
        axes[0, 1].set_ylabel("PnL")
        axes[0, 1].grid(True, alpha=0.3)

    # Drawdown
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        peak = np.maximum.accumulate(result.equity_curve)
        drawdown = (result.equity_curve - peak) / peak * 100
        axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color="red")
        axes[1, 0].set_title("Drawdown (%)")
        axes[1, 0].set_xlabel("Day")
        axes[1, 0].set_ylabel("Drawdown %")
        axes[1, 0].grid(True, alpha=0.3)

    # Trade PnL distribution
    if result.trades:
        trade_pnls = [t.total_pnl for t in result.trades]
        axes[1, 1].hist(trade_pnls, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        axes[1, 1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Trade PnL Distribution")
        axes[1, 1].set_xlabel("PnL")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Jump Mispricing Strategy Backtest", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Backtest Jump Mispricing Strategy")
    parser.add_argument("--asset", type=str, default="BTC", help="Asset name")
    parser.add_argument("--exchange", type=str, default="bybit", help="Exchange")
    parser.add_argument("--start", type=str, default=None, help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
    parser.add_argument("--window", type=int, default=60, help="Calibration window")
    parser.add_argument("--holding", type=int, default=5, help="Holding period")
    parser.add_argument("--threshold", type=float, default=0.05, help="Mispricing threshold")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    # Load data
    loader = DataLoader()
    if args.exchange.lower() == "bybit":
        df = loader.fetch_bybit(
            f"{args.asset.upper()}USDT", interval="D", limit=1000
        )
    else:
        df = loader.fetch_yahoo(args.asset.upper(), period="2y")

    print(f"Loaded {len(df)} rows of {args.asset} data")

    # Filter by date if provided
    if args.start:
        df = df[df["timestamp"] >= args.start]
    if args.end:
        df = df[df["timestamp"] <= args.end]

    print(f"After date filter: {len(df)} rows")

    # Run backtest
    strategy = JumpMispricingStrategy(
        calibration_window=args.window,
        holding_period=args.holding,
        mispricing_threshold=args.threshold,
    )

    result = strategy.run_backtest(df)

    # Print summary
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    summary = result.summary()
    for k, v in summary.items():
        print(f"  {k:>20}: {v}")
    print("=" * 50)

    # Plot results
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    plot_backtest_results(result, save_path=os.path.join(args.save_dir, "backtest.png"))


if __name__ == "__main__":
    main()
