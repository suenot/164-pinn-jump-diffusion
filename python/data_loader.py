"""
Data loader for market data from Bybit (crypto) and Yahoo Finance (stocks).

Provides functionality to:
    - Fetch OHLCV data from Bybit REST API
    - Fetch stock data from Yahoo Finance via yfinance
    - Estimate Merton jump diffusion parameters from historical returns
    - Identify jump events using statistical thresholds

Usage:
    loader = DataLoader()
    df = loader.fetch_bybit("BTCUSDT", interval="D", limit=1000)
    params = loader.estimate_jump_params(df)
"""

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests


@dataclass
class EstimatedParams:
    """Estimated Merton jump diffusion parameters from market data."""

    sigma: float        # Annualized diffusion volatility (non-jump days)
    lambda_j: float     # Jump intensity (jumps per year)
    mu_j: float         # Mean log-jump size
    sigma_j: float      # Jump size volatility
    n_jumps: int        # Number of detected jumps
    n_observations: int # Total number of observations
    threshold: float    # Jump detection threshold (in sigmas)

    def __str__(self) -> str:
        return (
            f"EstimatedParams(\n"
            f"  sigma={self.sigma:.4f}, lambda_j={self.lambda_j:.2f},\n"
            f"  mu_j={self.mu_j:.4f}, sigma_j={self.sigma_j:.4f},\n"
            f"  n_jumps={self.n_jumps}/{self.n_observations},\n"
            f"  threshold={self.threshold:.1f} sigma\n)"
        )


class DataLoader:
    """
    Multi-source data loader for financial time series.

    Supports:
        - Bybit API (crypto: BTC, ETH, SOL, etc.)
        - Yahoo Finance (stocks: SPY, AAPL, etc.)
    """

    BYBIT_BASE_URL = "https://api.bybit.com"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            cache_dir: Optional directory to cache fetched data
        """
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })

    def fetch_bybit(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "D",
        limit: int = 1000,
        category: str = "spot",
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV kline data from Bybit REST API.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Kline interval ("1", "3", "5", "15", "30", "60", "120",
                      "240", "360", "720", "D", "W", "M")
            limit: Number of candles to fetch (max 1000)
            category: Market category ("spot", "linear", "inverse")
            end_time: Optional end timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        endpoint = f"{self.BYBIT_BASE_URL}/v5/market/kline"

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if end_time is not None:
            params["end"] = end_time

        all_data = []
        remaining = limit
        current_end = end_time

        while remaining > 0:
            params["limit"] = min(remaining, 1000)
            if current_end is not None:
                params["end"] = current_end

            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("retCode") != 0:
                    print(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                    break

                result_list = data.get("result", {}).get("list", [])
                if not result_list:
                    break

                all_data.extend(result_list)
                remaining -= len(result_list)

                if len(result_list) < 1000:
                    break

                # Set end_time to the oldest timestamp - 1 for pagination
                oldest_ts = int(result_list[-1][0])
                current_end = oldest_ts - 1

                # Rate limiting
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Request error fetching Bybit data: {e}")
                break

        if not all_data:
            print("No data fetched from Bybit. Generating synthetic data.")
            return self._generate_synthetic_crypto(symbol, limit)

        # Parse data
        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_yahoo(
        self,
        symbol: str = "SPY",
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol
            period: Data period (e.g., "1y", "2y", "5y", "max")
            interval: Data interval (e.g., "1d", "1wk")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"No data from Yahoo Finance for {symbol}. Generating synthetic data.")
                return self._generate_synthetic_equity(symbol, 500)

            df = df.reset_index()
            df = df.rename(columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df

        except ImportError:
            print("yfinance not installed. Generating synthetic equity data.")
            return self._generate_synthetic_equity(symbol, 500)

    def _generate_synthetic_crypto(
        self,
        symbol: str,
        n_days: int,
    ) -> pd.DataFrame:
        """Generate synthetic crypto-like OHLCV data with jumps."""
        np.random.seed(42)

        base_price = 50000.0 if "BTC" in symbol.upper() else 3000.0
        sigma_daily = 0.03
        lambda_daily = 12.0 / 252.0
        mu_j = -0.05
        sigma_j = 0.10

        prices = [base_price]
        for _ in range(n_days - 1):
            # Normal diffusion
            ret = np.random.normal(0, sigma_daily)
            # Jump component
            if np.random.random() < lambda_daily:
                ret += np.random.normal(mu_j, sigma_j)
            prices.append(prices[-1] * math.exp(ret))

        prices = np.array(prices)
        timestamps = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

        # Generate OHLCV from close prices
        noise = np.random.uniform(0.005, 0.02, n_days)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.005, n_days)),
            "high": prices * (1 + noise),
            "low": prices * (1 - noise),
            "close": prices,
            "volume": np.random.lognormal(20, 1, n_days),
        })
        return df

    def _generate_synthetic_equity(
        self,
        symbol: str,
        n_days: int,
    ) -> pd.DataFrame:
        """Generate synthetic equity-like OHLCV data with jumps."""
        np.random.seed(123)

        base_price = 100.0
        sigma_daily = 0.012
        lambda_daily = 5.0 / 252.0
        mu_j = -0.03
        sigma_j = 0.08

        prices = [base_price]
        for _ in range(n_days - 1):
            ret = np.random.normal(0.0003, sigma_daily)
            if np.random.random() < lambda_daily:
                ret += np.random.normal(mu_j, sigma_j)
            prices.append(prices[-1] * math.exp(ret))

        prices = np.array(prices)
        timestamps = pd.date_range(end=datetime.now(), periods=n_days, freq="B")

        noise = np.random.uniform(0.003, 0.01, n_days)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.003, n_days)),
            "high": prices * (1 + noise),
            "low": prices * (1 - noise),
            "close": prices,
            "volume": np.random.lognormal(15, 1, n_days),
        })
        return df

    @staticmethod
    def estimate_jump_params(
        df: pd.DataFrame,
        price_col: str = "close",
        threshold_sigma: float = 3.0,
        annualization_factor: float = 252.0,
    ) -> EstimatedParams:
        """
        Estimate Merton jump diffusion parameters from historical returns.

        Method:
            1. Compute log-returns
            2. Estimate initial volatility (MAD-based robust estimator)
            3. Identify jumps as returns exceeding threshold * sigma
            4. Separate diffusion (non-jump) and jump components
            5. Estimate parameters for each

        Args:
            df: DataFrame with price data
            price_col: Column name for prices
            threshold_sigma: Number of sigmas for jump detection
            annualization_factor: 252 for daily, 365 for crypto daily

        Returns:
            EstimatedParams with estimated model parameters
        """
        prices = df[price_col].values
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)

        # Robust volatility estimate using MAD (Median Absolute Deviation)
        median_ret = np.median(log_returns)
        mad = np.median(np.abs(log_returns - median_ret))
        sigma_robust = mad * 1.4826  # MAD to sigma conversion

        # Identify jumps
        threshold = threshold_sigma * sigma_robust
        is_jump = np.abs(log_returns - median_ret) > threshold
        jump_returns = log_returns[is_jump]
        non_jump_returns = log_returns[~is_jump]

        n_jumps = int(np.sum(is_jump))

        # Diffusion volatility (annualized, from non-jump days)
        if len(non_jump_returns) > 1:
            sigma = float(np.std(non_jump_returns) * math.sqrt(annualization_factor))
        else:
            sigma = float(sigma_robust * math.sqrt(annualization_factor))

        # Jump intensity (annualized)
        lambda_j = float(n_jumps / n * annualization_factor)

        # Jump size distribution parameters
        if n_jumps > 0:
            mu_j = float(np.mean(jump_returns))
            sigma_j = float(np.std(jump_returns)) if n_jumps > 1 else 0.10
        else:
            mu_j = -0.05
            sigma_j = 0.10

        return EstimatedParams(
            sigma=sigma,
            lambda_j=max(lambda_j, 0.1),  # At least 0.1 jumps/year
            mu_j=mu_j,
            sigma_j=max(sigma_j, 0.01),
            n_jumps=n_jumps,
            n_observations=n,
            threshold=threshold_sigma,
        )

    @staticmethod
    def compute_return_statistics(df: pd.DataFrame, price_col: str = "close") -> dict:
        """
        Compute return distribution statistics for comparison with models.

        Args:
            df: DataFrame with price data
            price_col: Column name for prices

        Returns:
            Dictionary of statistics
        """
        prices = df[price_col].values
        log_returns = np.diff(np.log(prices))

        from scipy.stats import kurtosis, skew, jarque_bera

        jb_stat, jb_pval = jarque_bera(log_returns)

        return {
            "n_observations": len(log_returns),
            "mean_daily_return": float(np.mean(log_returns)),
            "std_daily_return": float(np.std(log_returns)),
            "annualized_volatility": float(np.std(log_returns) * math.sqrt(252)),
            "skewness": float(skew(log_returns)),
            "excess_kurtosis": float(kurtosis(log_returns)),
            "min_return": float(np.min(log_returns)),
            "max_return": float(np.max(log_returns)),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pval": float(jb_pval),
            "pct_above_3sigma": float(np.mean(np.abs(log_returns) > 3 * np.std(log_returns)) * 100),
        }


def main():
    """Demo: fetch data and estimate jump parameters."""
    loader = DataLoader()

    print("=" * 60)
    print("Fetching BTC/USDT data from Bybit...")
    print("=" * 60)
    btc_df = loader.fetch_bybit("BTCUSDT", interval="D", limit=500)
    print(f"Fetched {len(btc_df)} rows")
    print(btc_df.tail())

    print("\nEstimating jump parameters...")
    btc_params = loader.estimate_jump_params(btc_df, annualization_factor=365)
    print(btc_params)

    print("\nReturn statistics:")
    stats = loader.compute_return_statistics(btc_df)
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Fetching SPY data from Yahoo Finance...")
    print("=" * 60)
    spy_df = loader.fetch_yahoo("SPY", period="2y")
    print(f"Fetched {len(spy_df)} rows")
    print(spy_df.tail())

    print("\nEstimating jump parameters...")
    spy_params = loader.estimate_jump_params(spy_df)
    print(spy_params)


if __name__ == "__main__":
    main()
