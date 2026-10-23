#!/usr/bin/env python3
"""
Sentiment-Driven Markov Regime Switching Portfolio Optimizer
-----------------------------------------------------------
Historical research/backtesting tool; not investment advice.

Important modeling notes:
1. statsmodels MarkovRegression's `exog` enters the observation/mean equation.
   It does NOT make transition probabilities depend on sentiment. For genuinely
   sentiment-dependent transition probabilities, use a model that explicitly
   supports time-varying transition probabilities (e.g. a custom likelihood).
2. Smoothed probabilities use future observations. They are useful for
   retrospective regime visualization, but are not causal trading signals.
   This script uses one-step-ahead FILTERED probabilities for the backtest to
   avoid look-ahead bias, and plots smoothed probabilities for diagnosis.
3. The Alternative.me API is crypto-focused. If it is unavailable, the fallback
   sentiment is explicitly synthetic and must not be mistaken for observed data.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

LOG = logging.getLogger("sentiment_regime_switching")
TRADING_DAYS = 365.25
EPSILON = 1e-6
DEFAULT_START = "2018-01-01"


@dataclass
class DataBundle:
    frame: pd.DataFrame
    sentiment_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a sentiment-informed, two-state Markov regime strategy."
    )
    parser.add_argument("--asset", default="BTC-USD", help="Yahoo Finance ticker (default: BTC-USD)")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Exclusive end date YYYY-MM-DD (default: today)")
    parser.add_argument("--threshold", type=float, default=0.5, help="High-risk probability threshold")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost per allocation switch, in bps")
    parser.add_argument("--sentiment-url", default="https://api.alternative.me/fng/?limit=0&format=json",
                        help="Alternative.me Fear & Greed API endpoint")
    parser.add_argument("--output", default="regime_switching_performance.png", help="Output chart path")
    parser.add_argument("--sentiment-csv", default=None,
                        help="Optional CSV with date and sentiment columns; takes precedence over API")
    parser.add_argument("--seed", type=int, default=7, help="Seed for synthetic fallback sentiment")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _normalise_daily_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    idx = idx[~idx.isna()]
    return idx.tz_convert(None).normalize()


def fetch_prices(asset: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Download adjusted daily close and calculate daily log returns."""
    LOG.info("Downloading %s prices from %s to %s", asset, start, end or "today")
    try:
        raw = yf.download(
            asset, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed for {asset}: {exc}") from exc

    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {asset}. Check ticker, dates, or network.")

    # yfinance may return MultiIndex columns, including for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        close_block = raw["Close"] if "Close" in raw.columns.get_level_values(0) else None
        if close_block is None:
            raise RuntimeError("Could not locate Close column in Yahoo Finance response.")
        close = close_block.iloc[:, 0] if isinstance(close_block, pd.DataFrame) else close_block
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("Could not locate Close column in Yahoo Finance response.")
        close = raw["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = _normalise_daily_index(close.index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close > 0]
    if len(close) < 250:
        raise RuntimeError(f"Only {len(close)} valid price observations; need at least 250.")

    frame = pd.DataFrame({"price": close})
    frame["log_return"] = np.log(frame["price"]).diff()
    return frame.dropna()


def _read_sentiment_csv(path: str) -> pd.Series:
    raw = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    date_col = next((lower[k] for k in ("date", "timestamp", "time") if k in lower), None)
    value_col = next((lower[k] for k in ("sentiment", "value", "fear_greed", "fear & greed index") if k in lower), None)
    if date_col is None or value_col is None:
        raise ValueError("Sentiment CSV must contain a date/timestamp column and a sentiment/value column.")
    dates = pd.to_datetime(raw[date_col], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    values = pd.to_numeric(raw[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name="sentiment").dropna()
    series = series[~series.index.isna()]
    return series[~series.index.duplicated(keep="last")].sort_index()


def fetch_alternative_sentiment(url: str) -> pd.Series:
    """Fetch Alternative.me Fear & Greed daily values (0-100)."""
    import json
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "sentiment-regime-research/1.0"})
    with urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Alternative.me response did not contain a non-empty 'data' list.")
    raw = pd.DataFrame(rows)
    if not {"timestamp", "value"}.issubset(raw.columns):
        raise ValueError("Alternative.me response is missing timestamp/value fields.")
    dates = pd.to_datetime(pd.to_numeric(raw["timestamp"], errors="coerce"), unit="s", utc=True,
                           errors="coerce").dt.tz_convert(None).dt.normalize()
    values = pd.to_numeric(raw["value"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name="sentiment").dropna()
    series = series[~series.index.isna()]
    return series[~series.index.duplicated(keep="last")].sort_index()


def synthetic_sentiment(price_frame: pd.DataFrame, seed: int) -> pd.Series:
    """
    Deterministic fallback proxy (0-100), based only on past market behaviour.
    It is synthetic, not observed investor sentiment.
    """
    rng = np.random.default_rng(seed)
    ret = price_frame["log_return"]
    momentum = ret.rolling(14, min_periods=5).sum().shift(1)
    volatility = ret.rolling(14, min_periods=5).std().shift(1)
    momentum = momentum.fillna(0.0)
    volatility = volatility.fillna(volatility.median() if volatility.notna().any() else 0.0)
    scale = float(momentum.std()) if momentum.std() > 0 else 1.0
    vol_scale = float(volatility.median()) if volatility.median() > 0 else 1.0
    score = 50 + 22 * np.tanh(momentum / (scale + EPSILON)) - 12 * np.tanh(
        (volatility / (vol_scale + EPSILON)) - 1.0
    ) + rng.normal(0, 5, len(price_frame))
    return pd.Series(np.clip(score, 0, 100), index=price_frame.index, name="sentiment")


def build_dataset(args: argparse.Namespace) -> DataBundle:
    prices = fetch_prices(args.asset, args.start, args.end)
    source = "Alternative.me Fear & Greed API"
    try:
        if args.sentiment_csv:
            sentiment = _read_sentiment_csv(args.sentiment_csv)
            source = f"CSV: {args.sentiment_csv}"
        else:
            sentiment = fetch_alternative_sentiment(args.sentiment_url)
        if sentiment.empty:
            raise ValueError("Sentiment series is empty.")
    except Exception as exc:
        LOG.warning("Observed sentiment unavailable (%s). Using SYNTHETIC fallback.", exc)
        sentiment = synthetic_sentiment(prices, args.seed)
        source = "synthetic market-derived proxy (NOT observed sentiment)"

    # Align sentiment onto market dates, carrying the most recent known daily value.
    # Sentiment is shifted again below before model fitting.
    sentiment = sentiment.sort_index()
    sentiment = sentiment[~sentiment.index.duplicated(keep="last")]
    aligned = prices.join(sentiment.rename("sentiment"), how="left")
    aligned["sentiment"] = aligned["sentiment"].ffill()
    aligned = aligned.dropna(subset=["price", "log_return", "sentiment"])
    aligned["sentiment_lag1"] = aligned["sentiment"].shift(1)
    aligned = aligned.dropna(subset=["sentiment_lag1"])
    aligned["sentiment_z"] = (aligned["sentiment_lag1"] - aligned["sentiment_lag1"].mean()) / (
        aligned["sentiment_lag1"].std(ddof=0) + EPSILON
    )
    if len(aligned) < 250:
        raise RuntimeError(f"Only {len(aligned)} aligned observations remain after cleaning.")
    LOG.info("Aligned observations: %d; sentiment source: %s", len(aligned), source)
    return DataBundle(aligned, source)


def fit_markov_model(frame: pd.DataFrame):
    """Fit two-state Markov regression with robust optimizer fallbacks."""
    y = frame["log_return"].astype(float)
    exog = frame[["sentiment_z"]].astype(float)
    model = MarkovRegression(
        endog=y,
        k_regimes=2,
        trend="c",
        exog=exog,
        switching_variance=True,
    )

    errors = []
    for method in ("bfgs", "powell", "lbfgs"):
        try:
            LOG.info("Fitting MarkovRegression with optimizer=%s", method)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    method=method, maxiter=500, disp=False,
                    search_reps=20 if method == "bfgs" else 0,
                    em_iter=10,
                )
            convergence_warnings = [
                w for w in caught if issubclass(w.category, ConvergenceWarning)
            ]
            converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
            if convergence_warnings:
                LOG.warning("%s emitted %d ConvergenceWarning(s).", method, len(convergence_warnings))
            if converged and np.isfinite(result.llf):
                LOG.info("Model converged with %s (log-likelihood %.3f).", method, result.llf)
                return result
            errors.append(f"{method}: optimizer did not report convergence")
            LOG.warning("%s did not converge; trying fallback.", method)
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
            LOG.warning("Fit failed with %s: %s", method, exc)

    raise RuntimeError("All Markov model optimizers failed: " + " | ".join(errors))

   
def identify_states(result) -> Tuple[int, int, pd.Series]:
    """Identify low- and high-variance states from fitted model parameters."""
    param_names = list(result.model.param_names)
    params = np.asarray(result.params, dtype=float).ravel()
    param_map = dict(zip(param_names, params))
    variances = {}

    for state in range(2):
        candidates = [
            name for name in param_names
            if name.lower().startswith("sigma2")
            and f"[{state}]" in name
        ]

        if candidates:
            variance = float(param_map[candidates[0]])
        else:
            probabilities = np.asarray(
                result.smoothed_marginal_probabilities
            )[:, state]
            values = np.asarray(result.model.endog, dtype=float)

            valid = np.isfinite(probabilities) & np.isfinite(values)
            weights = probabilities[valid]
            observations = values[valid]

            if len(observations) == 0 or weights.sum() <= 0:
                raise RuntimeError(
                    f"Unable to estimate variance for state {state}."
                )

            mean = np.average(observations, weights=weights)
            variance = np.average(
                (observations - mean) ** 2,
                weights=weights,
            )

        if not np.isfinite(variance) or variance <= 0:
            raise RuntimeError(
                f"Invalid variance estimate for state {state}: {variance}"
            )

        variances[state] = variance

    variance_series = pd.Series(variances, dtype=float)
    high_risk_state = int(variance_series.idxmax())
    quiet_state = 1 - high_risk_state

    return quiet_state, high_risk_state, variance_series



def get_probability_series(result, frame: pd.DataFrame, state: int, kind: str) -> pd.Series:
    probs = getattr(result, f"{kind}_marginal_probabilities", None)
    if probs is None:
        raise RuntimeError(f"Model result has no {kind} marginal probabilities.")
    values = np.asarray(probs)
    if values.ndim != 2 or values.shape[1] != 2:
        raise RuntimeError(f"Unexpected {kind} probability matrix shape: {values.shape}")
    return pd.Series(values[:, state], index=frame.index, name=f"p_high_{kind}").clip(0, 1)


def compute_metrics(returns: pd.Series) -> dict:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {k: np.nan for k in ("cumulative_return", "annualized_return", "annualized_volatility",
                                    "sharpe", "sortino", "max_drawdown")}
    equity = np.exp(r.cumsum())
    periods = len(r)
    cumulative = float(equity.iloc[-1] - 1.0)
    annualized_return = float(
    np.exp(r.sum() * TRADING_DAYS / periods) - 1.0
    )
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if periods > 1 else np.nan
    sharpe = float((r.mean() / (r.std(ddof=1) + EPSILON)) * np.sqrt(TRADING_DAYS)) if periods > 1 else np.nan
    downside = r.clip(upper=0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(TRADING_DAYS))
    sortino = float((r.mean() * TRADING_DAYS) / (downside_dev + EPSILON))
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
    }


def backtest(frame: pd.DataFrame, p_high_filtered: pd.Series, p_high_smoothed: pd.Series,
             threshold: float, cost_bps: float) -> pd.DataFrame:
    """
    Fully vectorized execution. Filtered probabilities are causal; smoothed
    probabilities are retained only for retrospective analysis/plot shading.
    """
    out = frame.copy()
    out["p_high_filtered"] = p_high_filtered
    out["p_high_smoothed"] = p_high_smoothed
    out["signal_risky"] = (out["p_high_filtered"] <= threshold).astype(int)
    out["position"] = out["signal_risky"].shift(1).fillna(0).astype(int)
    out["asset_log_return"] = out["log_return"]
    out["gross_strategy_log_return"] = out["position"] * out["asset_log_return"]

    # A switch between cash and risky asset is one rebalance event.
    out["trade"] = out["position"].diff().abs().fillna(out["position"].abs()).astype(int)
    cost_fraction = float(cost_bps) / 10_000.0
    # Convert proportional transaction fee into log-return units.
    log_cost = -np.log1p(-cost_fraction) if cost_fraction < 1 else np.inf
    out["transaction_cost_log"] = out["trade"] * log_cost
    out["strategy_log_return"] = out["gross_strategy_log_return"] - out["transaction_cost_log"]
    out["buy_hold_log_return"] = out["asset_log_return"]
    out["strategy_equity"] = np.exp(out["strategy_log_return"].cumsum())
    out["buy_hold_equity"] = np.exp(out["buy_hold_log_return"].cumsum())
    # Retrospective label for charting, not used to trade.
    out["high_risk_smoothed"] = (out["p_high_smoothed"] > threshold).astype(int)
    return out


def plot_report(frame: pd.DataFrame, asset: str, output: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    ax_price, ax_equity = axes

    ax_price.plot(frame.index, frame["price"], linewidth=1.15, label=f"{asset} adjusted close")
    high = frame["high_risk_smoothed"].astype(bool).to_numpy()
    # Shade contiguous high-risk periods without an iterative per-row loop.
    transitions = np.diff(np.r_[False, high, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    for start_i, end_i in zip(starts, ends):
        ax_price.axvspan(frame.index[start_i],
                         frame.index[min(end_i, len(frame.index) - 1)],
                         color="red", alpha=0.13, linewidth=0)
    ax_price.set_title("Asset Price with Retrospective High-Volatility Regime Shading")
    ax_price.set_ylabel("Adjusted price")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left")
    ax_price.text(
        0.01, 0.02, "Red shading uses smoothed probabilities (retrospective; not a live signal).",
        transform=ax_price.transAxes, fontsize=9
    )

    ax_equity.plot(frame.index, frame["strategy_equity"], label="Regime strategy (after fees)", linewidth=1.4)
    ax_equity.plot(frame.index, frame["buy_hold_equity"], label="Buy & hold", linewidth=1.2, alpha=0.85)
    ax_equity.set_title("Strategy Equity Curve vs. Buy-and-Hold")
    ax_equity.set_ylabel("Growth of $1")
    ax_equity.set_xlabel("Date")
    ax_equity.grid(True, alpha=0.25)
    ax_equity.legend(loc="upper left")
    fig.suptitle(f"Sentiment-Driven Markov Regime Switching — {asset}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved performance chart to %s", output)


def print_report(frame: pd.DataFrame, result, quiet_state: int, high_state: int,
                 variances: pd.Series, source: str, asset: str) -> None:
    strategy = compute_metrics(frame["strategy_log_return"])
    benchmark = compute_metrics(frame["buy_hold_log_return"])
    total_trades = int(frame["trade"].sum())
    print("\n" + "=" * 76)
    print("SENTIMENT-DRIVEN MARKOV REGIME SWITCHING — BACKTEST REPORT")
    print("=" * 76)
    print(f"Asset:                 {asset}")
    print(f"Period:                {frame.index.min().date()} to {frame.index.max().date()}")
    print(f"Observations:          {len(frame):,}")
    print(f"Sentiment source:      {source}")
    print(f"Model log-likelihood:  {result.llf:.3f}")
    print(f"Quiet/low-vol state:   State {quiet_state} (estimated variance {variances[quiet_state]:.8f})")
    print(f"High-risk state:       State {high_state} (estimated variance {variances[high_state]:.8f})")
    print(f"Allocation switches:   {total_trades:,}")
    print(f"Strategy cumulative:   {strategy['cumulative_return']:.2%}")
    print("\nMetric                  Strategy       Buy & Hold")
    print("-" * 55)
    labels = [
        ("Annualized return", "annualized_return", "{:.2%}"),
        ("Annualized volatility", "annualized_volatility", "{:.2%}"),
        ("Sharpe ratio", "sharpe", "{:.3f}"),
        ("Sortino ratio", "sortino", "{:.3f}"),
        ("Maximum drawdown", "max_drawdown", "{:.2%}"),
        ("Cumulative return", "cumulative_return", "{:.2%}"),
    ]
    for label, key, fmt in labels:
        print(f"{label:<23} {fmt.format(strategy[key]):>12} {fmt.format(benchmark[key]):>15}")
    print("=" * 76)
    print("Research caveat: fitted MarkovRegression exog affects the return mean, not")
    print("transition probabilities. Smoothed regimes are retrospective; trading uses")
    print("lagged filtered probabilities. Synthetic sentiment is not observed sentiment.")
    print("This backtest excludes taxes, spread, market impact, funding, and cash yield.\n")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if not 0 < args.threshold < 1:
        LOG.error("--threshold must be strictly between 0 and 1.")
        return 2
    if args.cost_bps < 0 or args.cost_bps >= 10_000:
        LOG.error("--cost-bps must be in [0, 10000).")
        return 2

    try:
        bundle = build_dataset(args)
        frame = bundle.frame
        result = fit_markov_model(frame)
        quiet_state, high_state, variances = identify_states(result)
        p_high_filtered = get_probability_series(result, frame, high_state, "filtered")
        p_high_smoothed = get_probability_series(result, frame, high_state, "smoothed")
        results = backtest(frame, p_high_filtered, p_high_smoothed, args.threshold, args.cost_bps)
        print_report(results, result, quiet_state, high_state, variances, bundle.sentiment_source, args.asset)
        plot_report(results, args.asset, args.output)
        return 0
    except Exception as exc:
        LOG.exception("Run failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
