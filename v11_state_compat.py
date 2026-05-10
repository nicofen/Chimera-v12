"""
tests/conftest.py
Shared pytest fixtures — deterministic market data, pre-wired state objects,
and helper factories used across the entire test suite.

All price series are generated with a fixed seed so tests are reproducible.
No real API calls are made anywhere in the test suite.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from chimera_v12.utils.state import SharedState, TechnicalSignals, RiskParameters
from chimera_v12.oms.models import Order, OrderSide, OrderStatus
from chimera_v12.backtest.state import BacktestState


# ── RNG with fixed seed ───────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# ── Price series factories ────────────────────────────────────────────────────

@pytest.fixture
def trending_up(rng) -> np.ndarray:
    """300-bar uptrending price series — EMA ribbon aligned, ADX > 25."""
    base   = 100.0
    noise  = rng.normal(0, 0.3, 300)
    trend  = np.linspace(0, 30, 300)
    return base + trend + np.cumsum(noise)


@pytest.fixture
def trending_down(rng) -> np.ndarray:
    """300-bar downtrending price series."""
    base  = 200.0
    noise = rng.normal(0, 0.3, 300)
    trend = np.linspace(0, -40, 300)
    return base + trend + np.cumsum(noise)


@pytest.fixture
def ranging(rng) -> np.ndarray:
    """300-bar ranging/sideways price series — low ADX."""
    base  = 150.0
    noise = rng.normal(0, 0.5, 300)
    cycle = np.sin(np.linspace(0, 6 * np.pi, 300)) * 3
    return base + cycle + noise


@pytest.fixture
def ohlcv(rng, trending_up) -> dict[str, np.ndarray]:
    """Full OHLCV arrays built from a price series."""
    c = trending_up
    spread = np.abs(rng.normal(0, 0.15, len(c)))
    h = c + spread + np.abs(rng.normal(0, 0.2, len(c)))
    l = c - spread - np.abs(rng.normal(0, 0.2, len(c)))
    o = c + rng.normal(0, 0.1, len(c))
    v = np.abs(rng.normal(1_000_000, 200_000, len(c)))
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


# ── SharedState fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def state() -> SharedState:
    s = SharedState()
    s.equity = 100_000.0
    return s


@pytest.fixture
def state_with_stock(state, ohlcv) -> SharedState:
    """SharedState pre-loaded with one stock symbol's OHLCV data."""
    state.market.stocks["GME"] = {
        "close":          ohlcv["close"].tolist(),
        "high":           ohlcv["high"].tolist(),
        "low":            ohlcv["low"].tolist(),
        "volume":         ohlcv["volume"].tolist(),
        "short_interest": 0.25,
        "rvol":           3.5,
        "social_zscore":  2.1,
    }
    return state


@pytest.fixture
def state_with_crypto(state, ohlcv) -> SharedState:
    state.market.crypto["BTC/USD"] = {
        "close":  ohlcv["close"].tolist(),
        "high":   ohlcv["high"].tolist(),
        "low":    ohlcv["low"].tolist(),
        "volume": ohlcv["volume"].tolist(),
        "btc_exchange_inflow": 500_000,
        "sol_memecoin_vol_spike": False,
    }
    return state


# ── Order fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def long_order() -> Order:
    return Order(
        symbol       = "GME",
        sector       = "stocks",
        side         = OrderSide.BUY,
        qty          = 10.0,
        entry_price  = 100.0,
        fill_price   = 100.0,
        stop_price   = 94.0,
        initial_stop = 94.0,
        take_profit  = 109.0,
        atr          = 3.0,
        status       = OrderStatus.FILLED,
        filled_at    = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def short_order() -> Order:
    return Order(
        symbol       = "GME",
        sector       = "stocks",
        side         = OrderSide.SELL,
        qty          = 10.0,
        entry_price  = 100.0,
        fill_price   = 100.0,
        stop_price   = 106.0,
        initial_stop = 106.0,
        take_profit  = 91.0,
        atr          = 3.0,
        status       = OrderStatus.FILLED,
        filled_at    = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


# ── RiskParameters fixture ────────────────────────────────────────────────────

@pytest.fixture
def risk_params() -> RiskParameters:
    return RiskParameters(
        symbol        = "GME",
        position_size = 10.0,
        entry_price   = 100.0,
        stop_price    = 94.0,
        take_profit   = 109.0,
        kelly_fraction= 0.12,
        max_loss_usd  = 60.0,
    )


# ── Signal fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def bull_signal() -> TechnicalSignals:
    return TechnicalSignals(
        sector             = "stocks",
        symbol             = "GME",
        direction          = "long",
        confidence         = 0.75,
        sp_score           = 0.72,
        adx                = 32.0,
        rsi_divergence     = False,
        bb_squeeze         = True,
        ema_ribbon_aligned = True,
        atr                = 3.0,
        timestamp          = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


# ── Config fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "mode":                  "paper",
        "alpaca_key":            "test_key",
        "alpaca_secret":         "test_secret",
        "openai_api_key":        "sk-test",
        "base_risk_pct":         0.01,
        "kelly_lookback":        50,
        "avg_win_r":             1.5,
        "avg_loss_r":            1.0,
        "min_sp_score":          0.60,
        "btc_inflow_threshold":  1_000_000,
        "sol_memecoin_spike_threshold": 50_000_000,
        "max_open_positions":    5,
        "max_daily_loss_pct":    0.05,
        "max_risk_per_trade":    0.02,
        "allow_extended_hours":  False,
        "trailing_atr_multiple": 2.0,
        "breakeven_at_r":        1.0,
        "lock_profit_at_r":      2.0,
        "daily_loss_limit_pct":  0.05,
        "drawdown_limit_pct":    0.10,
        "loss_streak_limit":     4,
        "news_poll_seconds":     30,
        "strategy_interval_seconds": 15,
        "stocktwits_max_rph":    180,
        "social_spike_threshold": 2.0,
        "commission_per_trade":  1.0,
        "slippage_bps":          {"stocks": 5, "crypto": 10},
    }


# ── BacktestState fixture ─────────────────────────────────────────────────────

@pytest.fixture
def bt_state() -> BacktestState:
    return BacktestState(initial_equity=100_000.0)


# ── Async helper ──────────────────────────────────────────────────────────────

@pytest.fixture
def run_async():
    """Run a coroutine synchronously inside a test."""
    def _run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)
    return _run
