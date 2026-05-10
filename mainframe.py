"""
chimera_v12/utils/ta.py
Vectorised technical analysis helpers — numpy-only, no TA-Lib dependency.
All functions return Python native types (not numpy scalars) for 3.13 compat.
"""
from __future__ import annotations

import builtins
import numpy as np


def _pyb(x) -> builtins.bool:
    """Convert any numpy boolean to a guaranteed Python bool."""
    return builtins.bool(builtins.int(x))


# ── Trend ─────────────────────────────────────────────────────────────────────

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — returns full array."""
    alpha = 2.0 / (period + 1)
    out   = np.zeros_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI — returns current RSI as Python float."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices.astype(float))
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains[-period:]))
    avg_l  = float(np.mean(losses[-period:]))
    if avg_l == 0:
        return 100.0
    return round(100.0 - 100.0 / (1 + avg_g / avg_l), 4)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average Directional Index — returns Python float."""
    if len(close) < period * 2:
        return 0.0
    tr_arr, pdm_arr, ndm_arr = [], [], []
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    for i in range(1, len(c)):
        tr    = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        pdm_v = max(h[i] - h[i-1], 0.0)
        ndm_v = max(l[i-1] - l[i], 0.0)
        if pdm_v >= ndm_v:
            ndm_v = 0.0
        else:
            pdm_v = 0.0
        tr_arr.append(tr)
        pdm_arr.append(pdm_v)
        ndm_arr.append(ndm_v)
    tr_a  = np.array(tr_arr[-period * 2:])
    pdm_a = np.array(pdm_arr[-period * 2:])
    ndm_a = np.array(ndm_arr[-period * 2:])
    atr_v = float(np.mean(tr_a[-period:]))
    if atr_v == 0:
        return 0.0
    pdi = 100 * float(np.mean(pdm_a[-period:])) / atr_v
    ndi = 100 * float(np.mean(ndm_a[-period:])) / atr_v
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0
    return round(dx, 2)


def atr_value(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average True Range — returns Python float."""
    if len(close) < period + 1:
        return 0.0
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    trs = [
        max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        for i in range(1, len(c))
    ]
    return round(float(np.mean(trs[-period:])), 4)


# ── Volatility / Squeeze ──────────────────────────────────────────────────────

def bollinger_squeeze(prices: np.ndarray, period: int = 20, mult: float = 2.0) -> tuple:
    """
    Returns (is_squeeze: bool, bb_width_pct: float).

    is_squeeze: True when Bollinger Band width < 2% of price (tight coil)
                OR when BB width < Keltner Channel width (classic squeeze).
    bb_width_pct: percentage band width = (2×std) / mid — useful for monitoring.

    Note: returns Python builtins.bool, not numpy.bool_, for 3.13/numpy-2 compat.
    """
    if len(prices) < period:
        return builtins.bool(False), 0.0
    window       = prices[-period:].astype(float)
    std          = float(np.std(window))
    mid          = float(np.mean(window))
    bb_width_pct = (2.0 * std) / (mid + 1e-9)
    kc           = mult * float(np.mean(np.abs(np.diff(window))))
    bb_abs       = mult * std
    # Use Python float comparisons to guarantee Python bool result
    cond_a = builtins.bool(bb_width_pct < 0.02)
    cond_b = builtins.bool(bb_abs < kc)
    return builtins.bool(cond_a or cond_b), round(bb_width_pct, 6)


def detect_rsi_divergence(prices: np.ndarray, rsi_arr, lookback: int = 5) -> builtins.bool:
    """
    Returns True when price and RSI diverge (move in opposite directions).
    Returns Python bool (not numpy.bool_) for 3.13/numpy-2 compat.
    """
    closes_l  = [float(x) for x in prices]
    rsi_arr_l = [float(x) for x in rsi_arr]
    if len(closes_l) < lookback + 1 or len(rsi_arr_l) < lookback + 1:
        return builtins.bool(False)
    price_up = closes_l[-1] > closes_l[-lookback]
    rsi_up   = rsi_arr_l[-1] > rsi_arr_l[-lookback]
    return builtins.bool(price_up != rsi_up)


def detect_rsi_divergence_direction(prices: np.ndarray, rsi_arr, lookback: int = 14) -> str:
    """Returns 'bull', 'bear', or 'none'. Used in v12 orchestrator logic."""
    closes_l  = [float(x) for x in prices]
    rsi_arr_l = [float(x) for x in rsi_arr]
    if len(closes_l) < lookback or len(rsi_arr_l) < lookback:
        return "none"
    price_trend = closes_l[-1] - closes_l[-lookback]
    rsi_trend   = rsi_arr_l[-1] - rsi_arr_l[-lookback]
    if price_trend < 0 and rsi_trend > 0:
        return "bull"
    if price_trend > 0 and rsi_trend < 0:
        return "bear"
    return "none"


# ── Volume / Market Profile ───────────────────────────────────────────────────

def vwap(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray,
) -> float:
    """Volume-Weighted Average Price."""
    typical = (highs.astype(float) + lows.astype(float) + closes.astype(float)) / 3.0
    total_v = float(np.sum(volumes))
    if total_v == 0:
        return float(closes[-1])
    return float(np.sum(typical * volumes.astype(float)) / total_v)


def anchored_vwap(
    highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, volumes: np.ndarray,
    anchor: int,
) -> float:
    """Anchored VWAP from a specific bar index."""
    return vwap(highs[anchor:], lows[anchor:], closes[anchor:], volumes[anchor:])


def volume_profile(closes: np.ndarray, volumes: np.ndarray, bins: int = 20) -> dict:
    """
    Compute volume profile. Returns Point of Control (POC),
    Value Area High (VAH), Value Area Low (VAL) — covering 70% of volume.
    """
    if len(closes) < 2:
        return {"poc": float(closes[-1]) if len(closes) else 0.0, "vah": 0.0, "val": 0.0}
    counts, edges = np.histogram(closes.astype(float), bins=bins, weights=volumes.astype(float))
    poc_idx = int(np.argmax(counts))
    poc     = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    total   = float(np.sum(counts))
    target  = total * 0.70
    cumvol  = 0.0
    lo_idx, hi_idx = poc_idx, poc_idx
    while cumvol < target and (lo_idx > 0 or hi_idx < len(counts) - 1):
        lo_add = float(counts[lo_idx - 1]) if lo_idx > 0 else 0.0
        hi_add = float(counts[hi_idx + 1]) if hi_idx < len(counts) - 1 else 0.0
        if lo_add >= hi_add and lo_idx > 0:
            lo_idx -= 1
            cumvol += lo_add
        elif hi_idx < len(counts) - 1:
            hi_idx += 1
            cumvol += hi_add
        else:
            break
    return {
        "poc": poc,
        "vah": float((edges[hi_idx] + edges[hi_idx + 1]) / 2),
        "val": float((edges[lo_idx] + edges[lo_idx + 1]) / 2),
    }
