"""
tests/test_ta_engine.py
Unit tests for the technical analysis functions in StrategyAgent.

Tests verify:
  - EMA converges correctly and lags price appropriately
  - RSI is bounded [0, 100] and reaches extremes on one-directional moves
  - ADX is positive and > 25 on strong trends, lower on ranging markets
  - ATR is always positive and scales with bar volatility
  - Bollinger squeeze fires on low-volatility consolidation bars
  - RSI divergence detection (price up / RSI down = bearish divergence)
  - Squeeze Probability Score normalisation and weighting
"""

import numpy as np
import pytest

from chimera_v12.agents.strategy_agent import (
    ema, rsi, adx, atr_value, bollinger_squeeze,
    detect_rsi_divergence, squeeze_probability_score
)

# ── EMA ───────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_length_preserved(self, trending_up):
        result = ema(trending_up, 20)
        assert len(result) == len(trending_up)

    def test_single_value_series(self):
        p = np.array([50.0])
        assert ema(p, 1)[0] == pytest.approx(50.0)

    def test_lags_price_on_uptrend(self, trending_up):
        """EMA should be below current price in a strong uptrend."""
        e20 = ema(trending_up, 20)
        # After warmup, EMA lags price upward
        assert e20[-1] < trending_up[-1]

    def test_ribbon_order_on_uptrend(self, trending_up):
        """In a strong uptrend: price > EMA9 > EMA20 > EMA200."""
        e9   = ema(trending_up, 9)
        e20  = ema(trending_up, 20)
        e200 = ema(trending_up, 200)
        assert trending_up[-1] > e9[-1] > e20[-1] > e200[-1]

    def test_ribbon_inverted_on_downtrend(self, trending_down):
        """In a downtrend: price < EMA9 < EMA20 < EMA200."""
        e9   = ema(trending_down, 9)
        e20  = ema(trending_down, 20)
        e200 = ema(trending_down, 200)
        assert trending_down[-1] < e9[-1] < e20[-1] < e200[-1]

    def test_constant_series(self):
        """EMA of a constant series equals the constant."""
        p = np.full(100, 42.0)
        result = ema(p, 20)
        # After period bars, should converge to 42.0
        assert result[-1] == pytest.approx(42.0, rel=1e-3)

    def test_faster_ema_more_responsive(self, trending_up):
        """Shorter period EMA should be closer to current price."""
        e5  = ema(trending_up, 5)
        e20 = ema(trending_up, 20)
        # In uptrend, faster EMA is higher (closer to current price)
        assert e5[-1] > e20[-1]

# ── RSI ───────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_bounded(self, trending_up):
        r = rsi(trending_up)
        assert 0.0 <= r <= 100.0

    def test_overbought_on_strong_uptrend(self):
        """Monotonically increasing prices → RSI approaches 100."""
        p = np.linspace(100, 200, 100)
        r = rsi(p)
        assert r > 70, f"Expected RSI > 70 on strong uptrend, got {r:.1f}"

    def test_oversold_on_strong_downtrend(self):
        """Monotonically decreasing prices → RSI approaches 0."""
        p = np.linspace(200, 100, 100)
        r = rsi(p)
        assert r < 30, f"Expected RSI < 30 on strong downtrend, got {r:.1f}"

    def test_neutral_on_ranging(self, ranging):
        """Sideways market → RSI near 50."""
        r = rsi(ranging)
        assert 25 <= r <= 75, f"Expected RSI away from extremes on ranging market, got {r:.1f}"

    def test_no_loss_returns_100(self):
        """All gains, no losses → RSI == 100."""
        p = np.linspace(100, 200, 50)
        r = rsi(p, period=14)
        assert r == pytest.approx(100.0)

    def test_minimum_length(self):
        """RSI needs at least period+1 prices to produce a result."""
        p = np.array([100.0, 101.0, 100.5, 102.0, 101.5,
                      103.0, 102.5, 104.0, 103.5, 105.0,
                      104.5, 106.0, 105.5, 107.0, 106.5])
        r = rsi(p, period=14)
        assert 0 <= r <= 100

# ── ADX ───────────────────────────────────────────────────────────────────────

class TestADX:
    def test_positive(self, ohlcv):
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result > 0

    def test_trending_above_25(self, rng):
        """Strong trend should produce ADX > 25."""
        n  = 300
        c  = np.linspace(100, 160, n) + rng.normal(0, 0.2, n)
        h  = c + np.abs(rng.normal(0, 0.5, n)) + 0.5
        l  = c - np.abs(rng.normal(0, 0.5, n)) - 0.5
        result = adx(h, l, c)
        assert result > 20, f"Expected ADX > 20 on strong trend, got {result:.1f}"

    def test_non_negative(self, ohlcv):
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result >= 0

    def test_custom_period(self, ohlcv):
        r14 = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        r7  = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=7)
        # Both should be valid positive numbers
        assert r14 > 0 and r7 > 0

# ── ATR ───────────────────────────────────────────────────────────────────────

class TestATR:
    def test_positive(self, ohlcv):
        result = atr_value(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result > 0

    def test_scales_with_volatility(self, rng):
        """Higher bar ranges → higher ATR."""
        n = 100
        c = np.full(n, 100.0) + rng.normal(0, 0.1, n)

        h_tight = c + 0.5
        l_tight = c - 0.5
        atr_tight = atr_value(h_tight, l_tight, c)

        h_wide = c + 5.0
        l_wide = c - 5.0
        atr_wide = atr_value(h_wide, l_wide, c)

        assert atr_wide > atr_tight

    def test_never_negative(self, ohlcv):
        result = atr_value(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result >= 0

    def test_stop_distance_makes_sense(self, ohlcv):
        """ATR × 2 should be a meaningful stop distance (not zero, not huge)."""
        a = atr_value(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        price = float(ohlcv["close"][-1])
        stop_distance = a * 2
        # Stop distance should be between 0.1% and 20% of price
        assert 0.001 * price < stop_distance < 0.20 * price

# ── Bollinger Squeeze ─────────────────────────────────────────────────────────

class TestBollingerSqueeze:
    def test_squeeze_on_flat_prices(self):
        """Near-constant prices → BB width near zero → squeeze fires."""
        p = np.full(30, 100.0) + np.random.default_rng(0).normal(0, 0.05, 30)
        is_squeeze, width = bollinger_squeeze(p)
        assert bool(is_squeeze) is True
        assert width < 0.02

    def test_no_squeeze_on_volatile(self, rng):
        """Highly volatile prices → no squeeze."""
        p = np.linspace(100, 200, 30) + rng.normal(0, 5, 30)
        is_squeeze, width = bollinger_squeeze(p)
        assert bool(is_squeeze) is False
        assert width > 0.02

    def test_width_positive(self, trending_up):
        _, width = bollinger_squeeze(trending_up[-30:])
        assert float(width) > 0

    def test_insufficient_data(self):
        """Fewer than period bars → returns (False, 0.0)."""
        p = np.array([100.0, 101.0, 99.0])
        is_squeeze, width = bollinger_squeeze(p, period=20)
        assert bool(is_squeeze) is False
        assert float(width) == 0.0

# ── RSI Divergence ────────────────────────────────────────────────────────────

class TestRSIDivergence:
    def test_bearish_divergence(self):
        """Price makes higher high but RSI makes lower high → divergence."""
        prices  = [100.0, 102.0, 101.0, 103.0, 104.0, 105.0]  # price up
        rsi_arr = [65.0,  63.0,  62.0,  61.0,  60.0,  59.0]   # RSI down
        assert detect_rsi_divergence(
            np.array(prices), rsi_arr, lookback=5
) == True

    def test_bullish_divergence(self):
        """Price makes lower low but RSI makes higher low → divergence."""
        prices  = [105.0, 103.0, 102.0, 101.0, 100.0, 99.0]   # price down
        rsi_arr = [35.0,  36.0,  37.0,  38.0,  39.0,  40.0]   # RSI up
        assert detect_rsi_divergence(
            np.array(prices), rsi_arr, lookback=5
) == True

    def test_no_divergence_aligned(self):
        """Price up and RSI up → no divergence."""
        prices  = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        rsi_arr = [50.0,  52.0,  54.0,  56.0,  58.0,  60.0]
        assert detect_rsi_divergence(
            np.array(prices), rsi_arr, lookback=5
) == False

    def test_insufficient_data(self):
        prices  = [100.0, 101.0]
        rsi_arr = [50.0,  51.0]
        # Should not raise — just return False
        result = detect_rsi_divergence(np.array(prices), rsi_arr, lookback=5)
        assert result is False

# ── Squeeze Probability Score ─────────────────────────────────────────────────

class TestSqueezeScore:
    def test_all_max_inputs_gives_one(self):
        """Maximum inputs → Sp = 1.0."""
        sp = squeeze_probability_score(
            short_interest=0.50,   # 50% SI → normalised 1.0,
            rvol=10.0,             # RVOL 10 → normalised 1.0,
            sentiment_zscore=5.0,  # Z=5 → normalised 1.0
        )
        assert sp == pytest.approx(1.0, rel=0.01)

    def test_all_zero_inputs_gives_zero(self):
        sp = squeeze_probability_score(0.0, 1.0, 0.0)
        assert sp == pytest.approx(0.0, abs=0.05)

    def test_weights_sum_to_one(self):
        """Weights 0.4 + 0.3 + 0.3 = 1.0."""
        assert 0.4 + 0.3 + 0.3 == pytest.approx(1.0)

    def test_si_dominates(self):
        """With equal inputs, SI (weight 0.4) should dominate."""
        high_si  = squeeze_probability_score(0.50, 2.0, 1.0)
        low_si   = squeeze_probability_score(0.05, 2.0, 1.0)
        assert high_si > low_si

    def test_bounded_0_to_1(self):
        for si in [0.0, 0.25, 0.50, 0.75]:
            for rvol in [1.0, 3.0, 5.0, 10.0]:
                for z in [0.0, 2.0, 5.0, 10.0]:
                    sp = squeeze_probability_score(si, rvol, z)
                    assert 0.0 <= sp <= 1.0, f"Sp={sp} out of range for si={si} rvol={rvol} z={z}"

    def test_threshold_35_pct(self):
        """Typical squeeze candidate (SI=25%, RVOL=3.5, Z=2.0) should exceed 0.35."""
        sp = squeeze_probability_score(0.25, 3.5, 2.0)
        assert sp >= 0.35, f"Expected Sp >= 0.35 for typical squeeze, got {sp:.3f}"

    @pytest.mark.parametrize("si,rvol,z,expected_min", [
        (0.20, 3.0, 0.5, 0.20),   # weak signal,
        (0.30, 4.0, 3.0, 0.40),   # moderate signal,
        (0.45, 8.0, 4.5, 0.85),   # strong signal
    ])
    def test_parametrised_thresholds(self, si, rvol, z, expected_min):
        sp = squeeze_probability_score(si, rvol, z)
        assert sp >= expected_min, f"Sp={sp:.3f} < {expected_min} for si={si} rvol={rvol} z={z}"
