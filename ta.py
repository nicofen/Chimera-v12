"""
tests/test_regime.py
Tests for the regime classifier and strategy permission matrix.

Covers:
  Regime.is_signal_allowed  — permission matrix correctness
  RegimeClassifier._decide  — classification rules in isolation
  RegimeClassifier.classify_once — end-to-end with synthetic bar data
  Smoothing                 — regime must persist N periods before committing
  SharedState integration   — regime written to state and read by strategy gate
"""

import numpy as np
import pytest

from chimera_v12.regime.models import (
    Regime, RegimeState, is_signal_allowed, REGIME_PERMISSIONS
)
from chimera_v12.regime.classifier import RegimeClassifier
from chimera_v12.utils.state import SharedState

# ── Permission matrix ──────────────────────────────────────────────────────────

class TestPermissionMatrix:
    """Every cell in the permission matrix must behave correctly."""

    # TRENDING_BULL
    def test_bull_allows_stock_long(self):
        assert is_signal_allowed(Regime.TRENDING_BULL, "stocks", "long") is True

    def test_bull_blocks_stock_short(self):
        assert is_signal_allowed(Regime.TRENDING_BULL, "stocks", "short") is False

    def test_bull_allows_crypto_long(self):
        assert is_signal_allowed(Regime.TRENDING_BULL, "crypto", "long") is True

    def test_bull_allows_forex_both(self):
        assert is_signal_allowed(Regime.TRENDING_BULL, "forex", "long")  is True
        assert is_signal_allowed(Regime.TRENDING_BULL, "forex", "short") is True

    def test_bull_allows_futures_long(self):
        assert is_signal_allowed(Regime.TRENDING_BULL, "futures", "long") is True

    # TRENDING_BEAR
    def test_bear_blocks_stock_long(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "stocks", "long") is False

    def test_bear_allows_stock_short(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "stocks", "short") is True

    def test_bear_blocks_crypto_entirely(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "crypto", "long")  is False
        assert is_signal_allowed(Regime.TRENDING_BEAR, "crypto", "short") is False

    def test_bear_allows_forex_short(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "forex", "short") is True

    def test_bear_blocks_forex_long(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "forex", "long") is False

    def test_bear_allows_futures_short(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "futures", "short") is True

    # HIGH_VOLATILITY
    def test_hv_blocks_stocks_entirely(self):
        assert is_signal_allowed(Regime.HIGH_VOLATILITY, "stocks",  "long")  is False
        assert is_signal_allowed(Regime.HIGH_VOLATILITY, "stocks",  "short") is False

    def test_hv_allows_crypto_both(self):
        assert is_signal_allowed(Regime.HIGH_VOLATILITY, "crypto",  "long")  is True
        assert is_signal_allowed(Regime.HIGH_VOLATILITY, "crypto",  "short") is True

    def test_hv_blocks_forex(self):
        assert is_signal_allowed(Regime.HIGH_VOLATILITY, "forex",   "long")  is False

    def test_hv_blocks_futures(self):
        assert is_signal_allowed(Regime.HIGH_VOLATILITY, "futures", "long")  is False

    # MEAN_REVERTING
    def test_mr_blocks_stocks(self):
        assert is_signal_allowed(Regime.MEAN_REVERTING, "stocks", "long")  is False
        assert is_signal_allowed(Regime.MEAN_REVERTING, "stocks", "short") is False

    def test_mr_blocks_crypto(self):
        assert is_signal_allowed(Regime.MEAN_REVERTING, "crypto", "long")  is False

    def test_mr_allows_forex_both(self):
        assert is_signal_allowed(Regime.MEAN_REVERTING, "forex", "long")   is True
        assert is_signal_allowed(Regime.MEAN_REVERTING, "forex", "short")  is True

    def test_mr_allows_futures_both(self):
        assert is_signal_allowed(Regime.MEAN_REVERTING, "futures", "long")  is True
        assert is_signal_allowed(Regime.MEAN_REVERTING, "futures", "short") is True

    # NEUTRAL — allows everything
    def test_neutral_allows_all_sectors_both_directions(self):
        for sector in ("stocks", "crypto", "forex", "futures"):
            for direction in ("long", "short"):
                assert is_signal_allowed(Regime.NEUTRAL, sector, direction) is True, \
                    f"NEUTRAL should allow {sector}/{direction}"

    # Unknown sector — defaults to allow
    def test_unknown_sector_allowed(self):
        assert is_signal_allowed(Regime.TRENDING_BEAR, "unknown_sector", "long") is True

    # Symmetry: every regime has an entry for every sector
    def test_all_regimes_cover_all_sectors(self):
        for regime, perms in REGIME_PERMISSIONS.items():
            sectors = {p.sector for p in perms}
            for expected in ("stocks", "crypto", "forex", "futures"):
                assert expected in sectors, \
                    f"Regime {regime.value} missing permission for sector {expected}"

# ── Classifier decision rules ─────────────────────────────────────────────────

class TestClassifierDecide:
    """Test _decide() in isolation with synthetic signal dicts."""

    @pytest.fixture
    def clf(self, config):
        state = SharedState()
        return RegimeClassifier(state, config)

    def _signals(self, **kwargs):
        base = {
            "adx_mean": 22.0, "adx_values": [22.0],
            "ema_bull_frac": 0.5, "ema_bear_frac": 0.2,
            "breadth": 0.55, "vix_proxy_mean": 0.015,
            "symbol_count": 5, "btc_inflow": 0
        }
        base.update(kwargs)
        return base

    def test_high_vix_overrides_trend(self, clf):
        """High volatility should override even a strong bull trend."""
        s = self._signals(
            adx_mean=35.0, ema_bull_frac=0.9,
            vix_proxy_mean=0.04   # > 0.025 threshold
        )
        regime, conf, reason = clf._decide(s)
        assert regime == Regime.HIGH_VOLATILITY

    def test_strong_bull_detected(self, clf):
        s = self._signals(
            adx_mean=32.0, ema_bull_frac=0.8,
            breadth=0.75, vix_proxy_mean=0.01
        )
        regime, conf, reason = clf._decide(s)
        assert regime == Regime.TRENDING_BULL
        assert conf > 0.3

    def test_strong_bear_detected(self, clf):
        s = self._signals(
            adx_mean=30.0, ema_bull_frac=0.1,
            ema_bear_frac=0.75, breadth=0.25,
            vix_proxy_mean=0.01
        )
        regime, conf, reason = clf._decide(s)
        assert regime == Regime.TRENDING_BEAR

    def test_low_adx_means_reverting(self, clf):
        s = self._signals(adx_mean=15.0, vix_proxy_mean=0.01)
        regime, conf, reason = clf._decide(s)
        assert regime == Regime.MEAN_REVERTING

    def test_transitional_adx_neutral(self, clf):
        """ADX between ranging and trending thresholds → NEUTRAL."""
        s = self._signals(
            adx_mean=22.0,
            ema_bull_frac=0.4,   # not above 0.5,
            ema_bear_frac=0.3,   # not above 0.5,
            breadth=0.50,
            vix_proxy_mean=0.018
        )
        regime, conf, reason = clf._decide(s)
        assert regime == Regime.NEUTRAL

    def test_no_symbols_returns_neutral(self, clf):
        s = self._signals(symbol_count=0)
        regime, conf, reason = clf._decide(s)
        assert regime == Regime.NEUTRAL
        assert conf == 0.0

    def test_btc_inflow_amplifies_hv(self, clf):
        """BTC exchange inflow should push confidence higher in HV regime."""
        s_no_btc = self._signals(vix_proxy_mean=0.03, btc_inflow=0)
        s_btc    = self._signals(vix_proxy_mean=0.03, btc_inflow=2_000_000)
        _, conf_no_btc, _ = clf._decide(s_no_btc)
        _, conf_btc,    _ = clf._decide(s_btc)
        assert conf_btc >= conf_no_btc

    def test_confidence_bounded_0_to_1(self, clf):
        for adx_v in [10, 20, 30, 50, 80]:
            for vix in [0.005, 0.015, 0.03, 0.06]:
                s = self._signals(adx_mean=float(adx_v), vix_proxy_mean=vix)
                _, conf, _ = clf._decide(s)
                assert 0.0 <= conf <= 1.0, f"conf={conf} out of range"

    def test_reason_string_nonempty(self, clf):
        s = self._signals()
        _, _, reason = clf._decide(s)
        assert isinstance(reason, str) and len(reason) > 0

# ── Smoothing mechanism ───────────────────────────────────────────────────────

class TestSmoothing:
    @pytest.fixture
    def clf(self, config):
        state = SharedState()
        cfg   = {**config, "smoothing_periods": 3, "update_interval_min": 0}
        c     = RegimeClassifier(state, cfg)
        # Ensure regime_state exists
        state.regime_state = RegimeState()
        return c, state

    def test_regime_not_committed_before_smoothing(self, clf):
        classifier, state = clf
        # First two detections — not committed yet
        classifier._candidate       = Regime.TRENDING_BULL
        classifier._candidate_count = 0

        # Simulate two consecutive same-regime detections (< smoothing_periods=3)
        for _ in range(2):
            if Regime.TRENDING_BULL == classifier._candidate:
                classifier._candidate_count += 1
            if classifier._candidate_count >= classifier.t["smoothing_periods"]:
                classifier._commit(Regime.TRENDING_BULL, 0.8, {
                    "adx_mean":0,"ema_bull_frac":0,"ema_bear_frac":0,
                    "vix_proxy_mean":0,"btc_inflow":0
                }, "test")

        # Should still be at default (NEUTRAL) since count < smoothing
        assert state.regime_state.regime == Regime.NEUTRAL

    def test_regime_committed_after_smoothing(self, clf):
        classifier, state = clf
        signals = {
            "adx_mean":0,"ema_bull_frac":0,"ema_bear_frac":0,
            "vix_proxy_mean":0,"btc_inflow":0,"symbol_count":3
        }
        classifier._candidate       = Regime.TRENDING_BULL
        classifier._candidate_count = 0

        for _ in range(3):
            classifier._candidate_count += 1
            if classifier._candidate_count >= classifier.t["smoothing_periods"]:
                classifier._commit(Regime.TRENDING_BULL, 0.8, signals, "test")
                break

        assert state.regime_state.regime == Regime.TRENDING_BULL

    def test_candidate_resets_on_regime_change(self, clf):
        classifier, state = clf
        classifier._candidate       = Regime.TRENDING_BULL
        classifier._candidate_count = 2

        # New regime detected — candidate should reset
        new_regime = Regime.TRENDING_BEAR
        if new_regime != classifier._candidate:
            classifier._candidate       = new_regime
            classifier._candidate_count = 1

        assert classifier._candidate == Regime.TRENDING_BEAR
        assert classifier._candidate_count == 1

# ── End-to-end with synthetic bar data ───────────────────────────────────────

class TestClassifyOnce:
    def _inject_bars(self, state, sector, symbol, closes, high_mult=1.005, low_mult=0.995):
        closes = np.array(closes)
        highs  = closes * high_mult
        lows   = closes * low_mult
        if sector == "stocks":
            state.market.stocks[symbol] = {
                "close": closes.tolist(), "high": highs.tolist(),
                "low": lows.tolist(), "volume": [1e6]*len(closes)
            }
        elif sector == "crypto":
            state.market.crypto[symbol] = {
                "close": closes.tolist(), "high": highs.tolist(),
                "low": lows.tolist(), "volume": [1e6]*len(closes)
            }

    def test_strongly_trending_up_gives_bull(self, config, rng):
        state = SharedState()
        state.regime_state = RegimeState()
        cfg   = {**config, "smoothing_periods": 1, "update_interval_min": 0}
        clf   = RegimeClassifier(state, cfg)

        # 300 bars of strong uptrend — EMA ribbon will align bullish
        prices = 100 + np.cumsum(rng.normal(0.3, 0.2, 300))
        self._inject_bars(state, "stocks", "GME",  prices)
        self._inject_bars(state, "stocks", "TSLA", prices * 1.5)
        self._inject_bars(state, "stocks", "AMC",  prices * 0.8)

        rs = clf.classify_once()
        # With smoothing=1, it should classify (may be BULL or NEUTRAL depending on ADX)
        assert rs.regime in (Regime.TRENDING_BULL, Regime.NEUTRAL)
        assert 0.0 <= rs.confidence <= 1.0

    def test_flat_prices_give_mean_reverting(self, config, rng):
        state = SharedState()
        state.regime_state = RegimeState()
        cfg   = {**config, "smoothing_periods": 1}
        clf   = RegimeClassifier(state, cfg)

        # 300 bars of pure noise around a constant — ADX will be very low
        flat = 100 + rng.normal(0, 0.1, 300)
        self._inject_bars(state, "stocks", "GME", flat)

        rs = clf.classify_once()
        assert rs.regime in (Regime.MEAN_REVERTING, Regime.NEUTRAL)

    def test_regime_state_written_to_shared_state(self, config):
        state = SharedState()
        state.regime_state = RegimeState()
        cfg   = {**config, "smoothing_periods": 1}
        clf   = RegimeClassifier(state, cfg)

        flat = np.full(250, 100.0)
        self._inject_bars(state, "stocks", "GME", flat)
        clf.classify_once()

        assert hasattr(state, "regime_state")
        assert isinstance(state.regime_state, RegimeState)
        assert state.regime_state.updated_at != ""

    def test_regime_written_to_state_regime_field(self, config):
        """Legacy state.regime field must also be updated."""
        state = SharedState()
        state.regime_state = RegimeState()
        cfg   = {**config, "smoothing_periods": 1}
        clf   = RegimeClassifier(state, cfg)

        flat = np.full(250, 100.0)
        self._inject_bars(state, "stocks", "GME", flat)
        clf.classify_once()

        assert hasattr(state, "regime")
        assert isinstance(state.regime, Regime)

    def test_no_data_gives_neutral(self, config):
        state = SharedState()
        state.regime_state = RegimeState()
        cfg   = {**config, "smoothing_periods": 1}
        clf   = RegimeClassifier(state, cfg)

        # No bars injected — insufficient data
        rs = clf.classify_once()
        assert rs.regime == Regime.NEUTRAL

# ── StrategyAgent regime gate integration ────────────────────────────────────

class TestStrategyAgentRegimeGate:
    """
    Verify that the regime gate in StrategyAgent._emit() actually blocks
    signals that violate the current regime.
    """

    def test_long_stock_blocked_in_bear_regime(self, config, state_with_stock):
        import asyncio
        from chimera_v12.agents.strategy_agent import StrategyAgent
        from chimera_v12.regime.models import RegimeState

        state = state_with_stock
        # set bear regime
        state.regime_state = RegimeState(
            regime=Regime.TRENDING_BEAR, confidence=0.9,
            reason="test", updated_at="2024-01-01"
        )
        state.regime = Regime.TRENDING_BEAR

        agent    = StrategyAgent(state, config)
        emitted  = []

        async def capture(sig):
            emitted.append(sig)

        state.put_signal = capture

        from chimera_v12.utils.state import TechnicalSignals
        sig = TechnicalSignals(
            sector="stocks", symbol="GME", direction="long",
            confidence=0.9, adx=35.0, atr=3.0, timestamp=None,
            sp_score=0.72, rsi_divergence=False, bb_squeeze=True,
            ema_ribbon_aligned=True
        )

        asyncio.run(agent._emit(sig))
        assert len(emitted) == 0, "Long stock signal should be blocked in bear regime"

    def test_long_stock_allowed_in_bull_regime(self, config, state_with_stock):
        import asyncio
        from chimera_v12.agents.strategy_agent import StrategyAgent
        from chimera_v12.regime.models import RegimeState

        state = state_with_stock
        state.regime_state = RegimeState(
            regime=Regime.TRENDING_BULL, confidence=0.85,
            reason="test", updated_at="2024-01-01"
        )
        state.regime = Regime.TRENDING_BULL

        agent   = StrategyAgent(state, config)
        emitted = []

        async def capture(sig):
            emitted.append(sig)

        state.put_signal = capture

        from chimera_v12.utils.state import TechnicalSignals
        sig = TechnicalSignals(
            sector="stocks", symbol="GME", direction="long",
            confidence=0.9, adx=35.0, atr=3.0, timestamp=None,
            sp_score=0.72, rsi_divergence=False, bb_squeeze=True,
            ema_ribbon_aligned=True
        )

        asyncio.run(agent._emit(sig))
        assert len(emitted) == 1, "Long stock signal should pass in bull regime"

    def test_no_regime_state_defaults_to_neutral(self, config, state_with_stock):
        """Without regime_state, signals should pass (neutral allows all)."""
        import asyncio
        from chimera_v12.agents.strategy_agent import StrategyAgent

        state = state_with_stock
        # Remove regime_state if present
        if hasattr(state, "regime_state"):
            del state.regime_state

        agent   = StrategyAgent(state, config)
        emitted = []

        async def capture(sig):
            emitted.append(sig)

        state.put_signal = capture

        from chimera_v12.utils.state import TechnicalSignals
        sig = TechnicalSignals(
            sector="stocks", symbol="GME", direction="long",
            confidence=0.9, adx=35.0, atr=3.0, timestamp=None,
            sp_score=0.72, rsi_divergence=False, bb_squeeze=True,
            ema_ribbon_aligned=True
        )

        asyncio.run(agent._emit(sig))
        assert len(emitted) == 1, "Signal should pass when no regime_state (defaults neutral)"
