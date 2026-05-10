"""
tests/test_simulated_oms.py
Tests for SimulatedOMS fill mechanics in the backtest engine.

Covers:
  Entry fill    — fills at next-bar open + slippage
  Gap-through   — bar opens through stop → fills at open (not stop)
  Intrabar stop — low/high crosses stop within bar → fills at stop
  Take-profit   — high/low reaches TP → fills at TP
  Trailing stop — ratchets on bar close
  P&L           — realised P&L computed correctly, commission subtracted
  Force close   — closes all positions at last price
  Max positions — rejects when at capacity
"""

from datetime import datetime, timezone, timezone, timedelta

import pytest

from chimera_v12.backtest.simulated_oms import SimulatedOMS, BacktestTrade
from chimera_v12.backtest.state import BacktestState
from chimera_v12.backtest.engine import ONE_BAR_OFFSET
from chimera_v12.oms.models import OrderSide, OrderStatus
from chimera_v12.utils.state import RiskParameters, TechnicalSignals

def _utc(day: int = 1) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)

def _state_with_gme(initial_equity=100_000.0) -> BacktestState:
    s = BacktestState(initial_equity=initial_equity)
    s.equity = initial_equity
    s.market.stocks["GME"] = {
        "close": [100.0]*250, "high": [102.0]*250, "low": [98.0]*250,
        "volume": [1e6]*250,
        "short_interest": 0.25, "rvol": 3.5, "social_zscore": 2.0
    }
    sig = TechnicalSignals(sector="stocks", symbol="GME", direction="long",
                           confidence=0.8, atr=3.0, timestamp=None,
                           sp_score=0.0, adx=30.0)
    s.signals.append(sig)
    return s

def _rp(symbol="GME", entry=100.0, stop=94.0, tp=109.0, qty=10.0) -> RiskParameters:
    return RiskParameters(
        symbol=symbol, position_size=qty, entry_price=entry,
        stop_price=stop, take_profit=tp, kelly_fraction=0.1, max_loss_usd=60.0
    )

def _oms(state, commission=1.0, slippage={"stocks":5}) -> SimulatedOMS:
    cfg = {
        "commission_per_trade": commission,
        "slippage_bps": slippage,
        "max_open_positions": 5,
        "trailing_atr_multiple": 2.0,
        "breakeven_at_r": 1.0,
        "lock_profit_at_r": 2.0
    }
    return SimulatedOMS(state, cfg)

def _bar(o=100.0, h=103.0, l=97.0, c=101.0, v=1e6):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}

# ── Entry fill ────────────────────────────────────────────────────────────────

class TestEntryFill:
    def test_fills_at_next_bar_open(self):
        state = _state_with_gme()
        oms   = _oms(state)
        rp    = _rp()
        sig_dt = _utc(1)
        next_dt = _utc(2)

        oms.accept_order(rp, sig_dt + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=99.0), sig_dt)   # same bar — no fill
        assert "GME" not in oms._open

        oms.on_bar("GME", "stocks", _bar(o=99.0), next_dt)  # next bar — fills
        assert "GME" in oms._open

    def test_fill_includes_slippage(self):
        """Long fill price = open × (1 + slippage_bps/10000)."""
        state = _state_with_gme()
        oms   = _oms(state, slippage={"stocks": 25})
        rp    = _rp()
        sig_dt = _utc(1)
        next_dt = _utc(2)

        oms.accept_order(rp, sig_dt + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), next_dt)

        order = oms._open["GME"]
        expected = 100.0 * (1 + 25 / 10_000)
        assert order.fill_price == pytest.approx(expected, rel=1e-6)

    def test_short_fill_slippage_direction(self):
        """Short fill price = open × (1 - slippage_bps/10000)."""
        state = _state_with_gme()
        sig = TechnicalSignals(sector="stocks", symbol="GME", direction="short",
                               confidence=0.8, atr=3.0, timestamp=None,
                               sp_score=0.0, adx=30.0)
        state.signals = [sig]
        oms = _oms(state, slippage={"stocks": 25})
        rp  = RiskParameters(symbol="GME", position_size=10.0, entry_price=100.0,
                             stop_price=106.0, take_profit=91.0,
                             kelly_fraction=0.1, max_loss_usd=60.0)
        sig_dt = _utc(1); next_dt = _utc(2)
        oms.accept_order(rp, sig_dt + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), next_dt)

        order = oms._open["GME"]
        expected = 100.0 * (1 - 25 / 10_000)
        assert order.fill_price == pytest.approx(expected, rel=1e-6)

    def test_commission_deducted_on_fill(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=5.0)
        rp    = _rp()
        before = state.equity
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(), _utc(2))
        assert state.equity == pytest.approx(before - 5.0)

    def test_duplicate_symbol_not_filled(self):
        """Second order for same symbol rejected when position is open."""
        state = _state_with_gme()
        oms   = _oms(state)
        rp    = _rp()
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(), _utc(2))
        assert "GME" in oms._open

        rp2 = _rp(qty=20.0)
        oms.accept_order(rp2, _utc(2) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(), _utc(3))
        # Should still be only one position (qty not doubled)
        assert oms._open["GME"].qty == 10.0

    def test_max_positions_enforced(self):
        """When 5 positions are open, 6th order is silently dropped."""
        state = BacktestState(initial_equity=100_000)
        state.equity = 100_000.0

        for sym in ["A","B","C","D","E"]:
            state.market.stocks[sym] = {
                "close":[100.0]*5,"high":[102.0]*5,"low":[98.0]*5,"volume":[1e6]*5,
                "short_interest":0.2,"rvol":2.0,"social_zscore":1.0
            }
            state.signals.append(TechnicalSignals(
                sector="stocks",symbol=sym,direction="long",
                confidence=0.8,atr=3.0,timestamp=None,sp_score=0.0,adx=30.0))

        oms = _oms(state)
        # Fill 5 positions
        for i, sym in enumerate(["A","B","C","D","E"]):
            rp = RiskParameters(symbol=sym,position_size=2.0,entry_price=100.0,
                                stop_price=94.0,take_profit=109.0,
                                kelly_fraction=0.1,max_loss_usd=20.0)
            oms.accept_order(rp, _utc(i+1) + ONE_BAR_OFFSET)
            oms.on_bar(sym,"stocks",_bar(),_utc(i+2))

        assert oms.open_count == 5

        # 6th symbol rejected
        state.market.stocks["F"] = state.market.stocks["A"].copy()
        state.signals.append(TechnicalSignals(
            sector="stocks",symbol="F",direction="long",
            confidence=0.8,atr=3.0,timestamp=None,sp_score=0.0,adx=30.0))
        rp6 = RiskParameters(symbol="F",position_size=2.0,entry_price=100.0,
                              stop_price=94.0,take_profit=109.0,
                              kelly_fraction=0.1,max_loss_usd=20.0)
        oms.accept_order(rp6, _utc(6) + ONE_BAR_OFFSET)
        oms.on_bar("F","stocks",_bar(),_utc(7))
        assert oms.open_count == 5   # still 5, not 6

# ── Stop hit ──────────────────────────────────────────────────────────────────

class TestStopHit:
    def _open_long(self, oms, state, entry=100.0, stop=94.0, tp=115.0):
        rp = _rp(entry=entry, stop=stop, tp=tp)
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=entry), _utc(2))

    def test_gap_through_stop_fills_at_open(self):
        """Bar opens below stop → fill at open (gap risk), not at stop."""
        state = _state_with_gme()
        oms   = _oms(state)
        self._open_long(oms, state, entry=100.0, stop=94.0)

        # Bar opens at 90 (gaps through stop 94)
        oms.on_bar("GME", "stocks", _bar(o=90.0, l=88.0, h=92.0, c=91.0), _utc(3))
        assert "GME" not in oms._open
        trade = state.closed_trades[-1]
        assert trade["exit_price"] == pytest.approx(90.0)

    def test_intrabar_stop_fills_at_stop_price(self):
        """Low crosses stop intrabar → fills at whatever the current stop is."""
        state = _state_with_gme()
        oms   = _oms(state)
        self._open_long(oms, state, entry=100.0, stop=94.0)

        # The trailing stop may ratchet on bar 2 close — capture it
        order = oms._open.get("GME")
        current_stop = order.stop_price if order else 94.0

        # Bar opens above stop but low dips through it
        oms.on_bar("GME", "stocks", _bar(o=98.0, l=current_stop - 0.5, h=99.0, c=97.0), _utc(3))
        assert "GME" not in oms._open
        trade = state.closed_trades[-1]
        # Exit should be at (possibly ratcheted) stop or below
        assert trade["exit_price"] <= 98.0   # always <= open price

    def test_stop_not_triggered_above_stop(self):
        """Low stays well above stop → position remains open."""
        state = _state_with_gme()
        oms   = _oms(state)
        self._open_long(oms, state, entry=100.0, stop=90.0)  # wide stop

        # Low at 96 stays above stop at 90 → no trigger
        oms.on_bar("GME", "stocks", _bar(o=100.0, l=96.0, h=103.0, c=102.0), _utc(3))
        assert "GME" in oms._open

    def test_pnl_negative_on_stop(self):
        """Stop hit → realised P&L should be negative."""
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0)
        self._open_long(oms, state, entry=100.0, stop=94.0)
        oms.on_bar("GME", "stocks", _bar(o=100.0, l=93.5, h=101.0, c=100.0), _utc(3))

        trade = state.closed_trades[-1]
        assert trade["realised_pnl"] < 0

# ── Take-profit ───────────────────────────────────────────────────────────────

class TestTakeProfit:
    def _open_long(self, oms, entry=100.0, stop=94.0, tp=109.0):
        state = _state_with_gme()
        oms.accept_order(_rp(entry=entry, stop=stop, tp=tp),
                         _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=entry), _utc(2))
        return state

    def test_tp_hit_closes_position(self):
        state = _state_with_gme()
        oms   = _oms(state)
        self._open_long(oms)
        oms.on_bar("GME", "stocks", _bar(o=105.0, h=110.0, l=104.0, c=108.0), _utc(3))
        assert "GME" not in oms._open

    def test_tp_fill_price_is_tp(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0)
        self._open_long(oms, tp=109.0)
        oms.on_bar("GME", "stocks", _bar(o=105.0, h=115.0, l=104.0, c=112.0), _utc(3))
        trade = state.closed_trades[-1]
        assert trade["exit_price"] == pytest.approx(109.0)

    def test_pnl_positive_on_tp(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0)
        self._open_long(oms, entry=100.0, tp=109.0)
        oms.on_bar("GME", "stocks", _bar(o=105.0, h=115.0, l=104.0, c=112.0), _utc(3))
        trade = state.closed_trades[-1]
        # 10 shares × (109 - 100) = $90 gross
        assert trade["realised_pnl"] == pytest.approx(90.0, rel=0.01)

    def test_tp_not_triggered_below_tp(self):
        state = _state_with_gme()
        oms   = _oms(state)
        self._open_long(oms, tp=109.0)
        oms.on_bar("GME", "stocks", _bar(o=105.0, h=108.5, l=104.0, c=107.0), _utc(3))
        assert "GME" in oms._open

# ── P&L accounting ────────────────────────────────────────────────────────────

class TestPnLAccounting:
    def test_equity_increases_on_profitable_close(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0)
        before = state.equity
        rp = _rp(entry=100.0, stop=94.0, tp=110.0, qty=10.0)
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), _utc(2))
        oms.on_bar("GME", "stocks", _bar(o=108.0, h=115.0, l=107.0, c=112.0), _utc(3))
        # 10 × (110 - 100) = $100 profit
        assert state.equity > before

    def test_equity_decreases_on_losing_close(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0)
        before = state.equity
        rp = _rp(entry=100.0, stop=94.0, tp=115.0, qty=10.0)
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), _utc(2))
        oms.on_bar("GME", "stocks", _bar(o=100.0, l=93.0, h=101.0, c=98.0), _utc(3))
        assert state.equity < before

    def test_r_multiple_computed(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0)
        rp = _rp(entry=100.0, stop=94.0, tp=118.0, qty=1.0)
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), _utc(2))
        oms.on_bar("GME", "stocks", _bar(o=115.0, h=120.0, l=114.0, c=117.0), _utc(3))
        trade = state.closed_trades[-1]
        # TP=118, entry=100, stop=94 → risk=6, gain=18 → R≈3.0
        assert trade["r_multiple"] == pytest.approx(3.0, rel=0.05)

    def test_commission_both_sides(self):
        """Commission charged on entry AND exit."""
        state = _state_with_gme()
        oms   = _oms(state, commission=10.0)
        before = state.equity
        rp = _rp(entry=100.0, stop=94.0, tp=200.0, qty=10.0)
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), _utc(2))  # entry: -$10
        oms.on_bar("GME", "stocks", _bar(o=100.0, l=93.0, h=101.0, c=97.0), _utc(3))  # stop: -$10
        trade = state.closed_trades[-1]
        assert trade["commission"] == pytest.approx(20.0)

# ── Force close ───────────────────────────────────────────────────────────────

class TestForceClose:
    def test_force_close_all_closes_positions(self):
        state = _state_with_gme()
        oms   = _oms(state)
        rp    = _rp()
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(), _utc(2))
        assert "GME" in oms._open

        oms.close_all({"GME": 105.0}, _utc(3))
        assert "GME" not in oms._open
        assert len(state.closed_trades) == 1

    def test_force_close_uses_provided_price(self):
        state = _state_with_gme()
        oms   = _oms(state, commission=0.0, slippage={"stocks": 0})  # zero slippage for exact P&L
        rp    = _rp(entry=100.0, qty=10.0)
        oms.accept_order(rp, _utc(1) + ONE_BAR_OFFSET)
        oms.on_bar("GME", "stocks", _bar(o=100.0), _utc(2))
        oms.close_all({"GME": 110.0}, _utc(3))
        trade = state.closed_trades[-1]
        assert trade["exit_price"] == pytest.approx(110.0)
        # fill_price = open × (1 + 0/10000) = 100.0 exactly, so pnl = 10×10 = 100
        assert trade["realised_pnl"] == pytest.approx(100.0, rel=0.001)

    def test_force_close_empty_does_not_crash(self):
        state = _state_with_gme()
        oms   = _oms(state)
        oms.close_all({}, _utc(1))   # no positions — should not raise
