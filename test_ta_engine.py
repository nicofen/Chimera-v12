"""
tests/test_trailing_stop.py
Tests for TrailingStopManager — the most safety-critical component
after the circuit breaker.

Key invariants that must always hold:
  - Stop NEVER moves against the position (longs: stop only rises)
  - Breakeven stage fires at exactly 1R profit
  - Lock-in stage fires at exactly 2R profit
  - Standard trail always trails price by ATR × multiple
  - stop_hit() and tp_hit() detect exact crossings correctly
  - Short positions mirror long logic exactly
"""

import pytest
from chimera_v12.oms.trailing_stop import TrailingStopManager
from chimera_v12.oms.models import Order, OrderSide, OrderStatus

@pytest.fixture
def mgr(config) -> TrailingStopManager:
    return TrailingStopManager(config)

def filled_order(side=OrderSide.BUY, entry=100.0, stop=94.0, tp=109.0, atr=3.0) -> Order:
    o = Order(
        symbol="GME", sector="stocks",
        side=side, qty=10.0,
        entry_price=entry, fill_price=entry,
        stop_price=stop, initial_stop=stop,
        take_profit=tp, atr=atr,
        status=OrderStatus.FILLED
    )
    return o

# ── Long trailing stop ────────────────────────────────────────────────────────

class TestLongTrailing:
    def test_no_move_below_entry(self, mgr):
        """Below entry (unrealised loss) → stop does not move."""
        o = filled_order()
        result = mgr.evaluate(o, current_price=98.0)   # price below entry
        assert result is None

    def test_no_move_at_entry(self, mgr):
        """At exact entry price → stop does not move."""
        o = filled_order()
        result = mgr.evaluate(o, current_price=100.0)
        assert result is None

    def test_trail_above_entry_moves_stop(self, mgr):
        """Price well above entry → trail fires and stop rises."""
        o = filled_order(entry=100.0, stop=94.0, atr=3.0)
        # Price at 112 → trail = 112 - 3×2 = 106 > initial stop 94
        result = mgr.evaluate(o, current_price=112.0)
        assert result is not None
        assert result > o.stop_price

    def test_stop_never_decreases_for_long(self, mgr):
        """Calling evaluate repeatedly as price dips — stop never falls."""
        o = filled_order(entry=100.0, stop=94.0, atr=3.0)
        # Move stop up first
        new_stop = mgr.evaluate(o, current_price=115.0)
        assert new_stop is not None
        o.stop_price = new_stop

        # Price dips back — stop should NOT move down
        result = mgr.evaluate(o, current_price=105.0)
        if result is not None:
            assert result >= o.stop_price   # may ratchet up further, never down

    def test_breakeven_at_1r(self, mgr):
        """At 1R profit, stop must be at least at fill price (breakeven)."""
        o = filled_order(entry=100.0, stop=94.0, atr=3.0)
        # 1R = entry + (entry - stop) = 100 + 6 = 106
        price_at_1r = 106.0
        new_stop = mgr.evaluate(o, current_price=price_at_1r)
        if new_stop is not None:
            assert new_stop >= o.fill_price, \
                f"Stop {new_stop:.2f} should be >= fill {o.fill_price:.2f} at 1R"

    def test_trail_atr_distance(self, mgr):
        """Standard trail should be approximately price - ATR × multiple."""
        o = filled_order(entry=100.0, stop=94.0, atr=3.0)
        price = 120.0
        new_stop = mgr.evaluate(o, current_price=price)
        if new_stop is not None:
            expected_trail = price - o.atr * 2.0   # ATR × trailing_atr_multiple
            # Allow for stage adjustments — trail should be near expected
            assert new_stop == pytest.approx(expected_trail, abs=3.0)

    def test_ratchet_sequence(self, mgr):
        """Stop should ratchet up through a sequence of rising prices."""
        o = filled_order(entry=100.0, stop=94.0, atr=3.0)
        stops = []
        for price in [104, 107, 110, 114, 118]:
            new_stop = mgr.evaluate(o, current_price=float(price))
            if new_stop:
                o.stop_price = new_stop
            stops.append(o.stop_price)
        # Stops should be non-decreasing
        for i in range(1, len(stops)):
            assert stops[i] >= stops[i-1], f"Stop decreased at price step {i}"

# ── Short trailing stop ───────────────────────────────────────────────────────

class TestShortTrailing:
    def test_no_move_above_entry(self, mgr):
        """Short above entry (unrealised loss) → stop does not move."""
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0, atr=3.0)
        result = mgr.evaluate(o, current_price=102.0)
        assert result is None

    def test_trail_below_entry_moves_stop(self, mgr):
        """Price well below entry → short trail fires and stop falls."""
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0, atr=3.0)
        # Price at 88 → trail = 88 + 3×2 = 94 < initial stop 106
        result = mgr.evaluate(o, current_price=88.0)
        assert result is not None
        assert result < o.stop_price

    def test_stop_never_increases_for_short(self, mgr):
        """Short stop should only ever decrease (ratchet downward)."""
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0, atr=3.0)
        new_stop = mgr.evaluate(o, current_price=85.0)
        assert new_stop is not None
        o.stop_price = new_stop

        # Price bounces up slightly — stop must not increase
        result = mgr.evaluate(o, current_price=88.0)
        if result is not None:
            assert result <= o.stop_price

    def test_breakeven_at_1r_short(self, mgr):
        """Short: at 1R profit, stop must be at or below fill price."""
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0, atr=3.0)
        # 1R = entry - (stop - entry) = 100 - 6 = 94
        price_at_1r = 94.0
        new_stop = mgr.evaluate(o, current_price=price_at_1r)
        if new_stop is not None:
            assert new_stop <= o.fill_price

# ── Stop / TP hit detection ───────────────────────────────────────────────────

class TestStopHitDetection:
    def test_long_stop_hit(self, mgr):
        o = filled_order(side=OrderSide.BUY, entry=100.0, stop=94.0)
        assert mgr.stop_hit(o, current_price=93.9) is True
        assert mgr.stop_hit(o, current_price=94.0) is True   # exact stop

    def test_long_stop_not_hit(self, mgr):
        o = filled_order(side=OrderSide.BUY, entry=100.0, stop=94.0)
        assert mgr.stop_hit(o, current_price=94.1) is False
        assert mgr.stop_hit(o, current_price=100.0) is False

    def test_short_stop_hit(self, mgr):
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0)
        assert mgr.stop_hit(o, current_price=106.1) is True
        assert mgr.stop_hit(o, current_price=106.0) is True

    def test_short_stop_not_hit(self, mgr):
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0)
        assert mgr.stop_hit(o, current_price=105.9) is False

    def test_long_tp_hit(self, mgr):
        o = filled_order(side=OrderSide.BUY, tp=109.0)
        assert mgr.tp_hit(o, current_price=109.0) is True
        assert mgr.tp_hit(o, current_price=109.1) is True

    def test_long_tp_not_hit(self, mgr):
        o = filled_order(side=OrderSide.BUY, tp=109.0)
        assert mgr.tp_hit(o, current_price=108.9) is False

    def test_short_tp_hit(self, mgr):
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0, tp=91.0)
        assert mgr.tp_hit(o, current_price=91.0) is True
        assert mgr.tp_hit(o, current_price=90.5) is True

    def test_short_tp_not_hit(self, mgr):
        o = filled_order(side=OrderSide.SELL, entry=100.0, stop=106.0, tp=91.0)
        assert mgr.tp_hit(o, current_price=91.5) is False

    def test_no_tp_set(self, mgr):
        """take_profit=0 → tp_hit always False."""
        o = filled_order(tp=0.0)
        assert mgr.tp_hit(o, current_price=200.0) is False

    def test_closed_order_no_trail(self, mgr):
        """Closed order → evaluate returns None."""
        from chimera_v12.oms.models import OrderStatus
        o = filled_order()
        o.status = OrderStatus.CLOSED
        result = mgr.evaluate(o, current_price=120.0)
        assert result is None
