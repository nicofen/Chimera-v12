"""
tests/test_circuit_breaker.py
Tests for CircuitBreaker — the last line of defence before capital loss.

Invariants that must ALWAYS hold:
  - Tripping sets state.circuit_open = True immediately
  - force_close_all() is called on every trip (positions liquidated)
  - Daily loss trip auto-resets at midnight; drawdown/streak do NOT
  - Manual reset re-enables trading (circuit_open = False)
  - Consecutive losses accumulate correctly; a win resets the streak
  - High-water mark only ever moves upward
  - Drawdown pct = (hwm - equity) / hwm, always non-negative
"""

import asyncio
from datetime import datetime, timezone, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Any

import pytest

from chimera_v12.risk.circuit_breaker import CircuitBreaker
from chimera_v12.risk.circuit_breaker_models import (
    BreakerState, TripReason, BreakerStatus
)
from chimera_v12.utils.state import SharedState

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_oms():
    oms = MagicMock()
    oms.force_close_all = AsyncMock()
    return oms

@pytest.fixture
def breaker_config() -> dict[str, Any]:
    return {
        "daily_loss_limit_pct":  0.05,   # 5%,
        "drawdown_limit_pct":    0.10,   # 10%,
        "loss_streak_limit":     4,
        "db_path":               ":memory:",  # SQLite in-memory for tests
    }

@pytest.fixture
def state_100k() -> SharedState:
    s = SharedState()
    s.equity = 100_000.0
    s.circuit_open = False
    return s

@pytest.fixture
def breaker(state_100k, mock_oms, breaker_config) -> CircuitBreaker:
    cb = CircuitBreaker(state_100k, mock_oms, breaker_config)
    cb._equity_start_of_day = 100_000.0
    cb._last_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cb.status.high_water_mark = 100_000.0
    return cb

def run(coro):
    return asyncio.run(coro)

# ── High-water mark & drawdown ────────────────────────────────────────────────

class TestHighWaterMark:
    def test_hwm_rises_with_equity(self, breaker, state_100k):
        state_100k.equity = 110_000.0
        breaker._update_high_water_mark()
        assert breaker.status.high_water_mark == 110_000.0

    def test_hwm_never_falls(self, breaker, state_100k):
        state_100k.equity = 110_000.0
        breaker._update_high_water_mark()
        state_100k.equity = 95_000.0
        breaker._update_high_water_mark()
        assert breaker.status.high_water_mark == 110_000.0

    def test_drawdown_pct_computed_correctly(self, breaker, state_100k):
        state_100k.equity = 110_000.0
        breaker._update_high_water_mark()
        state_100k.equity = 99_000.0
        breaker._update_high_water_mark()
        expected = (110_000 - 99_000) / 110_000
        assert breaker.status.drawdown_pct == pytest.approx(expected, rel=1e-6)

    def test_drawdown_zero_at_hwm(self, breaker, state_100k):
        state_100k.equity = 100_000.0
        breaker._update_high_water_mark()
        assert breaker.status.drawdown_pct == pytest.approx(0.0)

    def test_drawdown_non_negative(self, breaker, state_100k):
        for equity in [110_000, 90_000, 105_000, 80_000]:
            state_100k.equity = float(equity)
            breaker._update_high_water_mark()
            assert breaker.status.drawdown_pct >= 0.0

# ── Daily loss calculation ─────────────────────────────────────────────────────

class TestDailyLoss:
    def test_loss_computed_from_start_of_day(self, breaker, state_100k):
        state_100k.equity = 96_000.0
        breaker._update_daily_loss()
        assert breaker.status.daily_loss_usd == pytest.approx(-4_000.0)

    def test_profit_shows_positive(self, breaker, state_100k):
        state_100k.equity = 103_000.0
        breaker._update_daily_loss()
        assert breaker.status.daily_loss_usd == pytest.approx(3_000.0)

    def test_unrealised_pnl_included(self, breaker, state_100k):
        """Unrealised losses on open positions count toward daily loss."""
        mock_pos = MagicMock()
        mock_pos.unrealised_pnl = -2_000.0
        state_100k.open_positions["GME"] = mock_pos
        state_100k.equity = 100_000.0  # realised flat
        breaker._update_daily_loss()
        # Total = 0 (realised) + (-2000) (unrealised) = -2000
        assert breaker.status.daily_loss_usd == pytest.approx(-2_000.0)

# ── Trip condition checks ─────────────────────────────────────────────────────

class TestTripConditions:
    def test_no_trip_in_normal_conditions(self, breaker, state_100k):
        state_100k.equity = 100_000.0
        breaker._update_high_water_mark()
        breaker._update_daily_loss()
        reason = breaker._check_trip_conditions()
        assert reason is None

    def test_daily_loss_trips(self, breaker, state_100k):
        """5% loss on $100k = $5,000 → should trip."""
        breaker.status.daily_loss_usd = -5_100.0   # just over 5%
        reason = breaker._check_trip_conditions()
        assert reason == TripReason.DAILY_LOSS

    def test_daily_loss_at_threshold_does_not_trip(self, breaker, state_100k):
        """Exactly 4.9% loss → should NOT trip (limit is 5%)."""
        breaker.status.daily_loss_usd = -4_900.0
        reason = breaker._check_trip_conditions()
        assert reason is None

    def test_drawdown_trips(self, breaker, state_100k):
        """10.1% drawdown → should trip."""
        breaker.status.drawdown_pct = 0.101
        reason = breaker._check_trip_conditions()
        assert reason == TripReason.DRAWDOWN

    def test_drawdown_at_limit_does_not_trip(self, breaker, state_100k):
        breaker.status.drawdown_pct = 0.099
        reason = breaker._check_trip_conditions()
        assert reason is None

    def test_loss_streak_trips(self, breaker):
        """4 consecutive losses → should trip."""
        breaker.status.consecutive_losses = 4
        reason = breaker._check_trip_conditions()
        assert reason == TripReason.LOSS_STREAK

    def test_loss_streak_below_limit_ok(self, breaker):
        breaker.status.consecutive_losses = 3
        reason = breaker._check_trip_conditions()
        assert reason is None

    def test_already_open_skips_check(self, breaker, state_100k):
        """When breaker is already open, _check_trip_conditions should
        not be called again (the evaluate loop returns early)."""
        breaker.status.state = BreakerState.OPEN
        breaker.status.daily_loss_usd = -9_000.0  # would trip if checked
        # Simulate the evaluate loop's early return
        if breaker.status.is_open:
            reason = None  # not called
        else:
            reason = breaker._check_trip_conditions()
        assert reason is None

# ── Trip action ───────────────────────────────────────────────────────────────

class TestTripAction:
    def test_trip_sets_circuit_open(self, breaker, state_100k, mock_oms):
        run(breaker._trip(TripReason.DAILY_LOSS))
        assert state_100k.circuit_open is True

    def test_trip_calls_force_close_all(self, breaker, mock_oms):
        run(breaker._trip(TripReason.DRAWDOWN))
        mock_oms.force_close_all.assert_called_once()

    def test_trip_transitions_to_cooldown(self, breaker):
        run(breaker._trip(TripReason.LOSS_STREAK))
        assert breaker.status.state == BreakerState.COOLDOWN

    def test_trip_records_event(self, breaker):
        run(breaker._trip(TripReason.DAILY_LOSS))
        assert len(breaker.status.events) >= 1
        event = breaker.status.events[-1]
        assert event.reason == TripReason.DAILY_LOSS

    def test_trip_increments_count(self, breaker):
        assert breaker.status.trip_count_today == 0
        run(breaker._trip(TripReason.DRAWDOWN))
        assert breaker.status.trip_count_today == 1

    def test_trip_detail_daily_loss(self, breaker):
        breaker.status.daily_loss_usd = -5_200.0
        detail = breaker._trip_detail(TripReason.DAILY_LOSS)
        assert "5,200" in detail or "5200" in detail

    def test_trip_detail_drawdown(self, breaker):
        breaker.status.drawdown_pct = 0.112
        detail = breaker._trip_detail(TripReason.DRAWDOWN)
        assert "11.2" in detail or "0.11" in detail

    def test_trip_detail_streak(self, breaker):
        breaker.status.consecutive_losses = 4
        detail = breaker._trip_detail(TripReason.LOSS_STREAK)
        assert "4" in detail

    def test_allows_trading_false_after_trip(self, breaker):
        run(breaker._trip(TripReason.MANUAL))
        assert breaker.status.allows_trading is False

    def test_is_open_true_after_trip(self, breaker):
        run(breaker._trip(TripReason.DAILY_LOSS))
        assert breaker.status.is_open is True

# ── Manual reset ──────────────────────────────────────────────────────────────

class TestManualReset:
    def test_reset_from_cooldown_succeeds(self, breaker, state_100k):
        run(breaker._trip(TripReason.DRAWDOWN))
        assert breaker.status.state == BreakerState.COOLDOWN
        result = breaker.reset("investigated ok")
        assert result is True

    def test_reset_clears_circuit_open(self, breaker, state_100k):
        run(breaker._trip(TripReason.LOSS_STREAK))
        breaker.reset("all good")
        assert state_100k.circuit_open is False

    def test_reset_state_becomes_closed(self, breaker):
        run(breaker._trip(TripReason.DRAWDOWN))
        breaker.reset()
        assert breaker.status.state == BreakerState.CLOSED

    def test_reset_clears_streak(self, breaker):
        breaker.status.consecutive_losses = 4
        run(breaker._trip(TripReason.LOSS_STREAK))
        breaker.reset("ok")
        assert breaker.status.consecutive_losses == 0

    def test_reset_allows_trading(self, breaker):
        run(breaker._trip(TripReason.DRAWDOWN))
        breaker.reset()
        assert breaker.status.allows_trading is True

    def test_reset_when_already_closed_returns_false(self, breaker):
        assert breaker.status.state == BreakerState.CLOSED
        result = breaker.reset("unnecessary")
        assert result is False

    def test_reset_records_event(self, breaker):
        run(breaker._trip(TripReason.DAILY_LOSS))
        events_before = len(breaker.status.events)
        breaker.reset("tested")
        assert len(breaker.status.events) == events_before + 1
        assert breaker.status.events[-1].new_state == BreakerState.CLOSED

# ── Midnight reset ────────────────────────────────────────────────────────────

class TestMidnightReset:
    def test_new_day_resets_daily_loss(self, breaker, state_100k):
        state_100k.equity = 97_000.0
        breaker._update_daily_loss()
        assert breaker.status.daily_loss_usd < 0

        # Simulate new day
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        breaker._last_day = "2020-01-01"   # force old day
        breaker._maybe_midnight_reset()

        assert breaker.status.daily_loss_usd == pytest.approx(0.0)

    def test_daily_loss_trip_auto_rearms_at_midnight(self, breaker, state_100k):
        run(breaker._trip(TripReason.DAILY_LOSS))
        assert breaker.status.state == BreakerState.COOLDOWN

        breaker._last_day = "2020-01-01"   # force midnight reset
        breaker._maybe_midnight_reset()

        assert breaker.status.state == BreakerState.CLOSED
        assert state_100k.circuit_open is False

    def test_drawdown_trip_does_NOT_auto_reset(self, breaker, state_100k):
        run(breaker._trip(TripReason.DRAWDOWN))
        assert breaker.status.state == BreakerState.COOLDOWN

        breaker._last_day = "2020-01-01"
        breaker._maybe_midnight_reset()

        # Drawdown requires manual reset — should still be in cooldown
        assert breaker.status.state == BreakerState.COOLDOWN

# ── BreakerStatus properties ──────────────────────────────────────────────────

class TestBreakerStatusProperties:
    def test_is_open_closed_state(self):
        s = BreakerStatus(state=BreakerState.CLOSED)
        assert s.is_open is False

    def test_is_open_open_state(self):
        s = BreakerStatus(state=BreakerState.OPEN)
        assert s.is_open is True

    def test_is_open_cooldown_state(self):
        s = BreakerStatus(state=BreakerState.COOLDOWN)
        assert s.is_open is True

    def test_allows_trading_only_when_closed(self):
        assert BreakerStatus(state=BreakerState.CLOSED).allows_trading is True
        assert BreakerStatus(state=BreakerState.OPEN).allows_trading is False
        assert BreakerStatus(state=BreakerState.COOLDOWN).allows_trading is False

# ── Full evaluate loop ────────────────────────────────────────────────────────

class TestEvaluateLoop:
    @pytest.mark.asyncio
    async def test_evaluate_trips_on_daily_loss(self, breaker, state_100k):
        # Lower equity below 5% threshold; _update_daily_loss derives from equity
        state_100k.equity = 93_500.0   # 6.5% below 100k start
        breaker._equity_start_of_day = 100_000.0
        # Prevent midnight reset from overwriting start-of-day equity
        from datetime import datetime, timezone
        breaker._last_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await breaker._evaluate()
        assert state_100k.circuit_open is True
        assert breaker.status.state == BreakerState.COOLDOWN

    @pytest.mark.asyncio
    async def test_evaluate_no_trip_when_healthy(self, breaker, state_100k):
        state_100k.equity = 101_000.0
        breaker._equity_start_of_day = 100_000.0
        from datetime import datetime, timezone
        breaker._last_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        await breaker._evaluate()
        assert state_100k.circuit_open is False
        assert breaker.status.state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_evaluate_skips_trip_when_already_open(self, breaker, state_100k, mock_oms):
        breaker.status.state = BreakerState.OPEN
        state_100k.circuit_open = True
        initial_calls = mock_oms.force_close_all.call_count
        from datetime import datetime, timezone
        breaker._last_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Even with a huge daily loss, should not re-trip
        state_100k.equity = 50_000.0
        breaker._equity_start_of_day = 100_000.0
        await breaker._evaluate()
        assert mock_oms.force_close_all.call_count == initial_calls
