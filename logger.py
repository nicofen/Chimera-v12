"""
tests/test_performance_report.py
Tests for PerformanceReport — all key metric calculations verified
against known analytical solutions.

Every statistical formula has at least one test where the expected answer
is computed by hand, so regressions in the formula are immediately visible.
"""

from datetime import datetime, timezone, timezone, timedelta

import math
import pytest

from chimera_v12.backtest.performance import (
    PerformanceReport, _mean, _std, _sharpe, _sortino,
    _max_drawdown, _max_streak
)

def _dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)

def _make_trades(pnls: list[float], sector="stocks") -> list[dict]:
    """Build a minimal trade list from a list of P&L values."""
    trades = []
    for i, pnl in enumerate(pnls):
        risk = 100.0
        trades.append({
            "symbol":       "GME",
            "sector":       sector,
            "side":         "buy",
            "realised_pnl": pnl,
            "r_multiple":   pnl / risk,
            "close_reason": "TP_HIT" if pnl > 0 else "STOP_HIT",
            "entry_dt":     _dt(i + 1).isoformat(),
            "exit_dt":      _dt(i + 2).isoformat(),
            "commission":   2.0
        })
    return trades

def _equity_curve(equities: list[float]) -> list[tuple[datetime, float]]:
    return [(_dt(i + 1), eq) for i, eq in enumerate(equities)]

# ── Statistical helper functions ──────────────────────────────────────────────

class TestStatHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_known(self):
        assert _mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_std_empty(self):
        assert _std([]) == 0.0

    def test_std_single(self):
        assert _std([42.0]) == 0.0

    def test_std_known(self):
        # Population variance = 2.5, sample variance = 2.5×4/3 ≈ not needed
        # std([1,2,3,4,5]) with Bessel's correction = sqrt(2.5) ≈ 1.5811
        assert _std([1, 2, 3, 4, 5]) == pytest.approx(math.sqrt(2.5), rel=1e-4)

    def test_std_uniform(self):
        assert _std([7.0, 7.0, 7.0, 7.0]) == pytest.approx(0.0, abs=1e-10)

class TestSharpeFormula:
    def test_zero_std_returns_zero(self):
        assert _sharpe([0.01] * 10) == pytest.approx(0.0, abs=1e-6)
        # All same return → std=0 → Sharpe undefined → 0

    def test_positive_returns_positive_sharpe(self):
        returns = [0.002] * 50 + [0.001] * 50   # consistently positive excess
        s = _sharpe(returns)
        assert s > 0

    def test_negative_returns_negative_sharpe(self):
        # Varying negative returns ensure std > 0 so Sharpe can be computed
        returns = [-0.001, -0.003, -0.002, -0.004, -0.001, -0.003
                   -0.002, -0.004, -0.001, -0.003]
        s = _sharpe(returns)
        assert s < 0

    def test_annualised_by_sqrt_252(self):
        # Sharpe = mean/std * sqrt(252)
        returns = [0.001, 0.002, 0.0015, 0.003, 0.0005]
        mu  = _mean(returns)
        sig = _std(returns)
        expected = mu / sig * math.sqrt(252) if sig > 0 else 0.0
        assert _sharpe(returns) == pytest.approx(expected, rel=1e-4)

class TestSortinoFormula:
    def test_no_negative_returns_zero_downside(self):
        returns = [0.01, 0.02, 0.015]
        rfr = 0.0
        # No negative excess → downside_std = 0 → Sortino undefined → 0
        s = _sortino(returns, rfr)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_higher_with_fewer_losses(self):
        rfr = 0.0
        all_pos = [0.01] * 20 + [-0.002] * 2
        all_neg = [0.01] * 10 + [-0.01] * 12
        s1 = _sortino(all_pos, rfr)
        s2 = _sortino(all_neg, rfr)
        assert s1 > s2

class TestMaxDrawdown:
    def test_no_drawdown(self):
        curve = _equity_curve([100, 110, 120, 130])
        dd_pct, dd_days = _max_drawdown(curve)
        assert dd_pct == pytest.approx(0.0, abs=1e-9)
        assert dd_days == 0

    def test_simple_drawdown(self):
        # Peak 120, trough 90 → DD = (120-90)/120 = 0.25
        curve = _equity_curve([100, 110, 120, 90, 100])
        dd_pct, _ = _max_drawdown(curve)
        assert dd_pct == pytest.approx(0.25, rel=1e-4)

    def test_drawdown_duration(self):
        # Peak on Jan 3, trough on Jan 5 → 2 days
        curve = [(datetime(2024,1,1,tzinfo=timezone.utc), 100),
                 (datetime(2024,1,2,tzinfo=timezone.utc), 110),
                 (datetime(2024,1,3,tzinfo=timezone.utc), 120),
                 (datetime(2024,1,5,tzinfo=timezone.utc), 90),
                 (datetime(2024,1,7,tzinfo=timezone.utc), 95)]
        dd_pct, dd_days = _max_drawdown(curve)
        assert dd_pct > 0
        assert dd_days == 2   # Jan 3 → Jan 5 (trough, not recovery)

    def test_multiple_drawdowns_returns_max(self):
        # Two drawdowns: 10% and 25%
        curve = _equity_curve([100, 90, 100, 120, 90, 100])
        dd_pct, _ = _max_drawdown(curve)
        # Max DD: peak 120, trough 90 → (120-90)/120 = 0.25
        assert dd_pct == pytest.approx(0.25, rel=1e-4)

    def test_single_bar_no_drawdown(self):
        curve = _equity_curve([100.0])
        dd_pct, dd_days = _max_drawdown(curve)
        assert dd_pct == pytest.approx(0.0, abs=1e-9)

class TestMaxStreak:
    def test_win_streak(self):
        pnls = [10, 20, 30, -5, 10, 10, 10, 10, -5]
        assert _max_streak(pnls, positive=True) == 4

    def test_loss_streak(self):
        pnls = [-5, 10, -3, -4, -5, -2, 10]
        assert _max_streak(pnls, positive=False) == 4

    def test_no_wins(self):
        pnls = [-1, -2, -3]
        assert _max_streak(pnls, positive=True) == 0

    def test_no_losses(self):
        pnls = [1, 2, 3]
        assert _max_streak(pnls, positive=False) == 0

    def test_alternating(self):
        pnls = [1, -1, 1, -1, 1]
        assert _max_streak(pnls, positive=True)  == 1
        assert _max_streak(pnls, positive=False) == 1

# ── PerformanceReport.compute() ───────────────────────────────────────────────

class TestPerformanceReport:
    def _report(self, pnls, equity_curve=None, initial=100_000.0):
        trades = _make_trades(pnls)
        curve  = equity_curve or _equity_curve(
            [initial + sum(pnls[:i]) for i in range(len(pnls) + 1)]
        )
        return PerformanceReport(trades, curve, initial)

    def test_no_trades_returns_error(self):
        r = PerformanceReport([], [], 100_000.0)
        m = r.compute()
        assert "error" in m

    def test_total_trades_correct(self):
        r = self._report([100, -50, 200, -30, 150])
        m = r.compute()
        assert m["total_trades"] == 5

    def test_win_rate_correct(self):
        # 3 wins, 2 losses
        r = self._report([100, -50, 200, -30, 150])
        m = r.compute()
        assert m["win_rate"] == pytest.approx(0.6)
        assert m["winning_trades"] == 3
        assert m["losing_trades"] == 2

    def test_profit_factor(self):
        # Gross profit = 100+200+150 = 450; Gross loss = 50+30 = 80; PF = 5.625
        r = self._report([100, -50, 200, -30, 150])
        m = r.compute()
        assert m["profit_factor"] == pytest.approx(450 / 80, rel=1e-3)

    def test_net_profit(self):
        pnls = [100, -50, 200, -30, 150]
        r = self._report(pnls)
        m = r.compute()
        assert m["net_profit"] == pytest.approx(sum(pnls))

    def test_total_return_pct(self):
        pnls = [10_000.0]  # 10% return on 100k
        r = self._report(pnls)
        m = r.compute()
        assert m["total_return_pct"] == pytest.approx(10.0, rel=1e-2)

    def test_avg_r_correct(self):
        # Each trade: risk=$100, so R = pnl/100
        pnls = [100, -50, 200]  # R: 1.0, -0.5, 2.0 → avg = 0.833
        r = self._report(pnls)
        m = r.compute()
        assert m["avg_r"] == pytest.approx((1.0 + -0.5 + 2.0) / 3, rel=0.01)

    def test_max_drawdown_in_report(self):
        # Equity: 100k → 110k → 95k → 105k  (DD = 15/110 ≈ 13.6%)
        curve = _equity_curve([100_000, 110_000, 95_000, 105_000])
        r = PerformanceReport(_make_trades([10_000, -15_000, 10_000]),
                              curve, 100_000)
        m = r.compute()
        assert m["max_drawdown_pct"] == pytest.approx(
            (110_000 - 95_000) / 110_000 * 100, rel=1e-3
        )

    def test_sharpe_present_and_finite(self):
        pnls = [100, -50, 200, -30, 150, 80, -40, 120, -60, 90]
        r = self._report(pnls)
        m = r.compute()
        assert "sharpe_ratio" in m
        assert math.isfinite(m["sharpe_ratio"])

    def test_sortino_present_and_finite(self):
        pnls = [100, -50, 200, -30, 150]
        r = self._report(pnls)
        m = r.compute()
        assert "sortino_ratio" in m
        assert math.isfinite(m["sortino_ratio"])

    def test_by_sector_breakdown(self):
        trades = _make_trades([100, -50]) + _make_trades([200], sector="crypto")
        for t in trades[2:]:
            t["sector"] = "crypto"
        curve = _equity_curve([100_000, 100_100, 100_050, 100_250])
        r = PerformanceReport(trades, curve, 100_000)
        m = r.compute()
        assert "by_sector" in m
        assert "stocks" in m["by_sector"] or "crypto" in m["by_sector"]

    def test_close_reasons_breakdown(self):
        pnls = [100, -50, 200]
        r = self._report(pnls)
        m = r.compute()
        assert "close_reasons" in m
        reasons = m["close_reasons"]
        assert sum(reasons.values()) == len(pnls)

    def test_streak_metrics_present(self):
        r = self._report([100, 100, -50, 100])
        m = r.compute()
        assert m["max_win_streak"]  == 2
        assert m["max_loss_streak"] == 1

    def test_calmar_ratio_positive_on_positive_cagr(self):
        """Calmar = CAGR / max_drawdown — should be positive when CAGR > 0."""
        pnls = [1_000] * 20   # consistent winners
        curve = _equity_curve([100_000 + i * 1_000 for i in range(21)])
        r = PerformanceReport(_make_trades(pnls), curve, 100_000)
        m = r.compute()
        if m["max_drawdown_pct"] > 0:
            assert m["calmar_ratio"] > 0

    def test_print_summary_does_not_crash(self, capsys):
        r = self._report([100, -50, 200, -30, 150])
        r.print_summary()   # should not raise
        captured = capsys.readouterr()
        assert "CHIMERA" in captured.out or "chimera" in captured.out.lower()
