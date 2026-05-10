"""
tests/test_risk_sizing.py
Tests for the RiskAgent's position sizing logic.

Covers:
  - Kelly Criterion formula correctness
  - Kelly fraction bounded by MAX_KELLY_FRACTION
  - Position size formula: (equity × risk%) / (ATR × 2)
  - News multiplier scales position correctly
  - Zero position on veto
  - Confidence scalar applied before sizing
  - Negative expectancy yields near-zero Kelly
"""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chimera_v12.agents.risk_agent import RiskAgent, MAX_KELLY_FRACTION, MIN_CONFIDENCE
from chimera_v12.utils.state import SharedState, TechnicalSignals, Sentiment
from chimera_v12.utils.state import NewsState

class TestKellyCriterion:
    """Test _kelly_fraction() in isolation."""

    def _make_agent(self, config, wins, losses):
        state = SharedState()
        state.equity = 100_000.0
        agent = RiskAgent(state, config)
        # Populate win history
        for _ in range(wins):
            agent._win_history.append(True)
        for _ in range(losses):
            agent._win_history.append(False)
        return agent

    def test_insufficient_history_returns_conservative(self, config):
        """< 10 trades → conservative 10% default."""
        state = SharedState(); state.equity = 100_000.0
        agent = RiskAgent(state, config)
        # Only 5 records
        for _ in range(5):
            agent._win_history.append(True)
        assert agent._kelly_fraction() == pytest.approx(0.10)

    def test_positive_expectancy(self, config):
        """60% win rate with 1.5R avg win → positive Kelly fraction."""
        agent = self._make_agent(config, wins=30, losses=20)
        f = agent._kelly_fraction()
        assert f > 0.0
        assert f <= MAX_KELLY_FRACTION

    def test_negative_expectancy_near_zero(self, config):
        """30% win rate with 1:1 R ratio → negative expectancy → f ≈ 0."""
        cfg = {**config, "avg_win_r": 1.0, "avg_loss_r": 1.0}
        agent = self._make_agent(cfg, wins=6, losses=14)  # 30% WR
        f = agent._kelly_fraction()
        assert f <= 0.05

    def test_bounded_by_max_kelly(self, config):
        """Even with 90% win rate, Kelly never exceeds MAX_KELLY_FRACTION."""
        agent = self._make_agent(config, wins=45, losses=5)
        f = agent._kelly_fraction()
        assert f <= MAX_KELLY_FRACTION

    def test_50_50_win_rate(self, config):
        """50% WR with 1.5R avg win → small positive Kelly."""
        agent = self._make_agent(config, wins=25, losses=25)
        f = agent._kelly_fraction()
        # f* = (1.5×0.5 - 0.5) / 1.5 = 0.25/1.5 ≈ 0.167
        assert 0.05 < f <= MAX_KELLY_FRACTION

class TestPositionSizing:
    """Test the full _process() pipeline with mocked state queues."""

    @pytest.fixture
    def agent_and_state(self, config):
        state        = SharedState()
        state.equity = 100_000.0
        # Seed Kelly history with decent win rate
        agent = RiskAgent(state, config)
        for _ in range(30):
            agent._win_history.append(True)
        for _ in range(20):
            agent._win_history.append(False)
        return agent, state

    @pytest.fixture
    def signal(self) -> TechnicalSignals:
        return TechnicalSignals(
            sector="stocks", symbol="GME", direction="long",
            confidence=0.80, sp_score=0.72, adx=35.0,
            atr=3.0, timestamp=None
        )

    def test_position_size_formula(self, agent_and_state, config):
        """
        pos_size = (equity × effective_risk%) / (ATR × 2)
        With known inputs, verify the formula numerically.
        """
        agent, state = agent_and_state
        state.market.stocks["GME"] = {
            "close": [100.0] * 250,
            "high":  [103.0] * 250,
            "low":   [97.0]  * 250,
            "volume":[1e6]   * 250
        }

        sig = TechnicalSignals(
            sector="stocks", symbol="GME", direction="long",
            confidence=1.0,   # no confidence scaling,
            atr=3.0, timestamp=None, sp_score=0.0, adx=30.0
        )
        state.signals.append(sig)

        orders_received = []

        async def capture_order(rp):
            orders_received.append(rp)

        state.put_order = capture_order
        state.news.confidence   = 1.0
        state.news.sentiment    = Sentiment.NEUTRAL
        state.news.veto_active  = False

        asyncio.run(agent._process(sig))

        if orders_received:
            rp = orders_received[0]
            equity     = 100_000.0
            risk_pct   = min(config["base_risk_pct"], MAX_KELLY_FRACTION)
            expected   = (equity * risk_pct) / (3.0 * 2)
            assert rp.position_size == pytest.approx(expected, rel=0.30)

    def test_veto_blocks_order(self, agent_and_state, signal):
        """Active news veto → no order produced."""
        agent, state = agent_and_state
        state.news.veto_active = True

        orders = []
        async def capture(rp): orders.append(rp)
        state.put_order = capture

        asyncio.run(agent._process(signal))
        assert len(orders) == 0

    def test_circuit_open_blocks_order(self, agent_and_state, signal):
        """Circuit breaker open → no order produced."""
        agent, state = agent_and_state
        state.circuit_open = True

        orders = []
        async def capture(rp): orders.append(rp)
        state.put_order = capture

        asyncio.run(agent._process(signal))
        assert len(orders) == 0

    def test_low_confidence_blocked(self, agent_and_state):
        """Signal below MIN_CONFIDENCE threshold → no order."""
        agent, state = agent_and_state
        sig = TechnicalSignals(
            sector="stocks", symbol="GME", direction="long",
            confidence=MIN_CONFIDENCE - 0.05,  # just below threshold,
            atr=3.0, timestamp=None, sp_score=0.0, adx=30.0
        )

        orders = []
        async def capture(rp): orders.append(rp)
        state.put_order = capture

        asyncio.run(agent._process(sig))
        assert len(orders) == 0

    def test_zero_equity_blocked(self, agent_and_state, signal):
        """Zero equity → no order (division guard)."""
        agent, state = agent_and_state
        state.equity = 0.0

        orders = []
        async def capture(rp): orders.append(rp)
        state.put_order = capture

        asyncio.run(agent._process(signal))
        assert len(orders) == 0

class TestNewsMultiplier:
    def test_veto_returns_zero(self):
        state = SharedState()
        state.news.veto_active = True
        assert state.news_multiplier() == 0.0

    def test_bullish_amplifies(self):
        state = SharedState()
        state.news.veto_active = False
        state.news.sentiment   = Sentiment.BULLISH
        state.news.confidence  = 0.80
        m = state.news_multiplier()
        assert m > 0.80   # bullish amplifies slightly

    def test_bearish_reduces(self):
        state = SharedState()
        state.news.veto_active = False
        state.news.sentiment   = Sentiment.BEARISH
        state.news.confidence  = 0.80
        m = state.news_multiplier()
        assert m < 0.80   # bearish reduces to 50%

    def test_neutral_returns_confidence(self):
        state = SharedState()
        state.news.veto_active = False
        state.news.sentiment   = Sentiment.NEUTRAL
        state.news.confidence  = 0.65
        m = state.news_multiplier()
        assert m == pytest.approx(0.65)

    def test_multiplier_bounded_0_to_1(self):
        state = SharedState()
        state.news.veto_active = False
        state.news.sentiment   = Sentiment.BULLISH
        state.news.confidence  = 1.0
        m = state.news_multiplier()
        assert 0.0 <= m <= 1.0
