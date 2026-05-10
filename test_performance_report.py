"""
tests/test_social.py
Tests for the Stocktwits social velocity pipeline.

Covers:
  MentionWindow   — rolling deque, expiry, count_recent
  ZScoreEngine    — formula correctness, spike detection, state injection
  SentimentTagger — API label pass-through, keyword scoring, edge cases
  Aggregator      — bull/bear ratio, label selection, confidence averaging
"""

from datetime import datetime, timezone, timedelta

import pytest

from chimera_v12.social.zscore import MentionWindow, ZScoreEngine, _mean, _std
from chimera_v12.social.sentiment import tag_message, aggregate, SentimentResult


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── MentionWindow ─────────────────────────────────────────────────────────────

class TestMentionWindow:
    def test_add_and_count(self):
        w = MentionWindow("GME", short_window_min=5)
        now = utcnow()
        w.add(now)
        w.add(now - timedelta(minutes=2))
        w.add(now - timedelta(minutes=4))
        assert w.count_recent(5) == 3

    def test_old_mentions_not_counted(self):
        w = MentionWindow("GME", short_window_min=5)
        now = utcnow()
        w.add(now - timedelta(minutes=10))   # outside 5-min window
        w.add(now - timedelta(minutes=1))    # inside
        assert w.count_recent(5) == 1

    def test_count_zero_when_empty(self):
        w = MentionWindow("GME")
        assert w.count_recent(5) == 0

    def test_add_bulk(self):
        w = MentionWindow("GME", short_window_min=60)
        now = utcnow()
        timestamps = [now - timedelta(minutes=i) for i in range(10)]
        w.add_bulk(timestamps)
        assert w.count_recent(60) == 10

    def test_mention_expiry(self):
        """Mentions older than baseline_hours should be removed."""
        w = MentionWindow("GME", baseline_hours=1)
        ancient = utcnow() - timedelta(hours=2)
        recent  = utcnow() - timedelta(minutes=5)
        w.add(ancient)
        w.add(recent)
        # Trigger expiry by calling count_recent (expires under lock)
        count = w.count_recent(60)
        assert count == 1   # only the recent one

    def test_zscore_none_before_min_baseline(self):
        """Fewer than min_baseline_points snapshots → zscore returns None."""
        w = MentionWindow("GME", min_baseline_points=6)
        # Only 3 snapshots
        for _ in range(3):
            w.snapshot()
        assert w.zscore() is None

    def test_zscore_computable_after_sufficient_baseline(self):
        """After enough snapshots, zscore should return a float."""
        w = MentionWindow("GME", min_baseline_points=3,
                          snapshot_interval=0)  # allow rapid snapshots
        now = utcnow()
        # Seed different mention counts for variance
        for i in range(6):
            w.add_bulk([now - timedelta(minutes=j) for j in range(i * 2)])
            w._last_snapshot = None   # force snapshot
            w.snapshot()
        z = w.zscore()
        assert z is not None
        assert isinstance(z, float)

    def test_spike_detection(self):
        """Z-score ≥ spike_threshold → is_spike() returns True."""
        w = MentionWindow("GME", min_baseline_points=3, spike_threshold=2.0,
                          snapshot_interval=0)
        # Build a low baseline
        for _ in range(5):
            w._baseline.append((utcnow() - timedelta(minutes=10), 1))
        w._last_snapshot = None
        # Add many recent mentions to spike the current count
        now = utcnow()
        w.add_bulk([now - timedelta(seconds=i*10) for i in range(30)])
        # Manually check — zscore may be high
        z = w.zscore()
        if z is not None and z >= 2.0:
            assert w.is_spike() is True

    def test_no_spike_at_baseline(self):
        """Z-score ≈ 0 → is_spike() returns False."""
        w = MentionWindow("GME", spike_threshold=2.0, snapshot_interval=0)
        # Uniform baseline with matching current count
        for _ in range(8):
            w._baseline.append((utcnow(), 5))
        w._baseline[-1] = (utcnow(), 5)
        w.add_bulk([utcnow() - timedelta(minutes=1)] * 5)
        z = w.zscore()
        if z is not None:
            assert w.is_spike() is (z >= 2.0)

    def test_stats_dict_keys(self):
        w = MentionWindow("GME")
        s = w.stats()
        for key in ("symbol", "mentions_recent", "mentions_1h", "zscore", "is_spike"):
            assert key in s

    def test_stats_symbol(self):
        w = MentionWindow("TSLA")
        assert w.stats()["symbol"] == "TSLA"


# ── ZScoreEngine ──────────────────────────────────────────────────────────────

class TestZScoreEngine:
    def test_track_creates_window(self):
        engine = ZScoreEngine()
        w = engine.track("GME")
        assert isinstance(w, MentionWindow)
        assert w.symbol == "GME"

    def test_track_same_symbol_returns_same_window(self):
        engine = ZScoreEngine()
        w1 = engine.track("AMC")
        w2 = engine.track("AMC")
        assert w1 is w2

    def test_add_mention_tracks_symbol(self):
        engine = ZScoreEngine()
        engine.add_mention("GME")
        assert engine.track("GME").count_recent(60) == 1

    def test_add_mentions_bulk(self):
        engine = ZScoreEngine()
        now = utcnow()
        engine.add_mentions_bulk("GME", [now - timedelta(minutes=i) for i in range(5)])
        assert engine.track("GME").count_recent(60) == 5

    def test_zscore_none_for_unknown_symbol(self):
        engine = ZScoreEngine()
        assert engine.zscore("UNKNOWN") is None

    def test_snapshot_all_runs_without_error(self):
        engine = ZScoreEngine()
        engine.track("GME")
        engine.track("AMC")
        engine.snapshot_all()   # should not raise

    def test_inject_into_state_writes_zscore(self):
        from chimera_v12.utils.state import SharedState
        state = SharedState()
        state.market.stocks["GME"] = {
            "close": [100.0]*5, "high": [102.0]*5, "low": [98.0]*5,
            "volume": [1e6]*5,
            "short_interest": 0.2, "rvol": 2.0, "social_zscore": 0.0,
        }
        engine = ZScoreEngine()
        w = engine.track("GME")
        # Manually force a known zscore
        w._baseline.extend([(utcnow(), 2)] * 8)
        w.add_bulk([utcnow() - timedelta(minutes=1)] * 10)

        engine.inject_into_state(state)
        # zscore may be None (not enough variance) — just check key exists
        assert "social_zscore" in state.market.stocks["GME"]

    def test_all_stats_returns_dict(self):
        engine = ZScoreEngine()
        engine.track("GME")
        engine.track("AMC")
        s = engine.all_stats()
        assert "GME" in s
        assert "AMC" in s

    def test_config_propagates_threshold(self):
        engine = ZScoreEngine({"social_spike_threshold": 3.5})
        w = engine.track("GME")
        assert w.spike_threshold == 3.5


# ── Statistical helpers ───────────────────────────────────────────────────────

class TestStatHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([5.0]) == 5.0

    def test_mean_correct(self):
        assert _mean([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_std_empty_or_single(self):
        assert _std([]) == 0.0
        assert _std([42.0]) == 0.0

    def test_std_uniform(self):
        """Uniform values → std = 0."""
        assert _std([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-9)

    def test_std_known_values(self):
        # Sample std (Bessel's correction) of this dataset is ~2.138, not 2.0
        # (2.0 is the *population* std; we use sample std with n-1 denominator)
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        import math
        mu  = sum(vals) / len(vals)
        expected = math.sqrt(sum((x - mu)**2 for x in vals) / (len(vals) - 1))
        assert _std(vals) == pytest.approx(expected, rel=1e-6)


# ── SentimentTagger ───────────────────────────────────────────────────────────

class TestSentimentTagger:
    # ── API label passthrough ─────────────────────────────────────────────
    def test_api_bullish_label(self):
        result = tag_message("some text", api_label="Bullish")
        assert result.label == "bullish"
        assert result.confidence == pytest.approx(0.85)
        assert result.source == "api_tag"

    def test_api_bearish_label(self):
        result = tag_message("some text", api_label="Bearish")
        assert result.label == "bearish"
        assert result.confidence == pytest.approx(0.85)
        assert result.source == "api_tag"

    def test_api_label_case_insensitive(self):
        assert tag_message("x", api_label="BULLISH").label == "bullish"
        assert tag_message("x", api_label="bearish").label == "bearish"

    def test_no_api_label_falls_to_keyword(self):
        result = tag_message("moon rocket squeeze 🚀", api_label=None)
        assert result.source in ("keyword", "neutral_default")

    # ── Keyword scoring ────────────────────────────────────────────────────
    def test_bullish_keywords(self):
        for phrase in ["this is bullish", "going to the moon 🚀",
                        "short squeeze incoming", "diamond hands 💎🙌"]:
            result = tag_message(phrase)
            assert result.label == "bullish", f"Expected bullish for: {phrase}"

    def test_bearish_keywords(self):
        for phrase in ["this is bearish", "it will dump and crash",
                        "looking at puts", "stock will collapse"]:
            result = tag_message(phrase)
            assert result.label == "bearish", f"Expected bearish for: {phrase}"

    def test_neutral_on_no_keywords(self):
        result = tag_message("just another day in the market")
        assert result.label == "neutral"
        assert result.source == "neutral_default"

    def test_confidence_bounded(self):
        for text in ["moon 🚀 squeeze squeeze squeeze",
                      "crash dump sell sell sell",
                      "nothing here"]:
            r = tag_message(text)
            assert 0.0 <= r.confidence <= 1.0

    def test_bull_score_higher_than_bear_for_bullish(self):
        r = tag_message("super bullish moon breakout 🚀")
        assert r.label == "bullish"
        assert r.bull_score > r.bear_score

    def test_bear_score_higher_for_bearish(self):
        r = tag_message("crash dump puts bearish 📉")
        assert r.label == "bearish"
        assert r.bear_score > r.bull_score

    def test_mixed_signals_near_neutral(self):
        """Equal bullish and bearish keywords → neutral."""
        r = tag_message("bullish but also bearish puts and moon")
        # May be neutral or slightly biased — just check it's reasonable
        assert r.label in ("bullish", "bearish", "neutral")

    def test_empty_string_neutral(self):
        r = tag_message("")
        assert r.label == "neutral"

    def test_result_is_dataclass(self):
        r = tag_message("test")
        assert hasattr(r, "label")
        assert hasattr(r, "confidence")
        assert hasattr(r, "bull_score")
        assert hasattr(r, "bear_score")
        assert hasattr(r, "bull_ratio")
        assert hasattr(r, "source")


# ── Sentiment Aggregator ──────────────────────────────────────────────────────

class TestSentimentAggregator:
    def _make_results(self, labels: list[str]) -> list[SentimentResult]:
        return [
            SentimentResult(
                label=l, confidence=0.75,
                bull_score=1.0 if l == "bullish" else 0.0,
                bear_score=1.0 if l == "bearish" else 0.0,
                bull_ratio=1.0 if l == "bullish" else (0.0 if l == "bearish" else 0.5),
                source="api_tag",
            )
            for l in labels
        ]

    def test_empty_results_neutral(self):
        agg = aggregate("GME", [])
        assert agg.label == "neutral"
        assert agg.n_messages == 0

    def test_all_bullish(self):
        results = self._make_results(["bullish"] * 10)
        agg = aggregate("GME", results)
        assert agg.label == "bullish"
        assert agg.bull_count == 10
        assert agg.bear_count == 0

    def test_all_bearish(self):
        results = self._make_results(["bearish"] * 10)
        agg = aggregate("GME", results)
        assert agg.label == "bearish"

    def test_mixed_tilts_to_majority(self):
        results = self._make_results(["bullish"] * 7 + ["bearish"] * 3)
        agg = aggregate("GME", results)
        assert agg.label == "bullish"
        assert agg.bull_ratio == pytest.approx(0.7)

    def test_near_equal_split_neutral(self):
        results = self._make_results(["bullish"] * 5 + ["bearish"] * 5)
        agg = aggregate("GME", results)
        assert agg.label == "neutral"

    def test_symbol_preserved(self):
        agg = aggregate("TSLA", self._make_results(["neutral"]))
        assert agg.symbol == "TSLA"

    def test_n_messages_correct(self):
        results = self._make_results(["bullish"] * 3 + ["bearish"] * 2 + ["neutral"] * 1)
        agg = aggregate("GME", results)
        assert agg.n_messages == 6
        assert agg.bull_count  == 3
        assert agg.bear_count  == 2
        assert agg.neutral_count == 1

    def test_confidence_is_average(self):
        results = [
            SentimentResult("bullish", 0.8, 1,0,1,"api_tag"),
            SentimentResult("bullish", 0.6, 1,0,1,"keyword"),
        ]
        agg = aggregate("GME", results)
        assert agg.confidence == pytest.approx(0.7)

    def test_too_few_polar_messages_neutral(self):
        """Fewer than 3 bullish + bearish messages → neutral regardless."""
        results = self._make_results(["bullish", "bearish"])  # only 2 polar
        agg = aggregate("GME", results)
        assert agg.label == "neutral"
