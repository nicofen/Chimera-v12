"""
tests/test_alerts.py
Tests for the alert system — dispatcher rate limiting, deduplication
Telegram/Discord message formatting, and event factory functions.

No real network calls are made — senders are mocked.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chimera_v12.alerts.models import (
    AlertEvent, Priority, EventType,
    evt_circuit_trip, evt_circuit_reset, evt_order_filled,
    evt_position_closed, evt_veto_raised, evt_veto_cleared,
    evt_signal, evt_daily_summary, evt_heartbeat, evt_warning
)
from chimera_v12.alerts.dispatcher import AlertDispatcher, RATE_WINDOWS
from chimera_v12.alerts.telegram_sender import TelegramSender, _escape
from chimera_v12.alerts.discord_sender import DiscordSender, _parse_body

def run(coro):
    return asyncio.run(coro)

# ── Event factory functions ────────────────────────────────────────────────────

class TestEventFactories:
    def test_circuit_trip_is_critical(self):
        e = evt_circuit_trip("daily_loss", 95_000, -5_100, 5.1)
        assert e.priority == Priority.CRITICAL
        assert e.event_type == EventType.CIRCUIT_TRIP

    def test_circuit_trip_contains_equity(self):
        e = evt_circuit_trip("drawdown", 90_000, -2_000, 10.2)
        assert "90,000" in e.body or "90000" in e.body

    def test_circuit_reset_is_high(self):
        e = evt_circuit_reset("investigated", 102_000)
        assert e.priority == Priority.HIGH

    def test_order_filled_is_high(self):
        e = evt_order_filled("GME", "buy", 10, 100.25, 94.0, 109.0, "stocks")
        assert e.priority == Priority.HIGH
        assert "GME" in e.title
        assert "BUY" in e.title

    def test_order_filled_body_contains_price(self):
        e = evt_order_filled("GME", "buy", 10, 100.25, 94.0, 109.0, "stocks")
        assert "100.25" in e.body

    def test_position_closed_profit_positive(self):
        e = evt_position_closed("GME", 250.0, 2.5, "TP_HIT", "stocks")
        assert "250" in e.title
        assert "2.5" in e.title

    def test_position_closed_loss_negative(self):
        e = evt_position_closed("GME", -120.0, -1.0, "STOP_HIT", "stocks")
        assert "120" in e.title and "-" in e.title

    def test_veto_raised_is_high(self):
        e = evt_veto_raised("FOMC rate decision", 600)
        assert e.priority == Priority.HIGH
        assert "veto" in e.event_type.lower() or "veto" in e.title.lower()

    def test_veto_cleared_is_normal(self):
        e = evt_veto_cleared()
        assert e.priority == Priority.NORMAL

    def test_signal_is_normal(self):
        e = evt_signal("GME", "stocks", "long", 0.75, 32.1, 0.68)
        assert e.priority == Priority.NORMAL
        assert "GME" in e.title
        assert "LONG" in e.title

    def test_signal_no_sp_score(self):
        """sp=None should not raise."""
        e = evt_signal("ES1!", "futures", "short", 0.65, 28.0, None)
        assert e.title is not None

    def test_daily_summary_contains_pnl(self):
        e = evt_daily_summary(105_000, 1_200, 0.64, 8, 3.2)
        assert "1,200" in e.body or "1200" in e.body

    def test_heartbeat_is_low(self):
        e = evt_heartbeat(100_000, 2, False, "closed")
        assert e.priority == Priority.LOW

    def test_all_events_have_emoji(self):
        events = [
            evt_circuit_trip("x", 100, -1, 1),
            evt_circuit_reset("ok", 100),
            evt_order_filled("X", "buy", 1, 100, 95, 110, "stocks"),
            evt_position_closed("X", 10, 0.1, "TP_HIT", "stocks"),
            evt_veto_raised("test", 300),
            evt_veto_cleared(),
            evt_signal("X", "stocks", "long", 0.7, 30, None),
            evt_heartbeat(100_000, 0, False, "closed")
        ]
        for e in events:
            assert e.emoji, f"No emoji for {e.event_type}"

    def test_event_timestamp_is_utc(self):
        e = evt_signal("X", "stocks", "long", 0.7, 30, None)
        assert e.ts.tzinfo is not None

# ── Priority levels ────────────────────────────────────────────────────────────

class TestPriorityLevels:
    def test_critical_less_than_high(self):
        assert Priority.CRITICAL < Priority.HIGH

    def test_high_less_than_normal(self):
        assert Priority.HIGH < Priority.NORMAL

    def test_normal_less_than_low(self):
        assert Priority.NORMAL < Priority.LOW

    def test_critical_rate_window_is_zero(self):
        assert RATE_WINDOWS[Priority.CRITICAL] == 0

    def test_low_rate_window_longest(self):
        assert RATE_WINDOWS[Priority.LOW] > RATE_WINDOWS[Priority.HIGH]

# ── AlertDispatcher ────────────────────────────────────────────────────────────

class TestAlertDispatcher:
    def _make_dispatcher(self):
        mock_sender = MagicMock()
        mock_sender.send = AsyncMock(return_value=True)
        config = {"alert_heartbeat_interval": 0}
        d = AlertDispatcher(config, telegram_sender=mock_sender)
        return d, mock_sender

    def test_send_nowait_enqueues(self):
        d, _ = self._make_dispatcher()
        e = evt_signal("X", "stocks", "long", 0.7, 30, None)
        d.send_nowait(e)
        assert d._queue.qsize() == 1

    def test_critical_always_enqueues_when_full(self):
        """Even a full queue must accept CRITICAL events."""
        d, _ = self._make_dispatcher()
        d._queue = asyncio.Queue(maxsize=2)
        # Fill with LOW priority
        for _ in range(2):
            d.send_nowait(evt_heartbeat(100_000, 0, False, "closed"))
        # Critical should still get in (by dropping a low event)
        critical = evt_circuit_trip("test", 95_000, -5_000, 5.1)
        d.send_nowait(critical)
        # Queue should contain the critical event
        items = []
        while not d._queue.empty():
            items.append(d._queue.get_nowait())
        assert any(i.priority == Priority.CRITICAL for i in items)

    def test_dispatch_calls_sender(self):
        d, mock_sender = self._make_dispatcher()
        e = evt_order_filled("GME", "buy", 10, 100, 94, 109, "stocks")
        run(d._dispatch(e))
        mock_sender.send.assert_called_once_with(e)

    def test_rate_limiting_suppresses_same_type(self):
        """Same event_type within the rate window should be suppressed."""
        d, mock_sender = self._make_dispatcher()
        # Override last_sent to simulate a recent send
        d._last_sent[Priority.NORMAL] = time.monotonic()   # just now
        d._last_type[Priority.NORMAL] = EventType.SIGNAL_EMITTED

        e = evt_signal("GME", "stocks", "long", 0.7, 30, None)
        run(d._dispatch(e))
        mock_sender.send.assert_not_called()

    def test_rate_limiting_allows_different_type(self):
        """Different event_type bypasses the same-type rate limit."""
        d, mock_sender = self._make_dispatcher()
        d._last_sent[Priority.NORMAL] = time.monotonic()
        d._last_type[Priority.NORMAL] = EventType.VETO_CLEARED  # different type

        e = evt_signal("GME", "stocks", "long", 0.7, 30, None)
        run(d._dispatch(e))
        mock_sender.send.assert_called_once()

    def test_critical_bypasses_rate_limit(self):
        """CRITICAL events are never rate-limited."""
        d, mock_sender = self._make_dispatcher()
        d._last_sent[Priority.CRITICAL] = time.monotonic()
        d._last_type[Priority.CRITICAL] = EventType.CIRCUIT_TRIP

        e = evt_circuit_trip("drawdown", 90_000, -2_000, 10.1)
        run(d._dispatch(e))
        mock_sender.send.assert_called_once()

    def test_no_senders_no_crash(self):
        """Dispatcher with no senders should silently succeed."""
        d = AlertDispatcher({"alert_heartbeat_interval": 0})
        e = evt_signal("X", "stocks", "long", 0.7, 30, None)
        run(d._dispatch(e))   # should not raise

    def test_sender_failure_logged_not_raised(self):
        """If a sender returns False, dispatcher continues without raising."""
        mock_sender = MagicMock()
        mock_sender.send = AsyncMock(return_value=False)
        d = AlertDispatcher({"alert_heartbeat_interval": 0}, telegram_sender=mock_sender)
        e = evt_order_filled("GME", "buy", 10, 100, 94, 109, "stocks")
        run(d._dispatch(e))   # should not raise

    def test_multiple_senders_both_called(self):
        """Both Telegram and Discord senders get the same event."""
        tg = MagicMock(); tg.send = AsyncMock(return_value=True)
        dc = MagicMock(); dc.send = AsyncMock(return_value=True)
        d  = AlertDispatcher({"alert_heartbeat_interval": 0},
                             telegram_sender=tg, discord_sender=dc)
        e  = evt_order_filled("GME", "buy", 10, 100, 94, 109, "stocks")
        run(d._dispatch(e))
        tg.send.assert_called_once_with(e)
        dc.send.assert_called_once_with(e)

# ── Telegram formatting ───────────────────────────────────────────────────────

class TestTelegramFormatting:
    def _sender(self):
        return TelegramSender("fake_token", "12345")

    def test_escape_special_chars(self):
        assert _escape("hello_world") == "hello\\_world"
        assert _escape("(test)") == "\\(test\\)"
        assert _escape("3.14") == "3\\.14"

    def test_escape_empty_string(self):
        assert _escape("") == ""

    def test_escape_no_specials(self):
        assert _escape("hello world") == "hello world"

    def test_format_critical_contains_badge(self):
        s = self._sender()
        e = evt_circuit_trip("daily_loss", 95_000, -5_100, 5.1)
        msg = s._format(e)
        assert "CRITICAL" in msg

    def test_format_normal_no_badge(self):
        s = self._sender()
        e = evt_signal("GME", "stocks", "long", 0.7, 30, None)
        msg = s._format(e)
        assert "CRITICAL" not in msg

    def test_format_contains_timestamp(self):
        s = self._sender()
        e = evt_heartbeat(100_000, 0, False, "closed")
        msg = s._format(e)
        assert "UTC" in msg

    def test_format_under_4096_chars(self):
        s = self._sender()
        e = evt_daily_summary(100_000, 1_200, 0.64, 8, 3.2)
        msg = s._format(e)
        assert len(msg) <= 4096

    def test_format_does_not_raise_on_any_event(self):
        """Formatting should never raise regardless of event content."""
        s = self._sender()
        for factory_fn, args in [
            (evt_circuit_trip,    ("reason", 95000, -5000, 5.1)),
            (evt_order_filled,    ("GME","buy",10,100,94,109,"stocks")),
            (evt_signal,          ("GME","stocks","long",0.7,30,None)),
            (evt_heartbeat,       (100000, 2, False, "closed"))
        ]:
            e = factory_fn(*args)
            msg = s._format(e)
            assert isinstance(msg, str) and len(msg) > 0

# ── Discord formatting ────────────────────────────────────────────────────────

class TestDiscordFormatting:
    def _sender(self):
        return DiscordSender("https://discord.com/api/webhooks/fake/url")

    def test_parse_body_extracts_fields(self):
        body = "*Symbol:* GME\n*Side:* BUY\n*Fill:* $100.25"
        desc, fields = _parse_body(body)
        assert len(fields) == 3
        assert fields[0]["name"] == "Symbol"
        assert fields[0]["value"] == "GME"

    def test_parse_body_description_lines(self):
        body = "This is description\n*Key:* Val\nMore description"
        desc, fields = _parse_body(body)
        assert "description" in desc.lower()
        assert "More description" in desc

    def test_embed_colour_critical(self):
        s = self._sender()
        e = evt_circuit_trip("x", 95_000, -5_000, 5.1)
        payload = s._build_payload(e)
        assert payload["embeds"][0]["color"] == 0xE03050

    def test_embed_colour_normal(self):
        s = self._sender()
        e = evt_signal("GME", "stocks", "long", 0.7, 30, None)
        payload = s._build_payload(e)
        assert payload["embeds"][0]["color"] == 0x3888E8

    def test_embed_has_timestamp(self):
        s = self._sender()
        e = evt_order_filled("GME", "buy", 10, 100, 94, 109, "stocks")
        payload = s._build_payload(e)
        assert "timestamp" in payload["embeds"][0]

    def test_embed_title_contains_emoji(self):
        s = self._sender()
        e = evt_circuit_trip("loss", 95_000, -5_000, 5.1)
        payload = s._build_payload(e)
        title = payload["embeds"][0]["title"]
        assert "🚨" in title

    def test_no_mention_by_default(self):
        s = self._sender()
        e = evt_circuit_trip("x", 95_000, -5_000, 5.1)
        payload = s._build_payload(e)
        assert not payload.get("content")   # None or empty

    def test_mention_when_configured(self):
        s = DiscordSender("https://discord.com/api/webhooks/fake/url",
                          mention_on_critical=True)
        e = evt_circuit_trip("x", 95_000, -5_000, 5.1)
        payload = s._build_payload(e)
        assert "@here" in (payload.get("content") or "")

    def test_low_priority_no_mention(self):
        s = DiscordSender("https://discord.com/api/webhooks/fake/url",
                          mention_on_critical=True)
        e = evt_heartbeat(100_000, 0, False, "closed")
        payload = s._build_payload(e)
        # @here should NOT appear on LOW priority even with mention_on_critical
        assert "@here" not in (payload.get("content") or "")

    def test_build_payload_does_not_raise(self):
        """Payload building should never raise for any valid event."""
        s = self._sender()
        for e in [
            evt_circuit_trip("x", 95000, -5000, 5.1),
            evt_order_filled("GME","buy",10,100,94,109,"stocks"),
            evt_signal("GME","stocks","long",0.7,30,None),
            evt_heartbeat(100000, 2, False, "closed")
        ]:
            payload = s._build_payload(e)
            assert "embeds" in payload
            assert len(payload["embeds"]) == 1

# ── build_dispatcher factory ──────────────────────────────────────────────────

class TestBuildDispatcher:
    def test_no_env_vars_returns_empty_dispatcher(self):
        from chimera_v12.alerts.dispatcher import build_dispatcher
        d = build_dispatcher({})   # no credentials set
        assert isinstance(d, AlertDispatcher)
        assert len(d.senders) == 0

    def test_with_telegram_config_creates_sender(self):
        from chimera_v12.alerts.dispatcher import build_dispatcher
        d = build_dispatcher({
            "telegram_bot_token": "fake_token_12345",
            "telegram_chat_id": "987654321"
        })
        assert len(d.senders) == 1

    def test_with_discord_config_creates_sender(self):
        from chimera_v12.alerts.dispatcher import build_dispatcher
        d = build_dispatcher({
            "discord_webhook_url": "https://discord.com/api/webhooks/fake/url"
        })
        assert len(d.senders) == 1

    def test_with_both_configs_creates_two_senders(self):
        from chimera_v12.alerts.dispatcher import build_dispatcher
        d = build_dispatcher({
            "telegram_bot_token":  "fake_token",
            "telegram_chat_id":    "12345",
            "discord_webhook_url": "https://discord.com/api/webhooks/fake/url"
        })
        assert len(d.senders) == 2
