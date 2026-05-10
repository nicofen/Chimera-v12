"""
chimera/agents/data_agent.py
Data Agent — manages all market data ingestion.

WebSocket streams: Alpaca (stocks, forex, futures), crypto exchange WS.
REST polling:      Whale Alert, CoinMarketCap, Dune Analytics, Finviz, Stocktwits.

Writes normalized OHLCV + metadata bars into state.market.{sector}.
"""

import asyncio
import json
import time
from typing import Any

import aiohttp
import websockets

from chimera_v12.utils.state import SharedState
from chimera_v12.utils.logger import setup_logger

log = setup_logger("data_agent")

ALPACA_WS_URL    = "wss://stream.data.alpaca.markets/v2/iex"
ALPACA_CRYPTO_WS = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"


class DataAgent:
    """
    Runs all ingestor coroutines concurrently.
    Each ingestor writes directly into the shared state's market dicts.
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state  = state
        self.config = config

    async def run(self) -> None:
        log.info("DataAgent started.")
        await asyncio.gather(
            self._alpaca_stocks_ws(),
            self._alpaca_crypto_ws(),
            self._whale_alert_poll(),
            self._finviz_poll(),
            self._dune_poll(),
            self._alpaca_futures_poll(),
        )

    # ── Alpaca Stocks WebSocket ───────────────────────────────────────────────

    async def _alpaca_stocks_ws(self) -> None:
        symbols = self.config.get("stock_symbols", ["AAPL", "TSLA", "GME"])
        headers = {
            "APCA-API-KEY-ID":     self.config["alpaca_key"],
            "APCA-API-SECRET-KEY": self.config["alpaca_secret"],
        }
        while True:
            try:
                async with websockets.connect(ALPACA_WS_URL, extra_headers=headers) as ws:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "bars":   symbols,
                        "trades": symbols,
                    }))
                    async for raw in ws:
                        msgs = json.loads(raw)
                        for msg in msgs:
                            if msg.get("T") == "b":   # bar message
                                sym = msg["S"]
                                bars = self.state.market.stocks.setdefault(sym, {
                                    "close": [], "high": [], "low": [], "volume": []
                                })
                                bars["close"].append(float(msg["c"]))
                                bars["high"].append(float(msg["h"]))
                                bars["low"].append(float(msg["l"]))
                                bars["volume"].append(float(msg["v"]))
                                # Keep a rolling 500-bar window
                                for k in bars:
                                    bars[k] = bars[k][-500:]
            except Exception as e:
                log.warning(f"Stocks WS error: {e} — reconnecting in 5s")
                await asyncio.sleep(5)

    # ── Alpaca Crypto WebSocket ───────────────────────────────────────────────

    async def _alpaca_crypto_ws(self) -> None:
        symbols = self.config.get("crypto_symbols", ["BTC/USD", "ETH/USD", "SOL/USD"])
        headers = {
            "APCA-API-KEY-ID":     self.config["alpaca_key"],
            "APCA-API-SECRET-KEY": self.config["alpaca_secret"],
        }
        while True:
            try:
                async with websockets.connect(ALPACA_CRYPTO_WS, extra_headers=headers) as ws:
                    await ws.send(json.dumps({"action": "subscribe", "bars": symbols}))
                    async for raw in ws:
                        msgs = json.loads(raw)
                        for msg in msgs:
                            if msg.get("T") == "b":
                                sym = msg["S"]
                                bars = self.state.market.crypto.setdefault(sym, {
                                    "close": [], "high": [], "low": [], "volume": []
                                })
                                bars["close"].append(float(msg["c"]))
                                bars["high"].append(float(msg["h"]))
                                bars["low"].append(float(msg["l"]))
                                bars["volume"].append(float(msg["v"]))
                                for k in bars:
                                    bars[k] = bars[k][-500:]
            except Exception as e:
                log.warning(f"Crypto WS error: {e} — reconnecting in 5s")
                await asyncio.sleep(5)

    # ── Whale Alert REST poll ─────────────────────────────────────────────────

    async def _whale_alert_poll(self) -> None:
        url      = "https://api.whale-alert.io/v1/transactions"
        api_key  = self.config.get("whale_alert_key", "")
        interval = self.config.get("whale_poll_seconds", 60)

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    params = {
                        "api_key":   api_key,
                        "cursor":    int(time.time()) - interval,
                        "min_value": 1_000_000,
                        "limit":     100,
                    }
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        data = await resp.json()
                        inflow = sum(
                            tx["amount_usd"]
                            for tx in data.get("transactions", [])
                            if tx.get("to", {}).get("owner_type") == "exchange"
                            and tx.get("symbol", "").upper() == "BTC"
                        )
                        self.state.market.crypto["btc_exchange_inflow"] = inflow
                        log.debug(f"BTC exchange inflow (last {interval}s): ${inflow:,.0f}")
                except Exception as e:
                    log.warning(f"Whale Alert poll error: {e}")
                await asyncio.sleep(interval)

    # ── Finviz screener poll (short interest + RVOL) ─────────────────────────

    async def _finviz_poll(self) -> None:
        """
        Uses finviz Python library (pip install finviz) to screen for
        high short-interest, high RVOL candidates every 5 minutes.
        """
        try:
            import finviz
        except ImportError:
            log.warning("finviz not installed — skipping stock screener.")
            return

        interval = self.config.get("finviz_poll_seconds", 300)
        filters  = ["sh_short_o20", "ta_relvol_o3"]   # SI > 20%, RVOL > 3

        while True:
            try:
                results = finviz.get_screener(filters=filters, table="Performance")
                for row in results[:20]:
                    sym = row.get("Ticker", "")
                    if sym in self.state.market.stocks:
                        self.state.market.stocks[sym]["short_interest"] = (
                            float(str(row.get("Short Float", "0")).strip("%")) / 100
                        )
                        self.state.market.stocks[sym]["rvol"] = float(row.get("Rel Volume", 1.0))
            except Exception as e:
                log.warning(f"Finviz poll error: {e}")
            await asyncio.sleep(interval)

    # ── Dune Analytics poll (Solana memecoin volume) ─────────────────────────

    async def _dune_poll(self) -> None:
        dune_key = self.config.get("dune_api_key", "")
        query_id = self.config.get("dune_memecoin_query_id", "3152691")  # memecoin wars
        interval = self.config.get("dune_poll_seconds", 300)
        url      = f"https://api.dune.com/api/v1/query/{query_id}/results"

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    headers = {"X-Dune-API-Key": dune_key}
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data  = await resp.json()
                        rows  = data.get("result", {}).get("rows", [])
                        vol   = sum(r.get("volume_usd", 0) for r in rows[:10])
                        spike = vol > self.config.get("sol_memecoin_spike_threshold", 50_000_000)
                        self.state.market.crypto["sol_memecoin_vol_spike"] = spike
                        log.debug(f"Sol memecoin vol: ${vol:,.0f} spike={spike}")
                except Exception as e:
                    log.warning(f"Dune poll error: {e}")
                await asyncio.sleep(interval)

    # ── Alpaca Futures REST poll (ES1!) ───────────────────────────────────────

    async def _alpaca_futures_poll(self) -> None:
        contracts = self.config.get("futures_contracts", ["ES1!"])
        url_base  = "https://data.alpaca.markets/v2/stocks/{sym}/bars"
        headers   = {
            "APCA-API-KEY-ID":     self.config["alpaca_key"],
            "APCA-API-SECRET-KEY": self.config["alpaca_secret"],
        }
        interval = self.config.get("futures_poll_seconds", 60)

        async with aiohttp.ClientSession() as session:
            while True:
                for sym in contracts:
                    try:
                        async with session.get(
                            url_base.format(sym=sym),
                            headers=headers,
                            params={"timeframe": "5Min", "limit": 200},
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            data = await resp.json()
                            bars_raw = data.get("bars", [])
                            if bars_raw:
                                self.state.market.futures[sym] = {
                                    "close":  [b["c"] for b in bars_raw],
                                    "high":   [b["h"] for b in bars_raw],
                                    "low":    [b["l"] for b in bars_raw],
                                    "volume": [b["v"] for b in bars_raw],
                                }
                    except Exception as e:
                        log.warning(f"Futures poll error for {sym}: {e}")
                await asyncio.sleep(interval)
