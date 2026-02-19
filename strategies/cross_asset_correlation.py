#!/usr/bin/env python3
"""
Cross-Asset Correlation Strategy — candle-based interface.

Uses ETH as a leading indicator for altcoin follow-through.
ETH ROC is queried from the DB (tick data); altcoin data comes
from context_candles.

When ETH shows strong momentum (>0.8% ROC over recent bars),
altcoins tend to follow within 2-4 bars. Only signal when the
altcoin hasn't already moved.
"""
import os
import sqlite3
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy, Signal

DB_PATH = os.getenv("BLOFIN_DB_PATH", "data/blofin_monitor.db")
# Lowered from 0.8 → 0.4: average 5m ETH move is ~0.11%, 0.8% was too rare
ETH_ROC_THRESHOLD = float(os.getenv("CROSS_ASSET_ETH_ROC_PCT", "0.4"))
ETH_ROC_PERIOD = int(os.getenv("CROSS_ASSET_ETH_ROC_PERIOD", "3"))
LAG_BARS = int(os.getenv("CROSS_ASSET_LAG_BARS", "2"))


class CrossAssetCorrelationStrategy(BaseStrategy):
    """
    ETH-lead altcoin follow-through strategy — candle interface.

    When ETH makes a significant move, altcoins tend to follow within
    2-4 candles. Trades the lag only if the altcoin hasn't started moving.
    """

    name = "cross_asset_correlation"
    version = "2.0"
    description = "ETH as leading indicator for altcoin follow-through (candle-based)"

    # ETH-USDT is used as the signal source so we skip it.
    # BTC-USDT removed from skip list — backtester only tests BTC+ETH and we
    # want to trade BTC using ETH as a leading indicator (they're highly correlated).
    SKIP_SYMBOLS = {"ETH-USDT"}

    def __init__(self):
        self.eth_roc_threshold = ETH_ROC_THRESHOLD
        self.eth_roc_period = ETH_ROC_PERIOD
        self.lag_bars = LAG_BARS

    def _get_eth_recent_roc(self, current_ts_ms: int) -> Optional[float]:
        """Fetch ETH ROC from DB ticks (30-minute window, lagged by lag_bars * 5m)."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            ref_ts = current_ts_ms - (self.lag_bars * 5 * 60 * 1000)
            start_ts = ref_ts - (30 * 60 * 1000)
            c.execute(
                "SELECT ts_ms, price FROM ticks "
                "WHERE symbol='ETH-USDT' AND ts_ms>=? AND ts_ms<=? "
                "ORDER BY ts_ms ASC",
                (start_ts, ref_ts)
            )
            rows = c.fetchall()
            conn.close()

            if len(rows) < 10:
                return None

            first_price = rows[0][1]
            last_price = rows[-1][1]
            if first_price <= 0:
                return None

            return (last_price - first_price) / first_price * 100
        except Exception:
            return None

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if symbol in self.SKIP_SYMBOLS:
            return None

        if len(context_candles) < 10:
            return None

        current = context_candles[-1]
        ts_ms = current['ts_ms']

        # Check if altcoin has already moved significantly (avoid chasing)
        lookback = min(5, len(context_candles))
        recent_closes = [c['close'] for c in context_candles[-lookback:]]
        alt_first = recent_closes[0]
        alt_last = current['close']
        alt_roc = ((alt_last - alt_first) / alt_first * 100) if alt_first > 0 else 0
        # Raised from 0.5 → 2.0: BTC average 5m move is ~0.11%, 0.5% was too restrictive
        if abs(alt_roc) > 2.0:
            return None  # Altcoin already moved — too late

        eth_roc = self._get_eth_recent_roc(ts_ms)
        if eth_roc is None:
            return None

        if eth_roc >= self.eth_roc_threshold:
            confidence = min(0.85, 0.50 + abs(eth_roc) / (self.eth_roc_threshold * 4))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'eth_roc_pct': round(eth_roc, 4),
                'lag_bars': self.lag_bars,
                'alt_roc_pct': round(alt_roc, 4),
            }
        elif eth_roc <= -self.eth_roc_threshold:
            confidence = min(0.85, 0.50 + abs(eth_roc) / (self.eth_roc_threshold * 4))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'eth_roc_pct': round(eth_roc, 4),
                'lag_bars': self.lag_bars,
                'alt_roc_pct': round(alt_roc, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "eth_roc_threshold": self.eth_roc_threshold,
            "eth_roc_period": self.eth_roc_period,
            "lag_bars": self.lag_bars,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "eth_roc_threshold" in params:
            self.eth_roc_threshold = float(params["eth_roc_threshold"])
        if "eth_roc_period" in params:
            self.eth_roc_period = int(params["eth_roc_period"])
        if "lag_bars" in params:
            self.lag_bars = int(params["lag_bars"])
