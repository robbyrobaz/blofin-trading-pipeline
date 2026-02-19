#!/usr/bin/env python3
"""
Cross-Asset Correlation Strategy
Uses ETH as a leading indicator for altcoin follow-through.

Logic:
  - When ETH shows strong momentum (>0.8% ROC over 3 bars on 5m),
    altcoins tend to follow within 2-4 bars.
  - Only signal when altcoin hasn't already moved (avoid chasing).
  - ETH data is loaded fresh per signal to avoid stale caches.

Reference: new_strategies_backtest.py → strategy_cross_asset_correlation
"""

import os
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, Signal

DB_PATH = os.getenv("BLOFIN_DB_PATH", "data/blofin_monitor.db")
ETH_ROC_THRESHOLD = float(os.getenv("CROSS_ASSET_ETH_ROC_PCT", "0.8"))
ETH_ROC_PERIOD = int(os.getenv("CROSS_ASSET_ETH_ROC_PERIOD", "3"))   # bars
LAG_BARS = int(os.getenv("CROSS_ASSET_LAG_BARS", "2"))


class CrossAssetCorrelationStrategy(BaseStrategy):
    """
    ETH-lead altcoin follow-through strategy.

    When ETH makes a significant move (ROC > threshold on 5m candles),
    altcoins tend to follow within 2-4 candles. This strategy trades
    that lag, only if the altcoin hasn't already started moving.
    """

    name = "cross_asset_correlation"
    version = "1.0"
    description = "ETH as leading indicator for altcoin follow-through"

    # Symbols that don't make sense for this strategy
    SKIP_SYMBOLS = {"ETH-USDT", "BTC-USDT"}

    def __init__(self):
        self.eth_roc_threshold = ETH_ROC_THRESHOLD
        self.eth_roc_period = ETH_ROC_PERIOD
        self.lag_bars = LAG_BARS
        self._eth_price_cache: List[Tuple[int, float]] = []
        self._eth_cache_ts: int = 0
        self._cache_ttl_ms = 60_000  # Refresh ETH cache every 60s

    def _get_eth_recent_roc(self, ts_ms: int) -> Optional[float]:
        """
        Fetch ETH 5m candle ROC using recent price history.
        Uses the passed prices window from the ingestor if ETH is available,
        otherwise queries DB directly.
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            start_ts = ts_ms - (30 * 60 * 1000)  # 30 minutes back
            c.execute(
                "SELECT ts_ms, price FROM ticks WHERE symbol='ETH-USDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC",
                (start_ts, ts_ms)
            )
            rows = c.fetchall()
            conn.close()

            if len(rows) < 10:
                return None

            # Get price at start and at lag offset
            first_price = rows[0][1]
            last_price = rows[-1][1]
            if first_price <= 0:
                return None

            # ROC over the window
            roc = (last_price - first_price) / first_price * 100
            return roc

        except Exception:
            return None

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:

        if symbol in self.SKIP_SYMBOLS:
            return None

        if len(prices) < 10:
            return None

        # Check if altcoin has already moved significantly (avoid chasing)
        window_5m = self._slice_window(prices, ts_ms, 5 * 60)
        if len(window_5m) < 2:
            return None

        alt_first = window_5m[0][1]
        alt_roc = ((price - alt_first) / alt_first * 100) if alt_first > 0 else 0
        if abs(alt_roc) > 0.5:
            return None  # Altcoin already moved — too late

        # Check ETH momentum
        eth_roc = self._get_eth_recent_roc(ts_ms - self.lag_bars * 5 * 60 * 1000)
        if eth_roc is None:
            return None

        if eth_roc >= self.eth_roc_threshold:
            confidence = min(0.85, 0.50 + abs(eth_roc) / (self.eth_roc_threshold * 4))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "eth_roc_pct": round(eth_roc, 4),
                    "lag_bars": self.lag_bars,
                    "alt_roc_pct": round(alt_roc, 4),
                }
            )
        elif eth_roc <= -self.eth_roc_threshold:
            confidence = min(0.85, 0.50 + abs(eth_roc) / (self.eth_roc_threshold * 4))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "eth_roc_pct": round(eth_roc, 4),
                    "lag_bars": self.lag_bars,
                    "alt_roc_pct": round(alt_roc, 4),
                }
            )

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
