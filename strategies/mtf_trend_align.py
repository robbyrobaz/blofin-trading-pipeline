#!/usr/bin/env python3
"""
Multi-timeframe trend alignment strategy.
5m RSI for entry signal + 1h EMA for trend filter.
BUY when RSI is oversold and price is above the 1h EMA (uptrend confirmed).
SELL when RSI is overbought and price is below the 1h EMA (downtrend confirmed).
"""
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class MTFTrendAlignStrategy(BaseStrategy):
    """Multi-timeframe trend alignment: 5m RSI entry + 1h EMA trend filter."""

    name = "mtf_trend_align"
    version = "1.0"
    description = "Multi-timeframe trend alignment: 5m RSI + 1h EMA trend filter"

    def __init__(self):
        # 5-minute window for RSI signal
        self.rsi_window_seconds = int(os.getenv("MTF_TREND_RSI_WINDOW_SECONDS", "300"))
        self.rsi_oversold = float(os.getenv("MTF_TREND_RSI_OVERSOLD", "30"))
        self.rsi_overbought = float(os.getenv("MTF_TREND_RSI_OVERBOUGHT", "70"))
        # 1-hour window for EMA trend filter
        self.ema_window_seconds = int(os.getenv("MTF_TREND_EMA_WINDOW_SECONDS", "3600"))
        self.min_rsi_periods = int(os.getenv("MTF_TREND_MIN_RSI_PERIODS", "10"))

    def _compute_rsi(self, prices: List[float]) -> Optional[float]:
        """Compute RSI from a list of prices. Returns None if insufficient data."""
        if len(prices) < 2:
            return None
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            return 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain > 0:
            return 100.0
        return 50.0

    def _compute_ema(self, prices: List[float]) -> Optional[float]:
        """Compute EMA for the full price list. Returns None if empty."""
        if not prices:
            return None
        # Use simple multiplier: 2/(n+1)
        n = len(prices)
        k = 2.0 / (n + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]],
    ) -> Optional[Signal]:
        # 5m window for RSI
        rsi_window = self._slice_window(prices, ts_ms, self.rsi_window_seconds)
        if len(rsi_window) < self.min_rsi_periods:
            return None

        # 1h window for EMA trend
        ema_window = self._slice_window(prices, ts_ms, self.ema_window_seconds)
        if len(ema_window) < 2:
            return None

        rsi_prices = [p for _, p in rsi_window]
        ema_prices = [p for _, p in ema_window]

        rsi = self._compute_rsi(rsi_prices)
        ema = self._compute_ema(ema_prices)
        if rsi is None or ema is None or ema <= 0:
            return None

        above_ema = price > ema
        below_ema = price < ema

        if rsi <= self.rsi_oversold and above_ema:
            confidence = min(0.85, max(0.55, (self.rsi_oversold - rsi) / self.rsi_oversold))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "rsi_5m": round(rsi, 2),
                    "ema_1h": round(ema, 6),
                    "price_vs_ema_pct": round((price - ema) / ema * 100, 4),
                    "rsi_window_s": self.rsi_window_seconds,
                    "ema_window_s": self.ema_window_seconds,
                },
            )

        if rsi >= self.rsi_overbought and below_ema:
            confidence = min(0.85, max(0.55, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "rsi_5m": round(rsi, 2),
                    "ema_1h": round(ema, 6),
                    "price_vs_ema_pct": round((price - ema) / ema * 100, 4),
                    "rsi_window_s": self.rsi_window_seconds,
                    "ema_window_s": self.ema_window_seconds,
                },
            )

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "rsi_window_seconds": self.rsi_window_seconds,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "ema_window_seconds": self.ema_window_seconds,
            "min_rsi_periods": self.min_rsi_periods,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "rsi_window_seconds" in params:
            self.rsi_window_seconds = int(params["rsi_window_seconds"])
        if "rsi_oversold" in params:
            self.rsi_oversold = float(params["rsi_oversold"])
        if "rsi_overbought" in params:
            self.rsi_overbought = float(params["rsi_overbought"])
        if "ema_window_seconds" in params:
            self.ema_window_seconds = int(params["ema_window_seconds"])
        if "min_rsi_periods" in params:
            self.min_rsi_periods = int(params["min_rsi_periods"])
