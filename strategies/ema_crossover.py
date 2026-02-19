#!/usr/bin/env python3
"""
EMA Crossover Strategy — candle-based interface.
Detects 9/21 EMA crossover signals on candle close prices.
"""
import os
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class EMACrossoverStrategy(BaseStrategy):
    """Detects EMA crossover signals (9/21 EMA by default) — candle interface."""

    name = "ema_crossover"
    version = "2.0"
    description = "Detects exponential moving average crossover signals (candle-based)"

    def __init__(self):
        self.fast_period = int(os.getenv("EMA_FAST_PERIOD", "9"))
        self.slow_period = int(os.getenv("EMA_SLOW_PERIOD", "21"))
        # Lowered from 0.15 → 0.02: BTC 5m avg move is ~0.11%, 0.15% was unreachable
        self.min_separation_pct = float(os.getenv("EMA_MIN_SEPARATION_PCT", "0.02"))

    def _ema_series(self, prices: List[float], period: int) -> List[float]:
        """Return full EMA series (length = len(prices) - period + 1)."""
        if len(prices) < period:
            return []
        multiplier = 2.0 / (period + 1)
        ema = [sum(prices[:period]) / period]
        for price in prices[period:]:
            ema.append(price * multiplier + ema[-1] * (1 - multiplier))
        return ema

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.slow_period + 5:
            return None

        closes = [c['close'] for c in context_candles]

        fast_ema = self._ema_series(closes, self.fast_period)
        slow_ema = self._ema_series(closes, self.slow_period)

        if len(fast_ema) < 2 or len(slow_ema) < 2:
            return None

        # Both series end at the same candle; compare last two values of each
        fast_now, fast_prev = fast_ema[-1], fast_ema[-2]
        slow_now, slow_prev = slow_ema[-1], slow_ema[-2]

        if slow_now == 0:
            return None

        separation_pct = abs((fast_now - slow_now) / slow_now) * 100.0

        # Bullish crossover: fast crosses above slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            if separation_pct >= self.min_separation_pct:
                confidence = min(0.85, 0.65 + (separation_pct / 2.0))
                return {
                    'signal': 'BUY',
                    'strategy': self.name,
                    'confidence': round(confidence, 4),
                    'fast_ema': round(fast_now, 6),
                    'slow_ema': round(slow_now, 6),
                    'separation_pct': round(separation_pct, 4),
                }

        # Bearish crossover: fast crosses below slow
        elif fast_prev >= slow_prev and fast_now < slow_now:
            if separation_pct >= self.min_separation_pct:
                confidence = min(0.85, 0.65 + (separation_pct / 2.0))
                return {
                    'signal': 'SELL',
                    'strategy': self.name,
                    'confidence': round(confidence, 4),
                    'fast_ema': round(fast_now, 6),
                    'slow_ema': round(slow_now, 6),
                    'separation_pct': round(separation_pct, 4),
                }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "min_separation_pct": self.min_separation_pct,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "fast_period" in params:
            self.fast_period = int(params["fast_period"])
        if "slow_period" in params:
            self.slow_period = int(params["slow_period"])
        if "min_separation_pct" in params:
            self.min_separation_pct = float(params["min_separation_pct"])
