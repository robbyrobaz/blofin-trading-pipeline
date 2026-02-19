#!/usr/bin/env python3
"""
MACD Divergence Strategy — candle-based interface.
Detects MACD histogram divergence from price action using candle close prices.
"""
import os
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class MACDDivergenceStrategy(BaseStrategy):
    """Detects MACD histogram divergence from price action — candle interface."""

    name = "macd_divergence"
    version = "2.0"
    description = "Detects MACD histogram divergence from price movement (candle-based)"

    def __init__(self):
        self.fast_period = int(os.getenv("MACD_FAST_PERIOD", "12"))
        self.slow_period = int(os.getenv("MACD_SLOW_PERIOD", "26"))
        self.signal_period = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))
        self.min_divergence_pct = float(os.getenv("MACD_MIN_DIVERGENCE_PCT", "0.3"))

    def _ema_series(self, prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return []
        k = 2.0 / (period + 1)
        ema = [sum(prices[:period]) / period]
        for p in prices[period:]:
            ema.append(p * k + ema[-1] * (1 - k))
        return ema

    def _calc_macd_histogram(self, closes: List[float]) -> Optional[List[float]]:
        """Return MACD histogram series aligned to the latest candles."""
        fast = self._ema_series(closes, self.fast_period)
        slow = self._ema_series(closes, self.slow_period)
        if len(fast) < 2 or len(slow) < 2:
            return None

        # Align: slow_ema[i] aligns with closes[slow_period-1+i]
        # fast_ema[i] aligns with closes[fast_period-1+i]
        # slow_ema[i] aligns with fast_ema[i + (slow_period - fast_period)]
        offset = self.slow_period - self.fast_period
        macd_line = [fast[i + offset] - slow[i] for i in range(len(slow))]

        if len(macd_line) < self.signal_period + 2:
            return None

        signal_line = self._ema_series(macd_line, self.signal_period)
        if len(signal_line) < 2:
            return None

        return [macd_line[i + (self.signal_period - 1)] - signal_line[i]
                for i in range(len(signal_line))]

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        min_needed = self.slow_period + self.signal_period + 15
        if len(context_candles) < min_needed:
            return None

        closes = [c['close'] for c in context_candles]

        histogram = self._calc_macd_histogram(closes)
        if histogram is None or len(histogram) < 2:
            return None

        hist_now = histogram[-1]
        hist_prev = histogram[-6] if len(histogram) >= 6 else histogram[0]

        recent_closes = closes[-10:]
        price_trend = ((recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
                       if recent_closes[0] > 0 else 0)
        histogram_change = hist_now - hist_prev

        # Bullish divergence: price making lower lows, MACD histogram rising
        if price_trend < -self.min_divergence_pct and hist_now > hist_prev:
            confidence = min(0.85, 0.65 + abs(histogram_change) * 5)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'histogram': round(hist_now, 6),
                'price_trend_pct': round(price_trend, 4),
                'divergence': 'bullish',
            }

        # Bearish divergence: price making higher highs, MACD histogram falling
        elif price_trend > self.min_divergence_pct and hist_now < hist_prev:
            confidence = min(0.85, 0.65 + abs(histogram_change) * 5)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'histogram': round(hist_now, 6),
                'price_trend_pct': round(price_trend, 4),
                'divergence': 'bearish',
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "min_divergence_pct": self.min_divergence_pct,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "fast_period" in params:
            self.fast_period = int(params["fast_period"])
        if "slow_period" in params:
            self.slow_period = int(params["slow_period"])
        if "signal_period" in params:
            self.signal_period = int(params["signal_period"])
        if "min_divergence_pct" in params:
            self.min_divergence_pct = float(params["min_divergence_pct"])
