#!/usr/bin/env python3
"""
MACD Momentum Strategy — candle-based interface.

Uses MACD (12/26/9 EMA) on close prices.
  BUY when MACD line crosses above signal line (bullish crossover)
  SELL when MACD line crosses below signal line (bearish crossover)

Additionally confirms with short-term EMA slope direction.
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """MACD crossover momentum strategy on candle close prices."""

    name = "momentum"
    version = "2.0"
    description = "MACD (12/26/9) crossover momentum with EMA slope confirmation"

    def __init__(self):
        self.min_candles = 30           # need enough for 26-period EMA
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.ema_slope_period = 5       # short EMA slope for trend confirmation

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _ema(self, values: List[float], period: int) -> List[float]:
        """Return EMA series for the given values and period."""
        if len(values) < period:
            return []
        k = 2.0 / (period + 1.0)
        ema_vals = [sum(values[:period]) / period]  # seed with SMA
        for v in values[period:]:
            ema_vals.append(v * k + ema_vals[-1] * (1.0 - k))
        return ema_vals

    def _calc_macd(self, closes: List[float]):
        """Returns (macd_line[-2], macd_line[-1], signal_line[-2], signal_line[-1])."""
        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)

        # Align: slow_ema is shorter; fast_ema has len(closes)-fast_period+1 values
        fast_len = len(fast_ema)
        slow_len = len(slow_ema)
        if fast_len < 2 or slow_len < 2:
            return None

        # Offset to align: fast_ema[i] aligns with slow_ema[i - (slow_period - fast_period)]
        offset = (self.slow_period - self.fast_period)
        macd_line = [fast_ema[i + offset] - slow_ema[i] for i in range(min(slow_len, fast_len - offset))]

        if len(macd_line) < self.signal_period + 2:
            return None

        signal_line = self._ema(macd_line, self.signal_period)
        if len(signal_line) < 2:
            return None

        return macd_line[-1], macd_line[-2], signal_line[-1], signal_line[-2]

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]

        result = self._calc_macd(closes)
        if result is None:
            return None

        macd_now, macd_prev, signal_now, signal_prev = result
        histogram_now = macd_now - signal_now
        histogram_prev = macd_prev - signal_prev

        # Bullish crossover: MACD crosses above signal
        crossed_up = histogram_prev < 0 and histogram_now >= 0
        # Bearish crossover: MACD crosses below signal
        crossed_down = histogram_prev > 0 and histogram_now <= 0

        if not crossed_up and not crossed_down:
            return None

        # EMA slope confirmation
        ema_short = self._ema(closes, self.ema_slope_period)
        if len(ema_short) >= 2:
            slope_up = ema_short[-1] > ema_short[-2]
        else:
            slope_up = True  # neutral if can't compute

        if crossed_up:
            confidence = min(0.80, 0.60 + abs(histogram_now) / max(abs(macd_now), 0.001) * 0.15)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'macd': round(macd_now, 6),
                'signal_line': round(signal_now, 6),
                'histogram': round(histogram_now, 6),
                'ema_slope_up': slope_up,
            }

        if crossed_down:
            confidence = min(0.80, 0.60 + abs(histogram_now) / max(abs(macd_now), 0.001) * 0.15)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'macd': round(macd_now, 6),
                'signal_line': round(signal_now, 6),
                'histogram': round(histogram_now, 6),
                'ema_slope_up': slope_up,
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'min_candles': self.min_candles,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'fast_period' in params:
            self.fast_period = int(params['fast_period'])
        if 'slow_period' in params:
            self.slow_period = int(params['slow_period'])
        if 'signal_period' in params:
            self.signal_period = int(params['signal_period'])
