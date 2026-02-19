#!/usr/bin/env python3
"""
RSI Divergence Strategy — candle-based interface.

Two modes:
  1. Standard RSI overbought/oversold (RSI < 30 or > 70)
  2. Classic RSI divergence: price makes new extreme but RSI doesn't confirm
     (bullish divergence = price lower low, RSI higher low → BUY)
     (bearish divergence = price higher high, RSI lower high → SELL)
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class RSIDivergenceStrategy(BaseStrategy):
    """RSI overbought/oversold signals with optional divergence detection."""

    name = "rsi_divergence"
    version = "2.0"
    description = "RSI overbought/oversold with classic price-RSI divergence detection"

    def __init__(self):
        self.min_candles = 20
        self.rsi_period = 14
        self.oversold = 30
        self.overbought = 70
        self.divergence_lookback = 10   # candles to look back for divergence pivot

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calc_rsi_series(self, closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Return full RSI series (same length as closes, None for initial values)."""
        result: List[Optional[float]] = [None] * len(closes)
        if len(closes) < period + 1:
            return result

        gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
        losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]

        ag = sum(gains[:period]) / period
        al = sum(losses[:period]) / period

        def rsi_from(ag, al):
            if al == 0:
                return 100.0 if ag > 0 else 50.0
            return 100.0 - 100.0 / (1.0 + ag / al)

        result[period] = rsi_from(ag, al)

        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
            result[i + 1] = rsi_from(ag, al)

        return result

    def _check_divergence(
        self,
        closes: List[float],
        rsi_series: List[Optional[float]],
        lookback: int,
    ):
        """
        Detect bullish/bearish divergence in the last `lookback` candles.
        Returns 'BUY', 'SELL', or None.
        """
        n = len(closes)
        if n < lookback + 2:
            return None

        recent_closes = closes[-lookback:]
        recent_rsi = rsi_series[-lookback:]

        # Filter to positions where RSI is valid
        valid = [(i, c, r) for i, (c, r) in enumerate(zip(recent_closes, recent_rsi)) if r is not None]
        if len(valid) < 4:
            return None

        # Current (last point)
        _, cur_close, cur_rsi = valid[-1]
        # Mid-point pivot (a few bars back)
        pivot_idx = len(valid) // 2
        _, piv_close, piv_rsi = valid[pivot_idx]

        # Bullish divergence: price lower, RSI higher
        if cur_close < piv_close and cur_rsi > piv_rsi and cur_rsi < 45:
            return 'BUY'

        # Bearish divergence: price higher, RSI lower
        if cur_close > piv_close and cur_rsi < piv_rsi and cur_rsi > 55:
            return 'SELL'

        return None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        rsi_series = self._calc_rsi_series(closes, self.rsi_period)
        current_rsi = rsi_series[-1]

        if current_rsi is None:
            return None

        # --- Standard overbought/oversold ---
        if current_rsi <= self.oversold:
            confidence = min(0.80, 0.55 + (self.oversold - current_rsi) / self.oversold * 0.25)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(current_rsi, 2),
                'mode': 'oversold',
            }

        if current_rsi >= self.overbought:
            confidence = min(0.80, 0.55 + (current_rsi - self.overbought) / (100 - self.overbought) * 0.25)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(current_rsi, 2),
                'mode': 'overbought',
            }

        # --- Divergence detection (only in middle RSI zone 35-65) ---
        if 35 <= current_rsi <= 65:
            div = self._check_divergence(closes, rsi_series, self.divergence_lookback)
            if div:
                return {
                    'signal': div,
                    'strategy': self.name,
                    'confidence': 0.65,
                    'rsi': round(current_rsi, 2),
                    'mode': 'divergence',
                }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'rsi_period': 21,
            'oversold': 25,
            'overbought': 75,
            'divergence_lookback': 20,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'rsi_period' in params:
            self.rsi_period = int(params['rsi_period'])
        if 'oversold' in params:
            self.oversold = float(params['oversold'])
        if 'overbought' in params:
            self.overbought = float(params['overbought'])
        if 'divergence_lookback' in params:
            self.divergence_lookback = int(params['divergence_lookback'])
