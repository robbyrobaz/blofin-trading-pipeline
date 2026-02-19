#!/usr/bin/env python3
"""
Pivot-Point Support/Resistance Strategy — candle-based interface.

Uses standard pivot points calculated from previous candle's H/L/C:
  Pivot = (H + L + C) / 3
  R1 = 2 * Pivot - L
  S1 = 2 * Pivot - H
  R2 = Pivot + (H - L)
  S2 = Pivot - (H - L)

Also detects price approaching and bouncing from these levels.
BUY when price touches/penetrates S1/S2 then closes back above.
SELL when price touches/penetrates R1/R2 then closes back below.
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class SupportResistanceStrategy(BaseStrategy):
    """Pivot point support/resistance bounce detection."""

    name = "support_resistance"
    version = "2.0"
    description = "Pivot-point S/R levels from prior candles with bounce detection"

    def __init__(self):
        self.min_candles = 20
        self.pivot_lookback = 5         # candles to average for pivot calculation
        self.touch_pct = 0.15          # price within 0.15% of level = "touching"
        self.vol_sma_period = 20
        self.vol_confirm_ratio = 1.1    # slight volume increase confirms bounce

    # ------------------------------------------------------------------
    # Pivot calculation
    # ------------------------------------------------------------------

    def _calc_pivots(self, candles: List[dict]) -> dict:
        """Standard pivot points from the lookback window."""
        h = max(c['high'] for c in candles)
        l = min(c['low'] for c in candles)
        c_avg = sum(c['close'] for c in candles) / len(candles)

        pivot = (h + l + c_avg) / 3.0
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)

        return {'pivot': pivot, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2}

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        # Pivot from previous candles (not current)
        pivot_candles = context_candles[-self.pivot_lookback - 1:-1]
        if not pivot_candles:
            return None

        pivots = self._calc_pivots(pivot_candles)
        current = context_candles[-1]
        prev = context_candles[-2]
        close = current['close']
        prev_close = prev['close']

        # Volume filter
        volumes = [c['volume'] for c in context_candles]
        vol_sma = sum(volumes[-self.vol_sma_period - 1:-1]) / self.vol_sma_period
        vol_ratio = current['volume'] / vol_sma if vol_sma > 0 else 1.0

        touch_pct = self.touch_pct / 100.0

        # Check for support bounce (S1, S2)
        for level_name, level in [('s1', pivots['s1']), ('s2', pivots['s2'])]:
            if level <= 0:
                continue
            # Previous candle touched/penetrated support
            prev_touched = prev_close <= level * (1 + touch_pct)
            # Current candle closed above support
            bounced = close > level and prev_close <= level * (1 + touch_pct)
            if prev_touched and close > level:
                if vol_ratio >= self.vol_confirm_ratio:
                    confidence = 0.72 if level_name == 's1' else 0.78
                    return {
                        'signal': 'BUY',
                        'strategy': self.name,
                        'confidence': confidence,
                        'level': round(level, 6),
                        'level_name': level_name,
                        'pivot': round(pivots['pivot'], 6),
                        'vol_ratio': round(vol_ratio, 4),
                    }

        # Check for resistance rejection (R1, R2)
        for level_name, level in [('r1', pivots['r1']), ('r2', pivots['r2'])]:
            if level <= 0:
                continue
            # Previous candle touched/penetrated resistance
            prev_touched = prev_close >= level * (1 - touch_pct)
            # Current candle closed below resistance
            if prev_touched and close < level:
                if vol_ratio >= self.vol_confirm_ratio:
                    confidence = 0.72 if level_name == 'r1' else 0.78
                    return {
                        'signal': 'SELL',
                        'strategy': self.name,
                        'confidence': confidence,
                        'level': round(level, 6),
                        'level_name': level_name,
                        'pivot': round(pivots['pivot'], 6),
                        'vol_ratio': round(vol_ratio, 4),
                    }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'pivot_lookback': self.pivot_lookback,
            'touch_pct': self.touch_pct,
            'vol_confirm_ratio': self.vol_confirm_ratio,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'pivot_lookback' in params:
            self.pivot_lookback = int(params['pivot_lookback'])
        if 'touch_pct' in params:
            self.touch_pct = float(params['touch_pct'])
        if 'vol_confirm_ratio' in params:
            self.vol_confirm_ratio = float(params['vol_confirm_ratio'])
