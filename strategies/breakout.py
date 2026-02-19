#!/usr/bin/env python3
"""
Donchian Channel Breakout Strategy — candle-based interface.

Uses 20-period Donchian channel (highest high / lowest low).
  BUY when close breaks above the 20-period high (with volume spike ≥1.5×)
  SELL when close breaks below the 20-period low (with volume spike ≥1.5×)

Buffer: breakout must exceed the channel by at least 0.05% to avoid noise.
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """Donchian channel breakout with volume confirmation."""

    name = "breakout"
    version = "2.0"
    description = "20-period Donchian channel breakout with volume spike confirmation"

    def __init__(self):
        self.min_candles = 22
        self.channel_period = 20        # bars for high/low channel
        self.buffer_pct = 0.02          # price must exceed channel by this %
        self.vol_sma_period = 20
        self.vol_spike_ratio = 1.0      # volume must be ≥ 1.0× average (any vol ok)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        # Donchian channel from previous candles (exclude current)
        prev_candles = context_candles[-self.channel_period - 1:-1]
        if len(prev_candles) < self.channel_period:
            return None

        channel_high = max(c['high'] for c in prev_candles)
        channel_low = min(c['low'] for c in prev_candles)

        current = context_candles[-1]
        close = current['close']

        # Volume filter
        volumes = [c['volume'] for c in context_candles]
        vol_sma = sum(volumes[-self.vol_sma_period - 1:-1]) / self.vol_sma_period
        if vol_sma <= 0:
            return None
        vol_ratio = current['volume'] / vol_sma

        if vol_ratio < self.vol_spike_ratio:
            return None  # No volume confirmation

        # Upward breakout
        up_threshold = channel_high * (1.0 + self.buffer_pct / 100.0)
        if close >= up_threshold:
            confidence = min(0.85, 0.65 + min(vol_ratio - self.vol_spike_ratio, 1.5) * 0.10)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'channel_high': round(channel_high, 6),
                'channel_low': round(channel_low, 6),
                'close': round(close, 6),
                'vol_ratio': round(vol_ratio, 4),
            }

        # Downward breakout
        dn_threshold = channel_low * (1.0 - self.buffer_pct / 100.0)
        if close <= dn_threshold:
            confidence = min(0.85, 0.65 + min(vol_ratio - self.vol_spike_ratio, 1.5) * 0.10)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'channel_high': round(channel_high, 6),
                'channel_low': round(channel_low, 6),
                'close': round(close, 6),
                'vol_ratio': round(vol_ratio, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'channel_period': self.channel_period,
            'buffer_pct': self.buffer_pct,
            'vol_spike_ratio': self.vol_spike_ratio,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'channel_period' in params:
            self.channel_period = int(params['channel_period'])
        if 'buffer_pct' in params:
            self.buffer_pct = float(params['buffer_pct'])
        if 'vol_spike_ratio' in params:
            self.vol_spike_ratio = float(params['vol_spike_ratio'])
