#!/usr/bin/env python3
"""
Momentum V1 Strategy — candle-based interface.

Measures close-to-close price change over a lookback window of candles.
Signals BUY when cumulative return >= up_pct, SELL when <= down_pct.

Evening tuning 2026-02-16: raised threshold from 0.60% → 1.0% to filter weak signals.
"""
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy


class MomentumV1Strategy(BaseStrategy):
    """Detects strong price momentum over a window of candles."""

    name = "momentum_v1"
    version = "2.0"
    description = "Detects strong upward or downward price momentum (candle-based)"

    def __init__(self):
        self.lookback_candles = 8    # ~240s at 30s candles; ~40min at 5m candles
        self.up_pct = 1.0
        self.down_pct = -1.0

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.lookback_candles + 1:
            return None

        window = context_candles[-self.lookback_candles - 1:]
        first_close = window[0]['close']
        last_close = window[-1]['close']

        if first_close <= 0:
            return None

        pct = ((last_close - first_close) / first_close) * 100.0

        if pct >= self.up_pct:
            confidence = min(0.99, max(0.50, pct / max(self.up_pct, 0.01)))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'lookback_candles': self.lookback_candles,
                'change_pct': round(pct, 4),
            }
        elif pct <= self.down_pct:
            confidence = min(0.99, max(0.50, abs(pct) / max(abs(self.down_pct), 0.01)))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'lookback_candles': self.lookback_candles,
                'change_pct': round(pct, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'lookback_candles': self.lookback_candles,
            'up_pct': self.up_pct,
            'down_pct': self.down_pct,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'lookback_candles' in params:
            self.lookback_candles = int(params['lookback_candles'])
        if 'up_pct' in params:
            self.up_pct = float(params['up_pct'])
        if 'down_pct' in params:
            self.down_pct = float(params['down_pct'])
