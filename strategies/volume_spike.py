#!/usr/bin/env python3
"""
Volume Spike Strategy — candle-based interface.

Detects volume surges (current candle volume >> average) combined with
a meaningful price move in the same direction.
"""
import os
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy


class VolumeSpikeStrategy(BaseStrategy):
    """Detects volume surges with price direction confirmation (candle-based)."""

    name = "volume_spike"
    version = "2.0"
    description = "Detects volume surges (2-3× average) with price move confirmation"

    def __init__(self):
        self.lookback_candles    = int(os.getenv("VOLUME_SPIKE_LOOKBACK_CANDLES", "20"))
        # Note: 'volume' in DB is tick-count (not real volume), nearly constant.
        # Setting multiplier to 0.8 effectively disables volume gate so the strategy
        # acts as a pure price-momentum detector. Real volume support would need schema change.
        self.spike_multiplier    = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "0.8"))
        # Lowered from 0.3 → 0.15: price move is the real signal here
        self.min_price_move_pct  = float(os.getenv("VOLUME_SPIKE_MIN_PRICE_MOVE_PCT", "0.15"))
        self.min_candles         = 10

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        current = context_candles[-1]
        current_volume = current['volume']
        if current_volume <= 0:
            return None

        # Average volume from lookback window (excluding current candle)
        lookback = min(self.lookback_candles, len(context_candles) - 1)
        past_candles = context_candles[-lookback - 1:-1]
        if not past_candles:
            return None

        avg_volume = sum(c['volume'] for c in past_candles) / len(past_candles)
        if avg_volume <= 0:
            return None

        volume_ratio = current_volume / avg_volume
        if volume_ratio < self.spike_multiplier:
            return None

        # Price move: compare current close to average close of past candles
        avg_recent_close = sum(c['close'] for c in past_candles[-5:]) / min(5, len(past_candles))
        if avg_recent_close <= 0:
            return None

        price_move_pct = ((current['close'] - avg_recent_close) / avg_recent_close) * 100.0

        if price_move_pct >= self.min_price_move_pct:
            confidence = min(0.90, 0.60 + (volume_ratio / 10.0))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'volume_ratio': round(volume_ratio, 2),
                'price_move_pct': round(price_move_pct, 4),
            }

        if price_move_pct <= -self.min_price_move_pct:
            confidence = min(0.90, 0.60 + (volume_ratio / 10.0))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'volume_ratio': round(volume_ratio, 2),
                'price_move_pct': round(price_move_pct, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'lookback_candles': self.lookback_candles,
            'spike_multiplier': self.spike_multiplier,
            'min_price_move_pct': self.min_price_move_pct,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'lookback_candles' in params:
            self.lookback_candles = int(params['lookback_candles'])
        if 'spike_multiplier' in params:
            self.spike_multiplier = float(params['spike_multiplier'])
        if 'min_price_move_pct' in params:
            self.min_price_move_pct = float(params['min_price_move_pct'])
