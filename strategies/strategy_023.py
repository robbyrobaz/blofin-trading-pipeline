#!/usr/bin/env python3
"""
Strategy 023: Volatility Expansion Volume Breakout — candle-based interface.
Type: volatility/breakout, Timeframe: 5m

Detects ultra-low volatility regimes and confirms breakouts with volume surge.
Works in ranging markets transitioning to directional moves.
"""
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy


class VolatilityExpansionVolumeBreakoutStrategy(BaseStrategy):
    """Ultra-low volatility breakout confirmed by volume surge (candle-based)."""

    name = "volatility_expansion_volume_breakout"
    version = "2.0"
    description = "Detects low-vol squeeze breakouts confirmed by volume surge"

    def __init__(self):
        self.volatility_window     = 20
        self.volatility_percentile = 15
        # Lowered from 1.8 → 1.3: tick-based volume max ratio ~2.2x
        self.volume_multiplier     = 1.3
        # Lowered from 0.015 (1.5%) → 0.003 (0.3%): avg 5m move is 0.11%, max was 1.15%
        self.breakout_threshold    = 0.003
        self.min_volume_bars       = 3
        self.min_candles           = self.volatility_window + 5

    def _calc_current_volatility(self, candles: list) -> float:
        """Normalized ATR proxy: (high - low) / close for current candle."""
        c = candles[-1]
        if c['close'] <= 0:
            return 0.0
        return (c['high'] - c['low']) / c['close']

    def _calc_vol_threshold(self, candles: list) -> float:
        """Percentile threshold of high-low range over volatility_window."""
        history = []
        for i in range(-self.volatility_window, 0):
            if abs(i) <= len(candles):
                c = candles[i]
                if c['close'] > 0:
                    history.append((c['high'] - c['low']) / c['close'])
        if not history:
            return float('inf')
        history.sort()
        idx = int(len(history) * self.volatility_percentile / 100)
        return history[min(idx, len(history) - 1)]

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        current = context_candles[-1]
        prev    = context_candles[-2]

        current_vol = self._calc_current_volatility(context_candles)
        vol_threshold = self._calc_vol_threshold(context_candles)

        is_ultra_low_vol = current_vol < vol_threshold

        # Volume surge: current volume vs average of preceding bars
        lookback = min(self.min_volume_bars, len(context_candles) - 1)
        past_vols = [c['volume'] for c in context_candles[-lookback - 1:-1]]
        avg_volume = sum(past_vols) / len(past_vols) if past_vols else 0
        volume_surge = (avg_volume > 0 and
                        current['volume'] > avg_volume * self.volume_multiplier)

        # Price breakout
        if prev['close'] <= 0:
            return None
        price_change = abs(current['close'] - prev['close']) / prev['close']
        is_breakout = price_change > self.breakout_threshold

        if is_ultra_low_vol and volume_surge and is_breakout:
            # Confidence based on how extreme each factor is
            vol_score   = max(0.0, 1 - (current_vol / vol_threshold)) if vol_threshold > 0 else 0
            vol_ratio   = current['volume'] / avg_volume if avg_volume > 0 else 1
            vol_s_score = min(1.0, (vol_ratio - 1) / max(self.volume_multiplier - 1, 0.01))
            brk_score   = min(1.0, price_change / self.breakout_threshold)
            confidence  = max(0.0, min(1.0, vol_score * 0.3 + vol_s_score * 0.4 + brk_score * 0.3))
            # Ensure minimum confidence for a real signal
            confidence  = max(0.50, confidence)

            signal = 'BUY' if current['close'] > prev['close'] else 'SELL'
            return {
                'signal': signal,
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'price_change_pct': round(price_change * 100, 4),
                'volume_ratio': round(vol_ratio, 2),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'volatility_window': self.volatility_window,
            'volatility_percentile': self.volatility_percentile,
            'volume_multiplier': self.volume_multiplier,
            'breakout_threshold': self.breakout_threshold,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'volatility_window' in params:
            self.volatility_window = int(params['volatility_window'])
        if 'volatility_percentile' in params:
            self.volatility_percentile = int(params['volatility_percentile'])
        if 'volume_multiplier' in params:
            self.volume_multiplier = float(params['volume_multiplier'])
        if 'breakout_threshold' in params:
            self.breakout_threshold = float(params['breakout_threshold'])
