#!/usr/bin/env python3
"""
Bollinger Band Squeeze Breakout — candle-based interface.

Logic:
  1. Detect squeeze: BB bandwidth < threshold (bands are tight)
  2. Wait for breakout: close moves outside the bands
  3. Volume confirmation: current volume > 1.2x average (confirms move)

Uses 20-period SMA ± 2 std dev on close prices.
Squeeze threshold: bandwidth (upper-lower)/SMA < 1.0% for 1m candles.
"""
import math
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class BBSqueezeStrategy(BaseStrategy):
    """Bollinger Band squeeze breakout with volume confirmation."""

    name = "bb_squeeze"
    version = "2.0"
    description = "BB squeeze breakout: tight bands → volume spike → directional breakout"

    def __init__(self):
        self.min_candles = 22
        self.bb_period = 20
        self.bb_std_mult = 2.0
        self.squeeze_bw_pct = 2.0       # band-width/SMA < 2% = squeeze (1m calibrated)
        self.min_bw_pct = 0.05          # ignore near-zero-volatility noise
        self.vol_sma_period = 20
        self.vol_confirm_ratio = 1.0    # current vol must be ≥ 1.0× average to confirm

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calc_bb(self, closes: List[float]) -> Tuple[float, float, float]:
        """Returns (sma, upper, lower) for the last bb_period closes."""
        window = closes[-self.bb_period:]
        sma = sum(window) / len(window)
        variance = sum((p - sma) ** 2 for p in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 0.0
        upper = sma + self.bb_std_mult * std
        lower = sma - self.bb_std_mult * std
        return sma, upper, lower

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        volumes = [c['volume'] for c in context_candles]
        price = closes[-1]

        sma, upper, lower = self._calc_bb(closes)
        if sma <= 0:
            return None

        bw_pct = (upper - lower) / sma * 100.0

        # Must be in squeeze
        if bw_pct > self.squeeze_bw_pct:
            return None
        # Must not be near-zero noise
        if bw_pct < self.min_bw_pct:
            return None

        # Volume confirmation
        vol_sma = sum(volumes[-self.vol_sma_period:]) / min(len(volumes), self.vol_sma_period)
        current_vol = volumes[-1]
        vol_ratio = current_vol / vol_sma if vol_sma > 0 else 1.0

        if vol_ratio < self.vol_confirm_ratio:
            return None  # No volume confirmation

        # Breakout direction
        if price > upper:
            confidence = min(0.85, 0.65 + vol_ratio / 10.0)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'sma': round(sma, 6),
                'upper': round(upper, 6),
                'bw_pct': round(bw_pct, 4),
                'vol_ratio': round(vol_ratio, 4),
            }

        if price < lower:
            confidence = min(0.85, 0.65 + vol_ratio / 10.0)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'sma': round(sma, 6),
                'lower': round(lower, 6),
                'bw_pct': round(bw_pct, 4),
                'vol_ratio': round(vol_ratio, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'bb_period': self.bb_period,
            'bb_std_mult': self.bb_std_mult,
            'squeeze_bw_pct': self.squeeze_bw_pct,
            'vol_confirm_ratio': self.vol_confirm_ratio,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'bb_period' in params:
            self.bb_period = int(params['bb_period'])
        if 'bb_std_mult' in params:
            self.bb_std_mult = float(params['bb_std_mult'])
        if 'squeeze_bw_pct' in params:
            self.squeeze_bw_pct = float(params['squeeze_bw_pct'])
        if 'vol_confirm_ratio' in params:
            self.vol_confirm_ratio = float(params['vol_confirm_ratio'])
