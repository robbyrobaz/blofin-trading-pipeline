#!/usr/bin/env python3
"""
BB Squeeze v2 — candle-based interface.
Detects Bollinger Band squeeze breakouts using OHLCV candles.
"""
import os
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy, Signal


class BBSqueezeV2Strategy(BaseStrategy):
    """Detects Bollinger Band squeeze breakouts — candle interface."""

    name = "bb_squeeze_v2"
    version = "2.0"
    description = "Detects breakouts from Bollinger Band squeeze conditions (candle-based)"

    def __init__(self):
        self.std_mult = float(os.getenv("BB_STD_MULT", "2.0"))
        self.squeeze_threshold = float(os.getenv("BB_SQUEEZE_THRESHOLD", "0.3"))
        # Min band width % to filter near-zero-vol noise
        self.min_band_width_pct = float(os.getenv("BB_MIN_BAND_WIDTH_PCT", "0.5"))
        self.min_candles = 20

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        current_price = closes[-1]

        mean = sum(closes) / len(closes)
        variance = sum((p - mean) ** 2 for p in closes) / len(closes)
        std = variance ** 0.5

        if mean <= 0 or std <= 0:
            return None

        upper_band = mean + (self.std_mult * std)
        lower_band = mean - (self.std_mult * std)
        band_width_pct = ((upper_band - lower_band) / mean) * 100.0

        # Check if bands are tight (squeeze)
        is_squeeze = band_width_pct <= (self.squeeze_threshold * 100)
        if not is_squeeze:
            return None

        # Require minimum band width to avoid noise
        if band_width_pct < self.min_band_width_pct:
            return None

        # Check for breakout direction
        if current_price > upper_band:
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': 0.75,
                'mean': round(mean, 6),
                'upper': round(upper_band, 6),
                'band_width_pct': round(band_width_pct, 4),
            }
        elif current_price < lower_band:
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': 0.75,
                'mean': round(mean, 6),
                'lower': round(lower_band, 6),
                'band_width_pct': round(band_width_pct, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "std_mult": self.std_mult,
            "squeeze_threshold": self.squeeze_threshold,
            "min_band_width_pct": self.min_band_width_pct,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "std_mult" in params:
            self.std_mult = float(params["std_mult"])
        if "squeeze_threshold" in params:
            self.squeeze_threshold = float(params["squeeze_threshold"])
        if "min_band_width_pct" in params:
            self.min_band_width_pct = float(params["min_band_width_pct"])
