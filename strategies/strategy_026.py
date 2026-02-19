#!/usr/bin/env python3
"""
Strategy 026: Volatility Expansion Breakout — candle-based interface.
Type: volatility, Timeframe: 5m

Detects Bollinger Band squeezes in low-volatility regimes and trades
breakouts confirmed by ATR expansion and price movement outside bands.
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy


class VolatilityExpansionBreakoutStrategy(BaseStrategy):
    """BB squeeze breakout with ATR expansion confirmation (candle-based)."""

    name = "volatility_expansion_breakout"
    version = "2.0"
    description = "BB squeeze breakout confirmed by ATR expansion"

    def __init__(self):
        self.bb_period              = 20
        self.bb_std_dev             = 1.5
        self.atr_period             = 14
        self.atr_expansion_threshold = 1.3
        self.squeeze_lookback       = 5
        self.min_squeeze_bars       = 3
        self.volume_filter          = 0.8
        self.min_candles            = self.bb_period + self.squeeze_lookback + 2

    def _calc_bb(self, closes: List[float], period: int, std_dev: float):
        """Returns (upper, mid, lower) for the last candle."""
        if len(closes) < period:
            return None, None, None
        recent = closes[-period:]
        mid = sum(recent) / period
        variance = sum((x - mid) ** 2 for x in recent) / period
        std = variance ** 0.5
        return mid + std_dev * std, mid, mid - std_dev * std

    def _calc_atr(self, candles: list, period: int) -> List[float]:
        n = len(candles)
        if n < 2:
            return [0.0] * n
        trs = [candles[0]['high'] - candles[0]['low']]
        for i in range(1, n):
            h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atrs = [0.0] * n
        if n >= period:
            atrs[period - 1] = sum(trs[:period]) / period
            for i in range(period, n):
                atrs[i] = (atrs[i - 1] * (period - 1) + trs[i]) / period
        return atrs

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        current = context_candles[-1]
        prev    = context_candles[-2]

        closes = [c['close'] for c in context_candles]
        upper_band, mid_band, lower_band = self._calc_bb(closes, self.bb_period, self.bb_std_dev)
        if upper_band is None or mid_band is None or lower_band is None:
            return None

        bb_width = upper_band - lower_band
        if bb_width == 0:
            return None

        atrs = self._calc_atr(context_candles, self.atr_period)
        current_atr = atrs[-1]
        prev_atr    = atrs[-2] if len(atrs) >= 2 else current_atr
        atr_expansion = current_atr / prev_atr if prev_atr > 0 else 1.0

        # Check for squeeze: compare band width against recent deviation proxy
        historical_widths = []
        for i in range(-self.squeeze_lookback - 1, -1):
            if abs(i) <= len(context_candles):
                c_close = context_candles[i]['close']
                historical_widths.append(abs(c_close - mid_band))
        avg_hist_width = sum(historical_widths) / len(historical_widths) if historical_widths else 1.0
        squeeze_ratio = bb_width / (avg_hist_width * 2) if avg_hist_width > 0 else 1.0
        is_in_squeeze = squeeze_ratio < 0.6

        volume_ratio = (current['volume'] / prev['volume']
                        if prev['volume'] > 0 else 1.0)

        if is_in_squeeze and atr_expansion > self.atr_expansion_threshold:
            # BUY: price breaks above upper band with volume
            if current['close'] > upper_band and volume_ratio > self.volume_filter:
                price_dist = abs(current['close'] - upper_band)
                band_conf  = min(price_dist / (bb_width * 0.5), 1.0)
                atr_conf   = min((atr_expansion - 1.0) / 0.5, 1.0)
                vol_conf   = min(volume_ratio, 1.0)
                confidence = max(0.50, band_conf * 0.4 + atr_conf * 0.35 + vol_conf * 0.25)
                return {
                    'signal': 'BUY',
                    'strategy': self.name,
                    'confidence': round(min(confidence, 1.0), 4),
                    'atr_expansion': round(atr_expansion, 3),
                    'bb_width': round(bb_width, 6),
                    'squeeze_ratio': round(squeeze_ratio, 3),
                }

            # SELL: price breaks below lower band with volume
            if current['close'] < lower_band and volume_ratio > self.volume_filter:
                price_dist = abs(current['close'] - lower_band)
                band_conf  = min(price_dist / (bb_width * 0.5), 1.0)
                atr_conf   = min((atr_expansion - 1.0) / 0.5, 1.0)
                vol_conf   = min(volume_ratio, 1.0)
                confidence = max(0.50, band_conf * 0.4 + atr_conf * 0.35 + vol_conf * 0.25)
                return {
                    'signal': 'SELL',
                    'strategy': self.name,
                    'confidence': round(min(confidence, 1.0), 4),
                    'atr_expansion': round(atr_expansion, 3),
                    'bb_width': round(bb_width, 6),
                    'squeeze_ratio': round(squeeze_ratio, 3),
                }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'atr_period': self.atr_period,
            'atr_expansion_threshold': self.atr_expansion_threshold,
            'squeeze_lookback': self.squeeze_lookback,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ('bb_period', 'atr_period', 'squeeze_lookback', 'min_squeeze_bars'):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ('bb_std_dev', 'atr_expansion_threshold', 'volume_filter'):
            if key in params:
                setattr(self, key, float(params[key]))
