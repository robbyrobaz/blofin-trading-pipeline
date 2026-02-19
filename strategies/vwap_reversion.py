#!/usr/bin/env python3
"""
VWAP Mean Reversion Strategy — candle-based interface.

Calculates VWAP from typical price (H+L+C)/3 × volume.
Signals when close deviates ≥0.3% from VWAP and RSI confirms direction.
"""
import math
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class VWAPReversionStrategy(BaseStrategy):
    """Detects mean reversion from VWAP deviation using OHLCV candle data."""

    name = "vwap_reversion"
    version = "2.0"
    description = "VWAP mean reversion: trade candle-close deviation from VWAP with RSI confirmation"

    def __init__(self):
        self.min_candles = 20
        self.deviation_pct = 0.30       # 0.30% deviation threshold (calibrated for 1m candles)
        self.rsi_period = 14
        self.rsi_oversold = 45          # Relaxed for 1m data (not extremes like 30)
        self.rsi_overbought = 55

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _calc_vwap(self, candles: List[dict]) -> float:
        """VWAP from typical price (H+L+C)/3 weighted by volume."""
        total_pv = 0.0
        total_v = 0.0
        for c in candles:
            tp = (c['high'] + c['low'] + c['close']) / 3.0
            total_pv += tp * c['volume']
            total_v += c['volume']
        return total_pv / total_v if total_v > 0 else 0.0

    def _calc_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        """Wilder RSI on close prices."""
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        # Initial averages
        ag = sum(gains[:period]) / period
        al = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        price = closes[-1]

        vwap = self._calc_vwap(context_candles)
        if vwap <= 0:
            return None

        deviation_pct = (price - vwap) / vwap * 100.0

        rsi = self._calc_rsi(closes, self.rsi_period)

        # BUY: price below VWAP + RSI not overbought
        if deviation_pct <= -self.deviation_pct:
            if rsi is not None and rsi > self.rsi_overbought:
                return None
            confidence = min(0.85, 0.60 + abs(deviation_pct) / (self.deviation_pct * 4) * 0.25)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'vwap': round(vwap, 6),
                'deviation_pct': round(deviation_pct, 4),
                'rsi': round(rsi, 2) if rsi is not None else None,
            }

        # SELL: price above VWAP + RSI not oversold
        if deviation_pct >= self.deviation_pct:
            if rsi is not None and rsi < self.rsi_oversold:
                return None
            confidence = min(0.85, 0.60 + abs(deviation_pct) / (self.deviation_pct * 4) * 0.25)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'vwap': round(vwap, 6),
                'deviation_pct': round(deviation_pct, 4),
                'rsi': round(rsi, 2) if rsi is not None else None,
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'deviation_pct': self.deviation_pct,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'min_candles': self.min_candles,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'deviation_pct' in params:
            self.deviation_pct = float(params['deviation_pct'])
        if 'rsi_oversold' in params:
            self.rsi_oversold = float(params['rsi_oversold'])
        if 'rsi_overbought' in params:
            self.rsi_overbought = float(params['rsi_overbought'])
