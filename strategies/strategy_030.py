#!/usr/bin/env python3
"""
Strategy Name: Volatility Compression Fade
Type: volatility/mean-reversion hybrid
Timeframe: 1m
Description: Detects periods of compressed volatility (ATR contraction) then fades
    sharp moves away from VWAP, betting on reversion during low-vol ranging regimes.
    Combines ATR ratio for volatility state, VWAP deviation for entry, and RSI
    divergence for confirmation. Designed for ranging_low_volatility conditions.
"""
import math
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class VolatilityCompressionFadeStrategy(BaseStrategy):
    """Fade sharp moves during compressed volatility regimes using ATR + VWAP + RSI."""

    name = "vol_compression_fade"
    version = "1.0"
    description = "Volatility compression fade: low ATR ratio + VWAP deviation + RSI confirmation"

    def __init__(self):
        self.min_candles = 40
        # ATR parameters
        self.atr_fast = 7
        self.atr_slow = 28
        self.compression_ratio = 0.75   # fast_atr / slow_atr < this = compressed
        # VWAP deviation trigger
        self.vwap_lookback = 20
        self.min_deviation_pct = 0.20   # must deviate at least this much from VWAP
        self.max_deviation_pct = 0.80   # too far = momentum, don't fade
        # RSI confirmation
        self.rsi_period = 14
        self.rsi_oversold = 40
        self.rsi_overbought = 60
        # Keltner channel for additional vol context
        self.kc_period = 20
        self.kc_mult = 1.5

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _calc_atr(self, candles: List[dict], period: int) -> float:
        """Average True Range over given period."""
        if len(candles) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            prev_close = candles[i - 1]['close']
            tr = max(
                c['high'] - c['low'],
                abs(c['high'] - prev_close),
                abs(c['low'] - prev_close),
            )
            trs.append(tr)
        if len(trs) < period:
            return 0.0
        # Wilder smoothing
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    def _calc_vwap(self, candles: List[dict]) -> float:
        """VWAP from typical price weighted by volume."""
        total_pv = 0.0
        total_v = 0.0
        for c in candles:
            tp = (c['high'] + c['low'] + c['close']) / 3.0
            total_pv += tp * c['volume']
            total_v += c['volume']
        return total_pv / total_v if total_v > 0 else 0.0

    def _calc_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        """Wilder RSI."""
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        ag = sum(gains[:period]) / period
        al = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    def _bb_bandwidth(self, closes: List[float], period: int = 20) -> float:
        """Bollinger bandwidth as percentage of SMA."""
        if len(closes) < period:
            return 999.0
        window = closes[-period:]
        sma = sum(window) / len(window)
        if sma <= 0:
            return 999.0
        variance = sum((p - sma) ** 2 for p in window) / len(window)
        std = math.sqrt(variance)
        return (4.0 * std) / sma * 100.0  # 2-std band width as pct

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        price = closes[-1]

        # 1. Volatility compression check (ATR ratio)
        atr_fast = self._calc_atr(context_candles, self.atr_fast)
        atr_slow = self._calc_atr(context_candles, self.atr_slow)
        if atr_slow <= 0 or atr_fast <= 0:
            return None

        atr_ratio = atr_fast / atr_slow
        if atr_ratio > self.compression_ratio:
            return None  # Volatility not compressed enough

        # 2. BB bandwidth as secondary vol filter — must be tight
        bb_bw = self._bb_bandwidth(closes)
        if bb_bw > 3.0:
            return None  # Bands too wide for a compression play

        # 3. VWAP deviation for entry trigger
        vwap_candles = context_candles[-self.vwap_lookback:]
        vwap = self._calc_vwap(vwap_candles)
        if vwap <= 0:
            return None

        deviation_pct = (price - vwap) / vwap * 100.0
        abs_dev = abs(deviation_pct)

        if abs_dev < self.min_deviation_pct or abs_dev > self.max_deviation_pct:
            return None  # Not in the sweet spot

        # 4. RSI confirmation (must agree with fade direction)
        rsi = self._calc_rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # Confidence components
        # Higher ATR compression = higher confidence
        compression_score = max(0.0, 1.0 - atr_ratio) * 0.4
        # Deviation magnitude within range
        dev_score = (abs_dev - self.min_deviation_pct) / (self.max_deviation_pct - self.min_deviation_pct) * 0.3
        # RSI extremity
        rsi_score = 0.0

        # FADE: price above VWAP → SELL, price below VWAP → BUY
        if deviation_pct > 0 and rsi > self.rsi_overbought:
            # Price extended above VWAP with overbought RSI → fade short
            rsi_score = min((rsi - self.rsi_overbought) / 30.0, 1.0) * 0.3
            confidence = min(0.85, 0.50 + compression_score + dev_score + rsi_score)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'atr_ratio': round(atr_ratio, 4),
                'bb_bw_pct': round(bb_bw, 4),
                'vwap': round(vwap, 6),
                'deviation_pct': round(deviation_pct, 4),
                'rsi': round(rsi, 2),
            }

        if deviation_pct < 0 and rsi < self.rsi_oversold:
            # Price extended below VWAP with oversold RSI → fade long
            rsi_score = min((self.rsi_oversold - rsi) / 30.0, 1.0) * 0.3
            confidence = min(0.85, 0.50 + compression_score + dev_score + rsi_score)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'atr_ratio': round(atr_ratio, 4),
                'bb_bw_pct': round(bb_bw, 4),
                'vwap': round(vwap, 6),
                'deviation_pct': round(deviation_pct, 4),
                'rsi': round(rsi, 2),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'atr_fast': self.atr_fast,
            'atr_slow': self.atr_slow,
            'compression_ratio': self.compression_ratio,
            'vwap_lookback': self.vwap_lookback,
            'min_deviation_pct': self.min_deviation_pct,
            'max_deviation_pct': self.max_deviation_pct,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ('atr_fast', 'atr_slow', 'rsi_period', 'vwap_lookback'):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ('compression_ratio', 'min_deviation_pct', 'max_deviation_pct',
                     'rsi_oversold', 'rsi_overbought'):
            if key in params:
                setattr(self, key, float(params[key]))