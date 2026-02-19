#!/usr/bin/env python3
"""
Volume-Weighted Mean Reversion — candle-based interface.

Targets low-volatility ranging markets. Only trades VWAP deviations when:
  1. ATR-based volatility is below threshold (ranging regime)
  2. Volume is stable — not spiking (no breakout risk)
  3. RSI confirms oversold/overbought
"""
import math
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class VolumeMeanReversionStrategy(BaseStrategy):
    """Mean reversion in low-volatility ranging markets. VWAP + ATR + RSI filter."""

    name = "volume_mean_reversion"
    version = "2.0"
    description = "Volume-weighted mean reversion for ranging markets: VWAP + ATR vol filter + RSI"

    def __init__(self):
        self.min_candles = 20
        self.vwap_lookback = 30         # candles for VWAP calculation
        self.deviation_pct = 0.25       # minimum VWAP deviation to signal
        self.atr_period = 14
        self.max_atr_pct = 0.4          # max ATR/price % — reject high-volatility candles
        self.vol_sma_period = 20
        self.max_vol_ratio = 2.0        # reject volume spikes (breakout risk)
        self.min_vol_ratio = 0.3        # reject thin markets
        self.rsi_period = 14
        self.rsi_oversold = 40
        self.rsi_overbought = 60

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calc_atr(self, candles: List[dict], period: int = 14) -> float:
        """Average True Range."""
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
        if not trs:
            return 0.0
        # Simple average of last `period` TRs
        return sum(trs[-period:]) / min(len(trs), period)

    def _calc_vwap(self, candles: List[dict]) -> float:
        total_pv, total_v = 0.0, 0.0
        for c in candles:
            tp = (c['high'] + c['low'] + c['close']) / 3.0
            total_pv += tp * c['volume']
            total_v += c['volume']
        return total_pv / total_v if total_v > 0 else 0.0

    def _calc_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
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

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        price = closes[-1]

        # --- VWAP ---
        lookback_candles = context_candles[-self.vwap_lookback:]
        vwap = self._calc_vwap(lookback_candles)
        if vwap <= 0:
            return None
        deviation_pct = (price - vwap) / vwap * 100.0

        # --- ATR volatility filter ---
        atr = self._calc_atr(context_candles, self.atr_period)
        atr_pct = atr / price * 100.0 if price > 0 else 99.0
        if atr_pct > self.max_atr_pct:
            return None  # Too volatile — not a ranging market

        # --- Volume filter ---
        volumes = [c['volume'] for c in context_candles]
        vol_sma = sum(volumes[-self.vol_sma_period:]) / min(len(volumes), self.vol_sma_period)
        current_vol = volumes[-1]
        if vol_sma <= 0:
            return None
        vol_ratio = current_vol / vol_sma
        if vol_ratio > self.max_vol_ratio or vol_ratio < self.min_vol_ratio:
            return None  # Volume spike or thin market

        # --- RSI ---
        rsi = self._calc_rsi(closes, self.rsi_period)

        # --- Signal ---
        if abs(deviation_pct) < self.deviation_pct:
            return None

        if deviation_pct <= -self.deviation_pct:
            if rsi is not None and rsi > self.rsi_overbought:
                return None
            confidence = min(0.85, 0.60 + abs(deviation_pct) / (self.deviation_pct * 3) * 0.20)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'vwap': round(vwap, 6),
                'deviation_pct': round(deviation_pct, 4),
                'atr_pct': round(atr_pct, 4),
                'vol_ratio': round(vol_ratio, 4),
                'rsi': round(rsi, 2) if rsi is not None else None,
                'regime': 'ranging_low_volatility',
            }

        if deviation_pct >= self.deviation_pct:
            if rsi is not None and rsi < self.rsi_oversold:
                return None
            confidence = min(0.85, 0.60 + abs(deviation_pct) / (self.deviation_pct * 3) * 0.20)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'vwap': round(vwap, 6),
                'deviation_pct': round(deviation_pct, 4),
                'atr_pct': round(atr_pct, 4),
                'vol_ratio': round(vol_ratio, 4),
                'rsi': round(rsi, 2) if rsi is not None else None,
                'regime': 'ranging_low_volatility',
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'deviation_pct': self.deviation_pct,
            'max_atr_pct': self.max_atr_pct,
            'max_vol_ratio': self.max_vol_ratio,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for k in ('deviation_pct', 'max_atr_pct', 'max_vol_ratio',
                  'rsi_oversold', 'rsi_overbought'):
            if k in params:
                setattr(self, k, float(params[k]))
