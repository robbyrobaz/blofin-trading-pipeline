#!/usr/bin/env python3
"""
ATR-Based Reversal Strategy — candle-based interface.

Detects price exhaustion and reversal using:
  1. ATR-normalized distance from recent extreme (high or low)
  2. Consecutive-candle momentum shift (last 3 candles changing direction)
  3. RSI confirming the extreme reached

BUY: Price bounced from recent low by ≥ 0.5 ATR AND last 2 candles green
SELL: Price fell from recent high by ≥ 0.5 ATR AND last 2 candles red
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class ReversalStrategy(BaseStrategy):
    """ATR-normalized reversal from local price extremes."""

    name = "reversal"
    version = "2.0"
    description = "ATR-normalized reversal: bounce from local extreme + momentum shift confirmation"

    def __init__(self):
        self.min_candles = 20
        self.atr_period = 14
        self.extreme_lookback = 20      # bars to find local high/low
        self.atr_threshold = 0.3        # must bounce ≥ 0.3 ATR from extreme
        self.rsi_period = 14
        self.rsi_extreme_buy = 50       # RSI must be below 50 for BUY reversal
        self.rsi_extreme_sell = 50      # RSI must be above 50 for SELL reversal
        self.confirm_candles = 1        # consecutive candles in new direction

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calc_atr(self, candles: List[dict], period: int) -> float:
        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i - 1]['close']
            tr = max(c['high'] - c['low'], abs(c['high'] - pc), abs(c['low'] - pc))
            trs.append(tr)
        if not trs:
            return 0.0
        return sum(trs[-period:]) / min(len(trs), period)

    def _calc_rsi(self, closes: List[float], period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
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
        current = context_candles[-1]
        price = closes[-1]

        atr = self._calc_atr(context_candles, self.atr_period)
        if atr <= 0:
            return None

        rsi = self._calc_rsi(closes, self.rsi_period)

        # Local extreme window (excluding current candle)
        lookback = context_candles[-self.extreme_lookback - 1:-1]
        if not lookback:
            return None

        local_low = min(c['low'] for c in lookback)
        local_high = max(c['high'] for c in lookback)

        # Check last `confirm_candles` are consistently directional
        recent = context_candles[-(self.confirm_candles + 1):]
        all_green = all(c['close'] > c['open'] for c in recent[-self.confirm_candles:])
        all_red = all(c['close'] < c['open'] for c in recent[-self.confirm_candles:])

        bounce_from_low = (price - local_low) / atr
        drop_from_high = (local_high - price) / atr
        range_pct = (local_high - local_low) / local_low * 100 if local_low > 0 else 0

        # Need meaningful price range (not flat market)
        if range_pct < 0.05:
            return None

        # BUY: price is near local low (small bounce) AND RSI below midpoint
        # Catches the early stages of reversal from oversold conditions
        if bounce_from_low >= self.atr_threshold and rsi < self.rsi_extreme_buy:
            confidence = min(0.80, 0.60 + min(bounce_from_low / 3.0, 0.15) + (self.rsi_extreme_buy - rsi) / 200)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'local_low': round(local_low, 6),
                'bounce_atr': round(bounce_from_low, 3),
                'atr': round(atr, 6),
                'rsi': round(rsi, 2),
            }

        # SELL: price is near local high (small drop) AND RSI above midpoint
        if drop_from_high >= self.atr_threshold and rsi > self.rsi_extreme_sell:
            confidence = min(0.80, 0.60 + min(drop_from_high / 3.0, 0.15) + (rsi - self.rsi_extreme_sell) / 200)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'local_high': round(local_high, 6),
                'drop_atr': round(drop_from_high, 3),
                'atr': round(atr, 6),
                'rsi': round(rsi, 2),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'atr_period': self.atr_period,
            'extreme_lookback': self.extreme_lookback,
            'atr_threshold': self.atr_threshold,
            'rsi_extreme_buy': self.rsi_extreme_buy,
            'rsi_extreme_sell': self.rsi_extreme_sell,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for k in ('atr_period', 'extreme_lookback', 'confirm_candles'):
            if k in params:
                setattr(self, k, int(params[k]))
        for k in ('atr_threshold', 'rsi_extreme_buy', 'rsi_extreme_sell'):
            if k in params:
                setattr(self, k, float(params[k]))
