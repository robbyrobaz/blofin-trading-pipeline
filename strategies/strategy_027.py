#!/usr/bin/env python3
"""
Strategy 027: Volatility Adaptive Mean Reversion — candle-based interface.
Type: volatility/mean-reversion, Timeframe: 5m

Mean-reversion at Bollinger Band extremes with RSI confirmation and ATR
volatility gating. Position sizing adapts to volatility (modeled via confidence).
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy


class VolatilityAdaptiveMeanReversionStrategy(BaseStrategy):
    """Volatility-adaptive mean-reversion at BB extremes with RSI confirmation."""

    name = "volatility_adaptive_mean_reversion"
    version = "2.0"
    description = "BB mean-reversion with RSI confirmation and ATR volatility gate"

    def __init__(self):
        self.bb_period         = 20
        self.bb_std_dev        = 2.0
        self.rsi_period        = 14
        self.rsi_oversold      = 35
        self.rsi_overbought    = 65
        self.atr_period        = 14
        self.atr_threshold     = 0.5   # fraction of BB range
        self.min_atr_pct       = 0.15  # minimum ATR as % of price (noise filter)
        self.max_atr_pct       = 2.0   # maximum ATR as % of price (chaos filter)
        self.min_candles       = self.bb_period + 2

    def _calc_bb(self, closes: List[float]):
        period = self.bb_period
        if len(closes) < period:
            return None, None, None
        recent = closes[-period:]
        mid = sum(recent) / period
        std = (sum((x - mid) ** 2 for x in recent) / period) ** 0.5
        return mid + self.bb_std_dev * std, mid, mid - self.bb_std_dev * std

    def _calc_rsi(self, closes: List[float]) -> float:
        period = self.rsi_period
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

    def _calc_atr(self, candles: list) -> float:
        """Return current ATR value."""
        period = self.atr_period
        n = len(candles)
        if n < 2:
            return 0.0
        trs = [candles[0]['high'] - candles[0]['low']]
        for i in range(1, n):
            h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        if n < period:
            return sum(trs) / n
        atr = sum(trs[:period]) / period
        for i in range(period, n):
            atr = (atr * (period - 1) + trs[i]) / period
        return atr

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        current_close = closes[-1]

        bb_upper, bb_middle, bb_lower = self._calc_bb(closes)
        if bb_upper is None or bb_lower is None:
            return None

        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return None

        atr = self._calc_atr(context_candles)
        if current_close <= 0:
            return None
        atr_pct = (atr / current_close) * 100.0

        # Noise filter: ATR too small or too large
        if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
            return None

        rsi = self._calc_rsi(closes)

        # Adaptive threshold based on volatility
        volatility_score = atr_pct / self.max_atr_pct
        volatility_mult  = max(0.5, min(volatility_score, 1.0))
        eff_atr_threshold = self.atr_threshold * volatility_mult

        price_position = (current_close - bb_lower) / bb_range

        buy_signal = (
            current_close <= bb_lower + (bb_range * eff_atr_threshold) and
            rsi < self.rsi_oversold
        )
        sell_signal = (
            current_close >= bb_upper - (bb_range * eff_atr_threshold) and
            rsi > self.rsi_overbought
        )

        if buy_signal:
            rsi_conf  = max(0, abs(rsi - 50) - 15) / 35.0
            band_conf = min(abs(price_position) * 2, 1.0)
            vol_conf  = min(atr_pct / self.max_atr_pct, 1.0)
            confidence = max(0.50, min(band_conf * 0.4 + rsi_conf * 0.3 + vol_conf * 0.3, 1.0))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(rsi, 2),
                'atr_pct': round(atr_pct, 4),
                'bb_position': round(price_position, 4),
            }

        if sell_signal:
            rsi_conf  = max(0, abs(rsi - 50) - 15) / 35.0
            band_conf = min(abs(price_position) * 2, 1.0)
            vol_conf  = min(atr_pct / self.max_atr_pct, 1.0)
            confidence = max(0.50, min(band_conf * 0.4 + rsi_conf * 0.3 + vol_conf * 0.3, 1.0))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(rsi, 2),
                'atr_pct': round(atr_pct, 4),
                'bb_position': round(price_position, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'bb_period': self.bb_period,
            'bb_std_dev': self.bb_std_dev,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'atr_period': self.atr_period,
            'min_atr_pct': self.min_atr_pct,
            'max_atr_pct': self.max_atr_pct,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ('bb_period', 'rsi_period', 'atr_period'):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ('bb_std_dev', 'rsi_oversold', 'rsi_overbought',
                    'atr_threshold', 'min_atr_pct', 'max_atr_pct'):
            if key in params:
                setattr(self, key, float(params[key]))
