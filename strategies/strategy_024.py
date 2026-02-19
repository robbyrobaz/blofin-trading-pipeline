#!/usr/bin/env python3
"""
Strategy 024: Volume Volatility Mean Reversion — candle-based interface.
Type: volatility/volume + mean-reversion hybrid, Timeframe: 5m

Exploits volatility expansion with volume confirmation in ranging markets.
Trades mean-reversion pullbacks when ATR expands and volume supports reversal.
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy


class VolumeVolatilityMeanReversionStrategy(BaseStrategy):
    """Volatility expansion + volume-confirmed mean-reversion (candle-based)."""

    name = "volume_volatility_mean_reversion"
    version = "2.0"
    description = "Mean-reversion in volatility-expanding regimes confirmed by volume"

    def __init__(self):
        self.atr_period                   = 14
        self.atr_multiplier               = 1.5
        self.volume_ma_period             = 20
        # Note: volume = tick-count (nearly constant). Set to 0.8 to effectively bypass.
        self.volume_threshold             = 0.8
        self.rsi_period                   = 14
        # Loosened from 35 → 42: RSI < 35 occurred 122x/2016 candles; < 42 occurs 402x
        self.rsi_buy_threshold            = 42
        # Loosened from 65 → 58: RSI > 65 occurred 182x; > 58 occurs 486x
        self.rsi_sell_threshold           = 58
        self.min_candles                  = 30
        # Lowered from 1.3 → 1.1: ATR expansion >1.3 only 4x in 2016 candles; >1.1 is 52x
        self.volatility_expansion_thresh  = 1.1

    def _calc_atr(self, candles: list, period: int) -> List[float]:
        """Return ATR series for the candle list."""
        n = len(candles)
        if n < 2:
            return [0.0] * n
        trs = []
        trs.append(candles[0]['high'] - candles[0]['low'])
        for i in range(1, n):
            h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atrs = [0.0] * n
        if n >= period:
            atrs[period - 1] = sum(trs[:period]) / period
            for i in range(period, n):
                atrs[i] = (atrs[i - 1] * (period - 1) + trs[i]) / period
        return atrs

    def _calc_rsi(self, closes: List[float], period: int) -> float:
        """Return current RSI value."""
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

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        current = context_candles[-1]
        prev    = context_candles[-2]

        atrs = self._calc_atr(context_candles, self.atr_period)
        current_atr = atrs[-1]
        prev_atr    = atrs[-2] if len(atrs) >= 2 else 0

        closes = [c['close'] for c in context_candles]
        rsi = self._calc_rsi(closes, self.rsi_period)

        volumes = [c['volume'] for c in context_candles[-self.volume_ma_period:]]
        volume_ma = sum(volumes) / len(volumes) if volumes else 0
        current_volume = current['volume']

        volatility_expanding = (prev_atr > 0 and
                                current_atr > prev_atr * self.volatility_expansion_thresh)
        volume_confirmed = (volume_ma > 0 and
                            current_volume > volume_ma * self.volume_threshold)

        if (volatility_expanding and volume_confirmed and
                rsi < self.rsi_buy_threshold and
                current['close'] < prev['close']):
            confidence = 0.5
            if current_atr > prev_atr * self.volatility_expansion_thresh:
                confidence += 0.15
            if current_volume > volume_ma * self.volume_threshold:
                confidence += 0.15
            if rsi < 30:
                confidence += 0.10
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(min(confidence, 1.0), 4),
                'rsi': round(rsi, 2),
                'atr_ratio': round(current_atr / prev_atr if prev_atr > 0 else 0, 3),
            }

        if (volatility_expanding and volume_confirmed and
                rsi > self.rsi_sell_threshold and
                current['close'] > prev['close']):
            confidence = 0.5
            if current_atr > prev_atr * self.volatility_expansion_thresh:
                confidence += 0.15
            if current_volume > volume_ma * self.volume_threshold:
                confidence += 0.15
            if rsi > 70:
                confidence += 0.10
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(min(confidence, 1.0), 4),
                'rsi': round(rsi, 2),
                'atr_ratio': round(current_atr / prev_atr if prev_atr > 0 else 0, 3),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'atr_period': self.atr_period,
            'volume_ma_period': self.volume_ma_period,
            'volume_threshold': self.volume_threshold,
            'rsi_buy_threshold': self.rsi_buy_threshold,
            'rsi_sell_threshold': self.rsi_sell_threshold,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ('atr_period', 'volume_ma_period'):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ('volume_threshold', 'rsi_buy_threshold', 'rsi_sell_threshold',
                    'atr_multiplier', 'volatility_expansion_thresh'):
            if key in params:
                setattr(self, key, float(params[key]))
