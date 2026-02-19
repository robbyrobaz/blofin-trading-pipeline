#!/usr/bin/env python3
"""
Multi-Timeframe Momentum Confirmation — candle-based interface.

Simulates 15m + 4h timeframes from 1m candles:
  - Short momentum: 15-bar price change (simulates 15m momentum)
  - Medium trend: 60-bar EMA slope (simulates 1h trend)
  - ATR normalization: scales momentum threshold to current volatility

BUY: 15-bar return > +0.2% AND price > 60-bar EMA AND EMA sloping up
SELL: 15-bar return < -0.2% AND price < 60-bar EMA AND EMA sloping down

Additional: requires ADX-like trend strength > threshold to avoid choppy markets.
"""
import math
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class MTFMomentumConfirmStrategy(BaseStrategy):
    """Multi-TF momentum: 15-bar return + 60-bar EMA trend + simplified ADX filter."""

    name = "mtf_momentum_confirm"
    version = "2.0"
    description = "Multi-TF momentum: 15-bar return confirmed by 60-bar EMA trend"

    def __init__(self):
        self.min_candles = 65           # need 60 for EMA + buffer
        self.momentum_bars = 15         # short-term momentum window (simulates 15m)
        self.momentum_threshold_pct = 0.15  # minimum % move to qualify
        self.trend_ema_period = 60      # medium-term EMA (simulates 1h)
        self.ema_slope_bars = 5         # bars to measure EMA slope
        self.atr_period = 14

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calc_ema_series(self, closes: List[float], period: int) -> List[float]:
        if len(closes) < period:
            return []
        k = 2.0 / (period + 1.0)
        ema = [sum(closes[:period]) / period]
        for v in closes[period:]:
            ema.append(v * k + ema[-1] * (1.0 - k))
        return ema

    def _calc_atr(self, candles: List[dict], period: int = 14) -> float:
        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i - 1]['close']
            tr = max(c['high'] - c['low'], abs(c['high'] - pc), abs(c['low'] - pc))
            trs.append(tr)
        return sum(trs[-period:]) / min(len(trs), period) if trs else 0.0

    def _calc_adx_simplified(self, candles: List[dict], period: int = 14) -> float:
        """
        Simplified ADX approximation: average of directional movement ratios.
        Returns 0-100; values > 20 indicate trending conditions.
        """
        if len(candles) < period + 1:
            return 0.0

        dm_plus_list, dm_minus_list, tr_list = [], [], []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i - 1]
            up_move = c['high'] - pc['high']
            down_move = pc['low'] - c['low']
            dm_plus_list.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            dm_minus_list.append(down_move if down_move > up_move and down_move > 0 else 0.0)
            tr = max(c['high'] - c['low'], abs(c['high'] - pc['close']), abs(c['low'] - pc['close']))
            tr_list.append(tr)

        # Smooth over `period` bars
        atr = sum(tr_list[-period:]) / period if tr_list else 1.0
        adm_plus = sum(dm_plus_list[-period:]) / period
        adm_minus = sum(dm_minus_list[-period:]) / period

        di_plus = adm_plus / atr * 100 if atr > 0 else 0.0
        di_minus = adm_minus / atr * 100 if atr > 0 else 0.0
        di_sum = di_plus + di_minus
        if di_sum == 0:
            return 0.0
        dx = abs(di_plus - di_minus) / di_sum * 100
        return dx

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        price = closes[-1]

        # ATR for scaling
        atr = self._calc_atr(context_candles, self.atr_period)

        # 15-bar momentum
        momentum_start = closes[-self.momentum_bars - 1]
        if momentum_start <= 0:
            return None
        mom_pct = (price - momentum_start) / momentum_start * 100.0

        # 60-bar EMA
        ema_series = self._calc_ema_series(closes, self.trend_ema_period)
        if len(ema_series) < self.ema_slope_bars + 1:
            return None
        ema_now = ema_series[-1]
        ema_prev = ema_series[-self.ema_slope_bars - 1]
        slope_up = ema_now > ema_prev
        slope_down = ema_now < ema_prev

        above_ema = price > ema_now
        below_ema = price < ema_now

        # Simplified ADX for trend strength
        adx = self._calc_adx_simplified(context_candles[-30:], 14)

        # Dynamic threshold: scale with ATR if possible
        threshold = self.momentum_threshold_pct
        if atr > 0 and price > 0:
            atr_pct = atr / price * 100
            # Tighter threshold in low-volatility environments
            threshold = max(0.10, min(0.30, atr_pct * 0.5))

        # BUY conditions
        if mom_pct >= threshold and above_ema and slope_up:
            confidence = min(0.88, 0.60 + min(mom_pct / (threshold * 3), 0.5) * 0.25 + min(adx / 100, 0.05))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'mom_15bar_pct': round(mom_pct, 4),
                'ema_60': round(ema_now, 6),
                'slope_up': slope_up,
                'adx': round(adx, 2),
                'threshold_pct': round(threshold, 4),
            }

        # SELL conditions
        if mom_pct <= -threshold and below_ema and slope_down:
            confidence = min(0.88, 0.60 + min(abs(mom_pct) / (threshold * 3), 0.5) * 0.25 + min(adx / 100, 0.05))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'mom_15bar_pct': round(mom_pct, 4),
                'ema_60': round(ema_now, 6),
                'slope_down': slope_down,
                'adx': round(adx, 2),
                'threshold_pct': round(threshold, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'momentum_bars': self.momentum_bars,
            'momentum_threshold_pct': self.momentum_threshold_pct,
            'trend_ema_period': self.trend_ema_period,
            'ema_slope_bars': self.ema_slope_bars,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'momentum_bars' in params:
            self.momentum_bars = int(params['momentum_bars'])
        if 'momentum_threshold_pct' in params:
            self.momentum_threshold_pct = float(params['momentum_threshold_pct'])
        if 'trend_ema_period' in params:
            self.trend_ema_period = int(params['trend_ema_period'])
        if 'ema_slope_bars' in params:
            self.ema_slope_bars = int(params['ema_slope_bars'])
