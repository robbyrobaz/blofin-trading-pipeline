#!/usr/bin/env python3
"""
Multi-Timeframe Trend Alignment — candle-based interface.

Simulates multi-timeframe analysis from 1m candle data:
  - "Short" timeframe: 5-candle RSI (recent momentum)
  - "Medium" timeframe: 20-candle EMA (local trend)
  - "Long" timeframe: 50-candle EMA slope (macro trend direction)

BUY: RSI oversold on short TF AND price above 20-EMA AND 50-EMA sloping up
SELL: RSI overbought on short TF AND price below 20-EMA AND 50-EMA sloping down
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class MTFTrendAlignStrategy(BaseStrategy):
    """Multi-timeframe trend alignment using simulated TF aggregation from 1m candles."""

    name = "mtf_trend_align"
    version = "2.0"
    description = "Multi-TF alignment: 5-bar RSI entry + 20-bar EMA trend + 50-bar EMA macro"

    def __init__(self):
        self.min_candles = 25           # need enough for indicators
        self.rsi_period = 14            # RSI filter (not overbought/oversold)
        self.rsi_neutral_low = 35       # RSI below this = don't take SELL
        self.rsi_neutral_high = 65      # RSI above this = don't take BUY
        self.fast_ema_period = 9        # fast EMA for crossover
        self.slow_ema_period = 21       # slow EMA for crossover
        self.slope_lookback = 3         # candles to measure EMA slope

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _calc_ema_series(self, closes: List[float], period: int) -> List[float]:
        """Return EMA series (starts after `period` values)."""
        if len(closes) < period:
            return []
        k = 2.0 / (period + 1.0)
        ema = [sum(closes[:period]) / period]
        for v in closes[period:]:
            ema.append(v * k + ema[-1] * (1.0 - k))
        return ema

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

        # RSI (filter only — prevents taking signals at extreme exhaustion)
        rsi = self._calc_rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # Fast EMA (9-period) and slow EMA (21-period) for crossover
        fast_ema_series = self._calc_ema_series(closes, self.fast_ema_period)
        slow_ema_series = self._calc_ema_series(closes, self.slow_ema_period)
        if len(fast_ema_series) < 2 or len(slow_ema_series) < 2:
            return None

        # Align series lengths (slow is shorter; fast has more values)
        offset = self.slow_ema_period - self.fast_ema_period
        if len(fast_ema_series) <= offset:
            return None

        fast_now = fast_ema_series[-1]
        fast_prev = fast_ema_series[-2]
        slow_now = slow_ema_series[-1]
        slow_prev = slow_ema_series[-2]

        # Crossover detection
        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        # BUY: fast EMA crosses above slow EMA, RSI not overbought
        if crossed_up and rsi < self.rsi_neutral_high:
            confidence = min(0.83, 0.63 + min(abs(fast_now - slow_now) / slow_now * 1000, 0.15))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(rsi, 2),
                'fast_ema': round(fast_now, 6),
                'slow_ema': round(slow_now, 6),
                'ema_spread_pct': round((fast_now - slow_now) / slow_now * 100, 4),
            }

        # SELL: fast EMA crosses below slow EMA, RSI not oversold
        if crossed_down and rsi > self.rsi_neutral_low:
            confidence = min(0.83, 0.63 + min(abs(fast_now - slow_now) / slow_now * 1000, 0.15))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(rsi, 2),
                'fast_ema': round(fast_now, 6),
                'slow_ema': round(slow_now, 6),
                'ema_spread_pct': round((fast_now - slow_now) / slow_now * 100, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'rsi_period': self.rsi_period,
            'rsi_neutral_low': self.rsi_neutral_low,
            'rsi_neutral_high': self.rsi_neutral_high,
            'fast_ema_period': self.fast_ema_period,
            'slow_ema_period': self.slow_ema_period,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'rsi_neutral_low' in params:
            self.rsi_neutral_low = float(params['rsi_neutral_low'])
        if 'rsi_neutral_high' in params:
            self.rsi_neutral_high = float(params['rsi_neutral_high'])
        if 'fast_ema_period' in params:
            self.fast_ema_period = int(params['fast_ema_period'])
        if 'slow_ema_period' in params:
            self.slow_ema_period = int(params['slow_ema_period'])
