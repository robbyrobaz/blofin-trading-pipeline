#!/usr/bin/env python3
"""
Volatility Regime Switch Strategy — candle-based interface.

Classifies market regime and applies appropriate entry rules:
  - TRENDING  (ADX > 25, ATR in top 60th percentile): EMA alignment crossover
  - RANGING   (ADX < 25): Bollinger Band mean reversion
  - CHAOTIC   (ADX > 45): No trade — sit out
"""
import os
import numpy as np
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy


class VolatilityRegimeSwitchStrategy(BaseStrategy):
    """Regime-aware strategy: trend-follow / mean-revert / sit-out."""

    name = "volatility_regime_switch"
    version = "2.0"
    description = "Regime-aware: trend-follow / mean-revert / sit-out based on ATR + ADX"

    def __init__(self):
        self.atr_period        = int(os.getenv("VRS_ATR_PERIOD", "14"))
        self.adx_period        = int(os.getenv("VRS_ADX_PERIOD", "14"))
        self.adx_trend_thresh  = float(os.getenv("VRS_ADX_TREND_THRESH", "25.0"))
        self.adx_chaos_thresh  = float(os.getenv("VRS_ADX_CHAOS_THRESH", "45.0"))
        self.atr_pct_window    = int(os.getenv("VRS_ATR_PCT_WINDOW", "50"))
        self.atr_pct_threshold = float(os.getenv("VRS_ATR_PCT_THRESHOLD", "0.60"))
        self.min_candles = 40

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        result = np.zeros(len(prices))
        k = 2.0 / (period + 1)
        result[0] = prices[0]
        for i in range(1, len(prices)):
            result[i] = prices[i] * k + result[i - 1] * (1 - k)
        return result

    def _atr(self, high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i]  - close[i - 1]))
        result = np.full(n, np.nan)
        result[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        return result

    def _adx(self, high, low, close, period=14):
        n = len(close)
        if n < period * 2 + 5:
            return np.full(n, np.nan)
        plus_dm  = np.zeros(n)
        minus_dm = np.zeros(n)
        tr       = np.zeros(n)
        for i in range(1, n):
            up   = high[i]    - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i]  = up   if (up > down and up > 0)   else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i]  - close[i - 1]))
        tr_s   = np.full(n, np.nan)
        pdm_s  = np.full(n, np.nan)
        mdm_s  = np.full(n, np.nan)
        if n <= period:
            return np.full(n, np.nan)
        tr_s[period]  = np.sum(tr[1:period + 1])
        pdm_s[period] = np.sum(plus_dm[1:period + 1])
        mdm_s[period] = np.sum(minus_dm[1:period + 1])
        for i in range(period + 1, n):
            tr_s[i]  = tr_s[i-1]  - tr_s[i-1]  / period + tr[i]
            pdm_s[i] = pdm_s[i-1] - pdm_s[i-1] / period + plus_dm[i]
            mdm_s[i] = mdm_s[i-1] - mdm_s[i-1] / period + minus_dm[i]
        pdi = np.where(tr_s > 0, 100 * pdm_s / tr_s, 0.0)
        mdi = np.where(tr_s > 0, 100 * mdm_s / tr_s, 0.0)
        dx  = np.where((pdi + mdi) > 0,
                       100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
        adx_out = np.full(n, np.nan)
        start = period * 2
        if start >= n:
            return adx_out
        valid = dx[period:start]
        if len(valid) > 0:
            adx_out[start - 1] = np.nanmean(valid)
        for i in range(start, n):
            if not np.isnan(adx_out[i - 1]):
                adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
        return adx_out

    def _sma(self, prices, period):
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def _bollinger(self, prices, period=20, std_mult=2.0):
        mid = self._sma(prices, period)
        std = np.array([
            np.std(prices[max(0, i - period + 1):i + 1]) if i >= period - 1 else np.nan
            for i in range(len(prices))
        ])
        return mid + std_mult * std, mid, mid - std_mult * std

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        high   = np.array([c['high']   for c in context_candles])
        low    = np.array([c['low']    for c in context_candles])
        closes = np.array([c['close']  for c in context_candles])
        n = len(closes)

        atr_vals = self._atr(high, low, closes, self.atr_period)
        adx_vals = self._adx(high, low, closes, self.adx_period)

        i = n - 1
        if np.isnan(atr_vals[i]) or np.isnan(adx_vals[i]):
            return None

        # ATR percentile
        lookback = min(self.atr_pct_window, n)
        recent_atrs = atr_vals[max(0, i - lookback):i]
        valid_atrs  = recent_atrs[~np.isnan(recent_atrs)]
        if len(valid_atrs) < 5:
            return None

        atr_pct = float(np.sum(valid_atrs <= atr_vals[i])) / len(valid_atrs)
        adx_val = adx_vals[i]

        # Classify regime
        if adx_val > self.adx_chaos_thresh:
            return None  # CHAOTIC — sit out

        if adx_val > self.adx_trend_thresh and atr_pct > self.atr_pct_threshold:
            regime = "TRENDING"
        else:
            regime = "RANGING"

        if regime == "TRENDING":
            e5  = self._ema(closes, 5)
            e13 = self._ema(closes, 13)
            e21 = self._ema(closes, 21)
            if i < 1:
                return None
            if e5[i] > e13[i] > e21[i] and e5[i - 1] <= e13[i - 1]:
                return {
                    'signal': 'BUY',
                    'strategy': self.name,
                    'confidence': 0.65,
                    'regime': 'TRENDING',
                    'adx': round(adx_val, 1),
                    'atr_pct': round(atr_pct, 2),
                }
            if e5[i] < e13[i] < e21[i] and e5[i - 1] >= e13[i - 1]:
                return {
                    'signal': 'SELL',
                    'strategy': self.name,
                    'confidence': 0.65,
                    'regime': 'TRENDING',
                    'adx': round(adx_val, 1),
                    'atr_pct': round(atr_pct, 2),
                }

        elif regime == "RANGING":
            bb_upper, _, bb_lower = self._bollinger(closes, 20, 2.0)
            if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
                return None
            if i < 1:
                return None
            if closes[i] <= bb_lower[i] and closes[i - 1] > bb_lower[i - 1]:
                return {
                    'signal': 'BUY',
                    'strategy': self.name,
                    'confidence': 0.60,
                    'regime': 'RANGING',
                    'adx': round(adx_val, 1),
                    'bb_lower': round(bb_lower[i], 6),
                }
            if closes[i] >= bb_upper[i] and closes[i - 1] < bb_upper[i - 1]:
                return {
                    'signal': 'SELL',
                    'strategy': self.name,
                    'confidence': 0.60,
                    'regime': 'RANGING',
                    'adx': round(adx_val, 1),
                    'bb_upper': round(bb_upper[i], 6),
                }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "atr_period": self.atr_period,
            "adx_period": self.adx_period,
            "adx_trend_thresh": self.adx_trend_thresh,
            "adx_chaos_thresh": self.adx_chaos_thresh,
            "atr_pct_window": self.atr_pct_window,
            "atr_pct_threshold": self.atr_pct_threshold,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ("atr_period", "adx_period", "atr_pct_window"):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ("adx_trend_thresh", "adx_chaos_thresh", "atr_pct_threshold"):
            if key in params:
                setattr(self, key, float(params[key]))
