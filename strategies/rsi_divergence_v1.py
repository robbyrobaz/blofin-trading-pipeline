#!/usr/bin/env python3
"""
RSI Divergence V1 Strategy — candle-based interface.

Calculates RSI from candle close prices and signals on overbought/oversold.
BUY when RSI <= oversold threshold, SELL when RSI >= overbought threshold.

Evening tuning 2026-02-16: require_trend_for_sell suppresses SELL signals
in ranging markets (requires price to be below window mean).
"""
import os
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy


class RSIDivergenceV1Strategy(BaseStrategy):
    """RSI overbought/oversold signals on candle close prices."""

    name = "rsi_divergence_v1"
    version = "2.0"
    description = "RSI overbought/oversold signals (candle-based)"

    def __init__(self):
        self.rsi_period = int(os.getenv("RSI_PERIOD", "14"))
        self.oversold   = float(os.getenv("RSI_OVERSOLD", "20"))
        self.overbought = float(os.getenv("RSI_OVERBOUGHT", "80"))
        self.require_trend_for_sell = (
            os.getenv("RSI_REQUIRE_TREND_FOR_SELL", "true").lower() == "true"
        )
        self.min_candles = self.rsi_period + 2

    def _calc_rsi(self, closes: List[float]) -> Optional[float]:
        """Calculate RSI from close prices. Returns current RSI or None."""
        if len(closes) < self.rsi_period + 1:
            return None

        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))

        avg_gain = sum(gains[:self.rsi_period]) / self.rsi_period
        avg_loss = sum(losses[:self.rsi_period]) / self.rsi_period

        # Wilder smoothing for remaining values
        for i in range(self.rsi_period, len(gains)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        closes = [c['close'] for c in context_candles]
        rsi = self._calc_rsi(closes)
        if rsi is None:
            return None

        mean_price = sum(closes) / len(closes)
        current_price = closes[-1]

        if rsi <= self.oversold:
            confidence = min(0.80, max(0.55, (self.oversold - rsi) / self.oversold))
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(rsi, 2),
                'threshold': self.oversold,
            }

        if rsi >= self.overbought:
            if self.require_trend_for_sell and current_price >= mean_price:
                return None  # No downtrend confirmation — skip
            confidence = min(0.80, max(0.55, (rsi - self.overbought) / (100 - self.overbought)))
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 4),
                'rsi': round(rsi, 2),
                'threshold': self.overbought,
                'trend_confirmed': current_price < mean_price,
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'rsi_period': self.rsi_period,
            'oversold': self.oversold,
            'overbought': self.overbought,
            'require_trend_for_sell': self.require_trend_for_sell,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'rsi_period' in params:
            self.rsi_period = int(params['rsi_period'])
        if 'oversold' in params:
            self.oversold = float(params['oversold'])
        if 'overbought' in params:
            self.overbought = float(params['overbought'])
        if 'require_trend_for_sell' in params:
            v = params['require_trend_for_sell']
            self.require_trend_for_sell = v if isinstance(v, bool) else str(v).lower() == "true"
