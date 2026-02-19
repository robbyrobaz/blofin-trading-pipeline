#!/usr/bin/env python3
"""
Ensemble Top-3 Strategy — candle-based interface.

Combines signals from 3 mean-reversion sub-strategies:
  1. VWAP Deviation  (price vs volume-weighted average)
  2. Z-Score Mean Reversion
  3. Bollinger Band Mean Reversion

Signal logic: Enter only when ≥ 2 of 3 agree on direction.
"""
import os
import numpy as np
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy


class EnsembleTop3Strategy(BaseStrategy):
    """Voting ensemble of top-3 mean-reversion strategies (candle-based)."""

    name = "ensemble_top3"
    version = "2.0"
    description = "Voting ensemble: 2/3 agreement required (VWAP + ZScore + BB)"

    def __init__(self):
        self.vwap_period   = int(os.getenv("ENS_VWAP_PERIOD", "20"))
        self.vwap_dev_pct  = float(os.getenv("ENS_VWAP_DEV_PCT", "1.0"))
        self.zscore_period = int(os.getenv("ENS_ZSCORE_PERIOD", "20"))
        self.zscore_thresh = float(os.getenv("ENS_ZSCORE_THRESH", "2.0"))
        self.bb_period     = int(os.getenv("ENS_BB_PERIOD", "20"))
        self.bb_std        = float(os.getenv("ENS_BB_STD", "2.0"))
        self.min_votes     = int(os.getenv("ENS_MIN_VOTES", "2"))

    def _signal_vwap(self, candles: list) -> int:
        """VWAP deviation: +1 (below → buy), -1 (above → sell), 0 (neutral)."""
        period = min(self.vwap_period, len(candles))
        recent = candles[-period:]
        total_v = sum(c['volume'] for c in recent)
        if total_v <= 0:
            return 0
        vwap = sum(c['close'] * c['volume'] for c in recent) / total_v
        price = candles[-1]['close']
        if vwap <= 0:
            return 0
        dev = (price - vwap) / vwap * 100
        if dev <= -self.vwap_dev_pct:
            return 1
        elif dev >= self.vwap_dev_pct:
            return -1
        return 0

    def _signal_zscore(self, candles: list) -> int:
        """Z-score mean reversion."""
        period = min(self.zscore_period, len(candles))
        closes = np.array([c['close'] for c in candles[-period:]])
        mean = np.mean(closes)
        std  = np.std(closes)
        if std <= 0:
            return 0
        price = candles[-1]['close']
        z = (price - mean) / std
        if z <= -self.zscore_thresh:
            return 1
        elif z >= self.zscore_thresh:
            return -1
        return 0

    def _signal_bb(self, candles: list) -> int:
        """Bollinger Band crossing."""
        period = min(self.bb_period, len(candles))
        if len(candles) < 2:
            return 0
        closes = np.array([c['close'] for c in candles[-period:]])
        mean = np.mean(closes)
        std  = np.std(closes)
        if std <= 0:
            return 0
        upper = mean + self.bb_std * std
        lower = mean - self.bb_std * std
        price      = candles[-1]['close']
        prev_price = candles[-2]['close']
        if price <= lower and prev_price > lower:
            return 1
        elif price >= upper and prev_price < upper:
            return -1
        return 0

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        min_needed = max(self.vwap_period, self.zscore_period, self.bb_period) + 2
        if len(context_candles) < min_needed:
            return None

        v1 = self._signal_vwap(context_candles)
        v2 = self._signal_zscore(context_candles)
        v3 = self._signal_bb(context_candles)
        total = v1 + v2 + v3

        if total >= self.min_votes:
            confidence = min(0.95, 0.55 + (total - 1) * 0.15)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 3),
                'votes': f"{total}/3",
                'vwap_vote': v1, 'zscore_vote': v2, 'bb_vote': v3,
            }
        elif total <= -self.min_votes:
            confidence = min(0.95, 0.55 + (abs(total) - 1) * 0.15)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 3),
                'votes': f"{total}/3",
                'vwap_vote': v1, 'zscore_vote': v2, 'bb_vote': v3,
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "vwap_period": self.vwap_period,
            "vwap_dev_pct": self.vwap_dev_pct,
            "zscore_period": self.zscore_period,
            "zscore_thresh": self.zscore_thresh,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "min_votes": self.min_votes,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ("vwap_period", "zscore_period", "bb_period", "min_votes"):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ("vwap_dev_pct", "zscore_thresh", "bb_std"):
            if key in params:
                setattr(self, key, float(params[key]))
