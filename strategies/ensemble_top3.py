#!/usr/bin/env python3
"""
Ensemble Top-3 Strategy

Combines signals from the 3 best-performing strategies:
  1. VWAP Deviation (WR 51.8% in library backtest)
  2. Z-Score Mean Reversion (WR 49.0%)
  3. BB Mean Reversion (WR 49.6%)

Signal logic: Enter only when ≥ 2 of 3 agree on direction.
Expected: higher win rate, fewer trades (precision over recall).

Reference: new_strategies_backtest.py → strategy_ensemble_top3
           strategy_library_summary.json (top performers)
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class EnsembleTop3Strategy(BaseStrategy):
    """
    Voting ensemble of top-3 mean-reversion strategies.

    Requires 2/3 sub-strategy agreement before signaling.
    This filters weak signals and targets higher-confidence setups.
    """

    name = "ensemble_top3"
    version = "1.0"
    description = "Voting ensemble: 2/3 agreement required (VWAP + ZScore + BB)"

    def __init__(self):
        # VWAP parameters
        self.vwap_period   = int(os.getenv("ENS_VWAP_PERIOD", "20"))
        self.vwap_dev_pct  = float(os.getenv("ENS_VWAP_DEV_PCT", "1.0"))
        # Z-score parameters
        self.zscore_period = int(os.getenv("ENS_ZSCORE_PERIOD", "20"))
        self.zscore_thresh = float(os.getenv("ENS_ZSCORE_THRESH", "2.0"))
        # BB parameters
        self.bb_period     = int(os.getenv("ENS_BB_PERIOD", "20"))
        self.bb_std        = float(os.getenv("ENS_BB_STD", "2.0"))
        # Min votes required
        self.min_votes     = int(os.getenv("ENS_MIN_VOTES", "2"))

    # ── Sub-strategy signal evaluators ──

    def _signal_vwap(
        self, price: float,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> int:
        """VWAP deviation: returns +1 (below VWAP → buy) / -1 (above → sell) / 0."""
        period = self.vwap_period
        if len(prices) < period:
            return 0

        recent_prices  = [p for _, p in prices[-period:]]
        recent_volumes = [v for _, v in volumes[-period:]] if len(volumes) >= period else [1.0] * period

        # Use tick counts as volume proxy if volumes are all 1s
        pv = np.array(recent_prices) * np.array(recent_volumes)
        total_v = sum(recent_volumes)
        if total_v <= 0:
            return 0

        vwap_val = sum(pv) / total_v
        dev = (price - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0

        if dev <= -self.vwap_dev_pct:
            return 1   # Below VWAP → mean-revert BUY
        elif dev >= self.vwap_dev_pct:
            return -1  # Above VWAP → mean-revert SELL
        return 0

    def _signal_zscore(
        self, price: float,
        prices: List[Tuple[int, float]]
    ) -> int:
        """Z-score mean reversion."""
        period = self.zscore_period
        if len(prices) < period:
            return 0

        recent = np.array([p for _, p in prices[-period:]])
        mean   = np.mean(recent)
        std    = np.std(recent)

        if std <= 0:
            return 0

        z = (price - mean) / std
        if z <= -self.zscore_thresh:
            return 1   # Extreme below mean → BUY
        elif z >= self.zscore_thresh:
            return -1  # Extreme above mean → SELL
        return 0

    def _signal_bb(
        self, price: float,
        prices: List[Tuple[int, float]],
        prev_price: float
    ) -> int:
        """BB mean reversion: price touches band (crossing)."""
        period   = self.bb_period
        std_mult = self.bb_std

        if len(prices) < period:
            return 0

        recent = np.array([p for _, p in prices[-period:]])
        mean   = np.mean(recent)
        std    = np.std(recent)

        if std <= 0:
            return 0

        upper = mean + std_mult * std
        lower = mean - std_mult * std

        # Current price crossed band
        if price <= lower and prev_price > lower:
            return 1   # Broke through lower → BUY
        elif price >= upper and prev_price < upper:
            return -1  # Broke through upper → SELL
        return 0

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:

        if len(prices) < max(self.vwap_period, self.zscore_period, self.bb_period) + 2:
            return None

        prev_price = prices[-2][1] if len(prices) >= 2 else price

        v1 = self._signal_vwap(price, prices, volumes)
        v2 = self._signal_zscore(price, prices)
        v3 = self._signal_bb(price, prices, prev_price)

        total_votes = v1 + v2 + v3  # Range: -3 to +3

        if total_votes >= self.min_votes:
            # Confidence increases with agreement strength
            confidence = min(0.95, 0.55 + (total_votes - 1) * 0.15)
            return Signal(
                symbol=symbol, signal="BUY", strategy=self.name,
                confidence=round(confidence, 3),
                details={
                    "votes": f"{total_votes}/3",
                    "vwap_vote": v1, "zscore_vote": v2, "bb_vote": v3,
                }
            )
        elif total_votes <= -self.min_votes:
            confidence = min(0.95, 0.55 + (abs(total_votes) - 1) * 0.15)
            return Signal(
                symbol=symbol, signal="SELL", strategy=self.name,
                confidence=round(confidence, 3),
                details={
                    "votes": f"{total_votes}/3",
                    "vwap_vote": v1, "zscore_vote": v2, "bb_vote": v3,
                }
            )

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
