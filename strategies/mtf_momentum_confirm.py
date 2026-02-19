#!/usr/bin/env python3
"""
Multi-timeframe momentum confirmation strategy.
15m momentum signal confirmed by 4h trend.
BUY when 15m momentum is positive AND price is above the 4h EMA.
SELL when 15m momentum is negative AND price is below the 4h EMA.
"""
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class MTFMomentumConfirmStrategy(BaseStrategy):
    """Multi-timeframe momentum: 15m signal confirmed by 4h trend."""

    name = "mtf_momentum_confirm"
    version = "1.0"
    description = "Multi-timeframe momentum: 15m momentum signal confirmed by 4h trend"

    def __init__(self):
        # 15m window for momentum signal
        self.momentum_window_seconds = int(os.getenv("MTF_MOM_WINDOW_SECONDS", "900"))
        self.momentum_up_pct = float(os.getenv("MTF_MOM_UP_PCT", "0.5"))
        self.momentum_down_pct = float(os.getenv("MTF_MOM_DOWN_PCT", "-0.5"))
        # 4h window for trend filter
        self.trend_window_seconds = int(os.getenv("MTF_MOM_TREND_WINDOW_SECONDS", "14400"))

    def _compute_ema(self, prices: List[float]) -> Optional[float]:
        """Compute EMA from a price list. Returns None if empty."""
        if not prices:
            return None
        k = 2.0 / (len(prices) + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1.0 - k)
        return ema

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]],
    ) -> Optional[Signal]:
        # 15m momentum window
        mom_window = self._slice_window(prices, ts_ms, self.momentum_window_seconds)
        if len(mom_window) < 2 or mom_window[0][1] <= 0:
            return None

        # 4h trend window
        trend_window = self._slice_window(prices, ts_ms, self.trend_window_seconds)
        if len(trend_window) < 2:
            return None

        first_price = mom_window[0][1]
        mom_pct = (price - first_price) / first_price * 100.0

        trend_prices = [p for _, p in trend_window]
        ema_4h = self._compute_ema(trend_prices)
        if ema_4h is None or ema_4h <= 0:
            return None

        above_ema = price > ema_4h
        below_ema = price < ema_4h
        ema_pct = (price - ema_4h) / ema_4h * 100.0

        if mom_pct >= self.momentum_up_pct and above_ema:
            confidence = min(0.90, max(0.55, mom_pct / max(self.momentum_up_pct, 0.01) * 0.5))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "mom_15m_pct": round(mom_pct, 4),
                    "ema_4h": round(ema_4h, 6),
                    "price_vs_ema_pct": round(ema_pct, 4),
                    "momentum_window_s": self.momentum_window_seconds,
                    "trend_window_s": self.trend_window_seconds,
                },
            )

        if mom_pct <= self.momentum_down_pct and below_ema:
            confidence = min(0.90, max(0.55, abs(mom_pct) / max(abs(self.momentum_down_pct), 0.01) * 0.5))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "mom_15m_pct": round(mom_pct, 4),
                    "ema_4h": round(ema_4h, 6),
                    "price_vs_ema_pct": round(ema_pct, 4),
                    "momentum_window_s": self.momentum_window_seconds,
                    "trend_window_s": self.trend_window_seconds,
                },
            )

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "momentum_window_seconds": self.momentum_window_seconds,
            "momentum_up_pct": self.momentum_up_pct,
            "momentum_down_pct": self.momentum_down_pct,
            "trend_window_seconds": self.trend_window_seconds,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if "momentum_window_seconds" in params:
            self.momentum_window_seconds = int(params["momentum_window_seconds"])
        if "momentum_up_pct" in params:
            self.momentum_up_pct = float(params["momentum_up_pct"])
        if "momentum_down_pct" in params:
            self.momentum_down_pct = float(params["momentum_down_pct"])
        if "trend_window_seconds" in params:
            self.trend_window_seconds = int(params["trend_window_seconds"])
