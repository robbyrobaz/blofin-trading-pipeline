#!/usr/bin/env python3
"""
Orderflow Imbalance Strategy

Uses tick direction (uptick vs downtick) as a proxy for buy/sell orderflow.
Aggregates tick pressure over rolling windows and signals when imbalance
exceeds a threshold.

Note: Blofin data has no explicit trade side (buy/sell). We infer direction
from consecutive tick price changes: uptick (+) vs downtick (-).

Reference: new_strategies_backtest.py → strategy_orderflow_imbalance
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class OrderflowImbalanceStrategy(BaseStrategy):
    """
    Tick-direction orderflow pressure strategy.

    For each time window in the price history:
      - Count price-up ticks (upticks) vs price-down ticks (downticks)
      - Compute uptick ratio = upticks / (upticks + downticks)
      - Signal when ratio exceeds imbalance_threshold (buy pressure)
        or falls below (1 - threshold) (sell pressure)

    Additional confirmation: current window confirms sustained direction.
    """

    name = "orderflow_imbalance"
    version = "1.0"
    description = "Tick-direction buy/sell pressure imbalance"

    def __init__(self):
        # Main pressure window (seconds)
        self.pressure_window_s = int(os.getenv("OFI_PRESSURE_WINDOW_S", "300"))  # 5 min
        # Confirmation window (shorter, more recent)
        self.confirm_window_s  = int(os.getenv("OFI_CONFIRM_WINDOW_S", "60"))   # 1 min
        # Imbalance threshold (e.g. 0.65 = 65% must be upticks)
        self.imbalance_threshold = float(os.getenv("OFI_IMBALANCE_THRESHOLD", "0.65"))
        # Minimum tick count for reliable signal
        self.min_ticks = int(os.getenv("OFI_MIN_TICKS", "20"))
        # Cooldown: minimum seconds between signals per symbol
        self.cooldown_s = int(os.getenv("OFI_COOLDOWN_S", "120"))
        self._last_signal_ts: Dict[str, int] = {}

    def _compute_tick_pressure(
        self,
        prices: List[Tuple[int, float]],
        window_end_ts: int,
        lookback_s: int
    ) -> Tuple[float, int]:
        """
        Compute uptick ratio and tick count for a lookback window.
        Returns (uptick_ratio, tick_count).
        """
        window = self._slice_window(prices, window_end_ts, lookback_s)
        if len(window) < 2:
            return 0.5, 0

        prices_in_window = [p for _, p in window]
        upticks   = 0
        downticks = 0
        for i in range(1, len(prices_in_window)):
            diff = prices_in_window[i] - prices_in_window[i - 1]
            if diff > 0:
                upticks += 1
            elif diff < 0:
                downticks += 1

        total = upticks + downticks
        if total == 0:
            return 0.5, 0

        return upticks / total, total

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:

        if len(prices) < self.min_ticks:
            return None

        # Check cooldown
        last_ts = self._last_signal_ts.get(symbol, 0)
        if ts_ms - last_ts < self.cooldown_s * 1000:
            return None

        # Compute pressure over main window
        main_ratio, main_ticks = self._compute_tick_pressure(
            prices, ts_ms, self.pressure_window_s
        )
        if main_ticks < self.min_ticks:
            return None

        # Compute pressure over confirmation (recent) window
        confirm_ratio, confirm_ticks = self._compute_tick_pressure(
            prices, ts_ms, self.confirm_window_s
        )
        if confirm_ticks < 5:
            return None

        is_buy_pressure  = (main_ratio >= self.imbalance_threshold and
                            confirm_ratio >= 0.55)
        is_sell_pressure = (main_ratio <= (1 - self.imbalance_threshold) and
                            confirm_ratio <= 0.45)

        if is_buy_pressure:
            confidence = min(0.90, 0.50 + (main_ratio - 0.5) * 2)
            self._last_signal_ts[symbol] = ts_ms
            return Signal(
                symbol=symbol, signal="BUY", strategy=self.name,
                confidence=round(confidence, 3),
                details={
                    "uptick_ratio": round(main_ratio, 3),
                    "confirm_ratio": round(confirm_ratio, 3),
                    "main_ticks": main_ticks,
                    "confirm_ticks": confirm_ticks,
                }
            )

        if is_sell_pressure:
            confidence = min(0.90, 0.50 + (0.5 - main_ratio) * 2)
            self._last_signal_ts[symbol] = ts_ms
            return Signal(
                symbol=symbol, signal="SELL", strategy=self.name,
                confidence=round(confidence, 3),
                details={
                    "uptick_ratio": round(main_ratio, 3),
                    "confirm_ratio": round(confirm_ratio, 3),
                    "main_ticks": main_ticks,
                    "confirm_ticks": confirm_ticks,
                }
            )

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "pressure_window_s": self.pressure_window_s,
            "confirm_window_s": self.confirm_window_s,
            "imbalance_threshold": self.imbalance_threshold,
            "min_ticks": self.min_ticks,
            "cooldown_s": self.cooldown_s,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ("pressure_window_s", "confirm_window_s", "min_ticks", "cooldown_s"):
            if key in params:
                setattr(self, key, int(params[key]))
        if "imbalance_threshold" in params:
            self.imbalance_threshold = float(params["imbalance_threshold"])
