#!/usr/bin/env python3
"""
Orderflow Imbalance Strategy — candle-based interface.

Uses candle close direction as a proxy for buy/sell orderflow pressure.
An "up-candle" (close > prev_close) is treated as buy pressure;
a "down-candle" (close < prev_close) as sell pressure.

When the ratio of up-candles exceeds the imbalance threshold over a
lookback window, signals BUY (and vice versa for SELL).
"""
import os
from typing import Optional, Dict, Any

from .base_strategy import BaseStrategy


class OrderflowImbalanceStrategy(BaseStrategy):
    """Candle-direction orderflow pressure imbalance strategy."""

    name = "orderflow_imbalance"
    version = "2.0"
    description = "Candle close-direction buy/sell pressure imbalance (candle-based)"

    def __init__(self):
        self.pressure_window = int(os.getenv("OFI_PRESSURE_WINDOW", "20"))
        self.confirm_window  = int(os.getenv("OFI_CONFIRM_WINDOW", "5"))
        self.imbalance_threshold = float(os.getenv("OFI_IMBALANCE_THRESHOLD", "0.65"))
        self.min_candles = 10

    def _uptick_ratio(self, candles: list) -> tuple:
        """Return (uptick_ratio, total_directional_candles) for a candle list."""
        upticks = 0
        downticks = 0
        for i in range(1, len(candles)):
            diff = candles[i]['close'] - candles[i - 1]['close']
            if diff > 0:
                upticks += 1
            elif diff < 0:
                downticks += 1
        total = upticks + downticks
        if total == 0:
            return 0.5, 0
        return upticks / total, total

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        # Main pressure window
        main_window = context_candles[-self.pressure_window - 1:]
        main_ratio, main_count = self._uptick_ratio(main_window)
        if main_count < 5:
            return None

        # Confirmation window (most recent candles)
        confirm_window = context_candles[-self.confirm_window - 1:]
        confirm_ratio, confirm_count = self._uptick_ratio(confirm_window)
        if confirm_count < 2:
            return None

        is_buy  = (main_ratio >= self.imbalance_threshold and confirm_ratio >= 0.55)
        is_sell = (main_ratio <= (1 - self.imbalance_threshold) and confirm_ratio <= 0.45)

        if is_buy:
            confidence = min(0.90, 0.50 + (main_ratio - 0.5) * 2)
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(confidence, 3),
                'uptick_ratio': round(main_ratio, 3),
                'confirm_ratio': round(confirm_ratio, 3),
                'main_candles': main_count,
            }

        if is_sell:
            confidence = min(0.90, 0.50 + (0.5 - main_ratio) * 2)
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(confidence, 3),
                'uptick_ratio': round(main_ratio, 3),
                'confirm_ratio': round(confirm_ratio, 3),
                'main_candles': main_count,
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'pressure_window': self.pressure_window,
            'confirm_window': self.confirm_window,
            'imbalance_threshold': self.imbalance_threshold,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'pressure_window' in params:
            self.pressure_window = int(params['pressure_window'])
        if 'confirm_window' in params:
            self.confirm_window = int(params['confirm_window'])
        if 'imbalance_threshold' in params:
            self.imbalance_threshold = float(params['imbalance_threshold'])
