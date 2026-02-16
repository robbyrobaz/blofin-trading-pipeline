#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Detects strong price momentum in a time window."""
    
    name = "momentum"
    version = "1.0"
    description = "Detects strong upward or downward price momentum"
    
    def __init__(self):
        self.window_seconds = int(os.getenv("MOMENTUM_WINDOW_SECONDS", "240"))
        self.up_pct = float(os.getenv("MOMENTUM_UP_PCT", "1.00"))
        self.down_pct = float(os.getenv("MOMENTUM_DOWN_PCT", "-1.00"))
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        window = self._slice_window(prices, ts_ms, self.window_seconds)
        
        if len(window) < 2 or window[0][1] <= 0:
            return None
        
        first_price = window[0][1]
        pct = ((price - first_price) / first_price) * 100.0
        
        if pct >= self.up_pct:
            confidence = min(0.99, max(0.50, pct / max(self.up_pct, 0.01)))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "window_s": self.window_seconds,
                    "change_pct": round(pct, 4),
                }
            )
        elif pct <= self.down_pct:
            confidence = min(0.99, max(0.50, abs(pct) / max(abs(self.down_pct), 0.01)))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "window_s": self.window_seconds,
                    "change_pct": round(pct, 4),
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "up_pct": self.up_pct,
            "down_pct": self.down_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "window_seconds" in params:
            self.window_seconds = int(params["window_seconds"])
        if "up_pct" in params:
            self.up_pct = float(params["up_pct"])
        if "down_pct" in params:
            self.down_pct = float(params["down_pct"])
