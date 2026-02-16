#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """Detects price breaking above/below recent high/low with buffer."""
    
    name = "breakout"
    version = "1.0"
    description = "Detects price breakouts above recent highs or below recent lows"
    
    def __init__(self):
        self.lookback_seconds = int(os.getenv("BREAKOUT_LOOKBACK_SECONDS", "900"))
        self.buffer_pct = float(os.getenv("BREAKOUT_BUFFER_PCT", "0.18"))
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        window = self._slice_window(prices, ts_ms, self.lookback_seconds)
        
        if len(window) < 10:
            return None
        
        prev_prices = [p for _, p in window[:-1]]
        hi = max(prev_prices)
        lo = min(prev_prices)
        
        up_threshold = hi * (1 + self.buffer_pct / 100.0)
        dn_threshold = lo * (1 - self.buffer_pct / 100.0)
        
        if price >= up_threshold:
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=0.70,
                details={
                    "lookback_s": self.lookback_seconds,
                    "prev_high": hi,
                    "threshold": up_threshold,
                }
            )
        elif price <= dn_threshold:
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=0.70,
                details={
                    "lookback_s": self.lookback_seconds,
                    "prev_low": lo,
                    "threshold": dn_threshold,
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "lookback_seconds": self.lookback_seconds,
            "buffer_pct": self.buffer_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "lookback_seconds" in params:
            self.lookback_seconds = int(params["lookback_seconds"])
        if "buffer_pct" in params:
            self.buffer_pct = float(params["buffer_pct"])
