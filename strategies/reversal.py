#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class ReversalStrategy(BaseStrategy):
    """Detects price bouncing from local lows or rejecting from local highs."""
    
    name = "reversal"
    version = "1.0"
    description = "Detects price reversals from local extremes"
    
    def __init__(self):
        self.lookback_seconds = int(os.getenv("REVERSAL_LOOKBACK_SECONDS", "600"))
        self.bounce_pct = float(os.getenv("REVERSAL_BOUNCE_PCT", "0.35"))
    
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
        
        price_values = [p for _, p in window]
        low = min(price_values)
        high = max(price_values)
        
        # Check for bounce from low
        if low > 0:
            bounce_pct = ((price - low) / low) * 100.0
            if bounce_pct >= self.bounce_pct:
                return Signal(
                    symbol=symbol,
                    signal="BUY",
                    strategy=self.name,
                    confidence=0.65,
                    details={
                        "lookback_s": self.lookback_seconds,
                        "low": low,
                        "bounce_pct": round(bounce_pct, 4),
                    }
                )
        
        # Check for rejection from high
        if high > 0:
            reject_pct = ((high - price) / high) * 100.0
            if reject_pct >= self.bounce_pct:
                return Signal(
                    symbol=symbol,
                    signal="SELL",
                    strategy=self.name,
                    confidence=0.65,
                    details={
                        "lookback_s": self.lookback_seconds,
                        "high": high,
                        "reject_pct": round(reject_pct, 4),
                    }
                )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "lookback_seconds": self.lookback_seconds,
            "bounce_pct": self.bounce_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "lookback_seconds" in params:
            self.lookback_seconds = int(params["lookback_seconds"])
        if "bounce_pct" in params:
            self.bounce_pct = float(params["bounce_pct"])
