#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class VWAPReversionStrategy(BaseStrategy):
    """Detects mean reversion from VWAP deviation."""
    
    name = "vwap_reversion"
    version = "1.0"
    description = "Detects price deviation from VWAP for mean reversion trades"
    
    def __init__(self):
        self.lookback_seconds = int(os.getenv("VWAP_LOOKBACK_SECONDS", "1200"))
        self.deviation_pct = float(os.getenv("VWAP_DEVIATION_PCT", "1.20"))
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        price_window = self._slice_window(prices, ts_ms, self.lookback_seconds)
        volume_window = self._slice_window(volumes, ts_ms, self.lookback_seconds)
        
        if len(price_window) < 10 or len(volume_window) != len(price_window):
            return None
        
        # Calculate VWAP
        total_pv = sum(p * v for (_, p), (_, v) in zip(price_window, volume_window))
        total_v = sum(v for _, v in volume_window)
        
        if total_v <= 0:
            return None
        
        vwap = total_pv / total_v
        if vwap <= 0:
            return None
        
        deviation_pct = ((price - vwap) / vwap) * 100.0
        
        # Buy when price is below VWAP
        if deviation_pct <= -self.deviation_pct:
            confidence = min(0.85, max(0.60, abs(deviation_pct) / max(self.deviation_pct, 0.01)))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "lookback_s": self.lookback_seconds,
                    "vwap": round(vwap, 6),
                    "deviation_pct": round(deviation_pct, 4),
                }
            )
        # Sell when price is above VWAP
        elif deviation_pct >= self.deviation_pct:
            confidence = min(0.85, max(0.60, abs(deviation_pct) / max(self.deviation_pct, 0.01)))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "lookback_s": self.lookback_seconds,
                    "vwap": round(vwap, 6),
                    "deviation_pct": round(deviation_pct, 4),
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "lookback_seconds": self.lookback_seconds,
            "deviation_pct": self.deviation_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "lookback_seconds" in params:
            self.lookback_seconds = int(params["lookback_seconds"])
        if "deviation_pct" in params:
            self.deviation_pct = float(params["deviation_pct"])
