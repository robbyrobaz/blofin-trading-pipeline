#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class VolumeSpikeStrategy(BaseStrategy):
    """Detects volume surges with price direction confirmation."""
    
    name = "volume_spike"
    version = "1.0"
    description = "Detects volume surges (2-3x average) with price direction confirmation"
    
    def __init__(self):
        self.lookback_seconds = int(os.getenv("VOLUME_SPIKE_LOOKBACK_SECONDS", "900"))
        self.spike_multiplier = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "2.5"))
        self.min_price_move_pct = float(os.getenv("VOLUME_SPIKE_MIN_PRICE_MOVE_PCT", "0.3"))
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        volume_window = self._slice_window(volumes, ts_ms, self.lookback_seconds)
        price_window = self._slice_window(prices, ts_ms, self.lookback_seconds)
        
        if len(volume_window) < 10 or volume <= 0:
            return None
        
        # Calculate average volume (excluding current tick)
        past_volumes = [v for _, v in volume_window[:-1]]
        if not past_volumes:
            return None
        
        avg_volume = sum(past_volumes) / len(past_volumes)
        
        if avg_volume <= 0:
            return None
        
        # Check for volume spike
        volume_ratio = volume / avg_volume
        
        if volume_ratio < self.spike_multiplier:
            return None
        
        # Confirm with price direction
        if len(price_window) < 5:
            return None
        
        recent_prices = [p for _, p in price_window[-5:]]
        avg_recent_price = sum(recent_prices) / len(recent_prices)
        
        if avg_recent_price <= 0:
            return None
        
        price_move_pct = ((price - avg_recent_price) / avg_recent_price) * 100.0
        
        # Volume spike with upward price movement
        if price_move_pct >= self.min_price_move_pct:
            confidence = min(0.90, 0.60 + (volume_ratio / 10.0))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "volume_ratio": round(volume_ratio, 2),
                    "avg_volume": round(avg_volume, 2),
                    "current_volume": round(volume, 2),
                    "price_move_pct": round(price_move_pct, 4),
                }
            )
        
        # Volume spike with downward price movement
        elif price_move_pct <= -self.min_price_move_pct:
            confidence = min(0.90, 0.60 + (volume_ratio / 10.0))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "volume_ratio": round(volume_ratio, 2),
                    "avg_volume": round(avg_volume, 2),
                    "current_volume": round(volume, 2),
                    "price_move_pct": round(price_move_pct, 4),
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "lookback_seconds": self.lookback_seconds,
            "spike_multiplier": self.spike_multiplier,
            "min_price_move_pct": self.min_price_move_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "lookback_seconds" in params:
            self.lookback_seconds = int(params["lookback_seconds"])
        if "spike_multiplier" in params:
            self.spike_multiplier = float(params["spike_multiplier"])
        if "min_price_move_pct" in params:
            self.min_price_move_pct = float(params["min_price_move_pct"])
