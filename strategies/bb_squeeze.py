#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class BBSqueezeStrategy(BaseStrategy):
    """Detects Bollinger Band squeeze breakouts."""
    
    name = "bb_squeeze"
    version = "1.0"
    description = "Detects breakouts from Bollinger Band squeeze conditions"
    
    def __init__(self):
        self.lookback_seconds = int(os.getenv("BB_LOOKBACK_SECONDS", "1200"))
        self.std_mult = float(os.getenv("BB_STD_MULT", "2.0"))
        self.squeeze_threshold = float(os.getenv("BB_SQUEEZE_THRESHOLD", "0.3"))
    
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
        
        if len(window) < 20:
            return None
        
        price_values = [p for _, p in window]
        mean = sum(price_values) / len(price_values)
        variance = sum((p - mean) ** 2 for p in price_values) / len(price_values)
        std = variance ** 0.5
        
        if mean <= 0 or std <= 0:
            return None
        
        upper_band = mean + (self.std_mult * std)
        lower_band = mean - (self.std_mult * std)
        band_width_pct = ((upper_band - lower_band) / mean) * 100.0
        
        # Check if bands are tight (squeeze)
        is_squeeze = band_width_pct <= (self.squeeze_threshold * 100)
        
        if not is_squeeze:
            return None
        
        # Check for breakout direction
        if price > upper_band:
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=0.75,
                details={
                    "lookback_s": self.lookback_seconds,
                    "mean": round(mean, 6),
                    "upper": round(upper_band, 6),
                    "band_width_pct": round(band_width_pct, 4),
                }
            )
        elif price < lower_band:
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=0.75,
                details={
                    "lookback_s": self.lookback_seconds,
                    "mean": round(mean, 6),
                    "lower": round(lower_band, 6),
                    "band_width_pct": round(band_width_pct, 4),
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "lookback_seconds": self.lookback_seconds,
            "std_mult": self.std_mult,
            "squeeze_threshold": self.squeeze_threshold,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "lookback_seconds" in params:
            self.lookback_seconds = int(params["lookback_seconds"])
        if "std_mult" in params:
            self.std_mult = float(params["std_mult"])
        if "squeeze_threshold" in params:
            self.squeeze_threshold = float(params["squeeze_threshold"])
