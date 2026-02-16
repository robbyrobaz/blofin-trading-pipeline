#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

from .base_strategy import BaseStrategy, Signal


class SupportResistanceStrategy(BaseStrategy):
    """Detects support/resistance levels via price clustering and rejection."""
    
    name = "support_resistance"
    version = "1.0"
    description = "Detects support/resistance levels via price clustering and rejection"
    
    def __init__(self):
        self.lookback_seconds = int(os.getenv("SR_LOOKBACK_SECONDS", "1800"))
        self.cluster_tolerance_pct = float(os.getenv("SR_CLUSTER_TOLERANCE_PCT", "0.5"))
        self.min_touches = int(os.getenv("SR_MIN_TOUCHES", "3"))
        self.rejection_pct = float(os.getenv("SR_REJECTION_PCT", "0.25"))
    
    def _find_levels(self, prices: List[float]) -> List[float]:
        """Find clustered price levels."""
        if not prices:
            return []
        
        # Group prices into clusters
        clusters = defaultdict(list)
        for p in prices:
            # Find if price belongs to existing cluster
            found_cluster = False
            for level in list(clusters.keys()):
                tolerance = level * (self.cluster_tolerance_pct / 100.0)
                if abs(p - level) <= tolerance:
                    clusters[level].append(p)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters[p].append(p)
        
        # Filter clusters with minimum touches
        levels = []
        for level, touches in clusters.items():
            if len(touches) >= self.min_touches:
                # Use average of cluster as the level
                avg_level = sum(touches) / len(touches)
                levels.append(avg_level)
        
        return sorted(levels)
    
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
        
        if len(window) < self.min_touches * 2:
            return None
        
        price_values = [p for _, p in window[:-1]]  # Exclude current price
        levels = self._find_levels(price_values)
        
        if not levels:
            return None
        
        # Check for support bounce (price near support, moving up)
        for level in levels:
            if price < level:
                continue
            
            distance_pct = ((price - level) / level) * 100.0
            
            # Price recently bounced from support
            if 0 < distance_pct <= self.rejection_pct:
                # Check if price was below level recently
                recent_prices = [p for _, p in window[-5:]]
                if any(p <= level for p in recent_prices):
                    confidence = min(0.80, 0.60 + (len([p for p in price_values if abs(p - level) / level * 100 <= self.cluster_tolerance_pct]) * 0.05))
                    return Signal(
                        symbol=symbol,
                        signal="BUY",
                        strategy=self.name,
                        confidence=confidence,
                        details={
                            "level": round(level, 6),
                            "touches": len([p for p in price_values if abs(p - level) / level * 100 <= self.cluster_tolerance_pct]),
                            "distance_pct": round(distance_pct, 4),
                            "type": "support_bounce",
                        }
                    )
        
        # Check for resistance rejection (price near resistance, moving down)
        for level in levels:
            if price > level:
                continue
            
            distance_pct = ((level - price) / level) * 100.0
            
            # Price recently rejected at resistance
            if 0 < distance_pct <= self.rejection_pct:
                # Check if price was above level recently
                recent_prices = [p for _, p in window[-5:]]
                if any(p >= level for p in recent_prices):
                    confidence = min(0.80, 0.60 + (len([p for p in price_values if abs(p - level) / level * 100 <= self.cluster_tolerance_pct]) * 0.05))
                    return Signal(
                        symbol=symbol,
                        signal="SELL",
                        strategy=self.name,
                        confidence=confidence,
                        details={
                            "level": round(level, 6),
                            "touches": len([p for p in price_values if abs(p - level) / level * 100 <= self.cluster_tolerance_pct]),
                            "distance_pct": round(distance_pct, 4),
                            "type": "resistance_rejection",
                        }
                    )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "lookback_seconds": self.lookback_seconds,
            "cluster_tolerance_pct": self.cluster_tolerance_pct,
            "min_touches": self.min_touches,
            "rejection_pct": self.rejection_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "lookback_seconds" in params:
            self.lookback_seconds = int(params["lookback_seconds"])
        if "cluster_tolerance_pct" in params:
            self.cluster_tolerance_pct = float(params["cluster_tolerance_pct"])
        if "min_touches" in params:
            self.min_touches = int(params["min_touches"])
        if "rejection_pct" in params:
            self.rejection_pct = float(params["rejection_pct"])
