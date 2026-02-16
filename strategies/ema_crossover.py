#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class EMACrossoverStrategy(BaseStrategy):
    """Detects EMA crossover signals (9/21 EMA by default)."""
    
    name = "ema_crossover"
    version = "1.0"
    description = "Detects exponential moving average crossover signals"
    
    def __init__(self):
        self.fast_period = int(os.getenv("EMA_FAST_PERIOD", "9"))
        self.slow_period = int(os.getenv("EMA_SLOW_PERIOD", "21"))
        self.min_separation_pct = float(os.getenv("EMA_MIN_SEPARATION_PCT", "0.15"))
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate EMA for given prices and period."""
        if len(prices) < period:
            return None
        
        # Start with SMA
        sma = sum(prices[:period]) / period
        multiplier = 2 / (period + 1)
        ema = sma
        
        # Calculate EMA
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        # Need enough data for slow EMA + some history
        if len(prices) < self.slow_period + 5:
            return None
        
        price_values = [p for _, p in prices]
        
        # Calculate current EMAs
        fast_ema = self._calculate_ema(price_values, self.fast_period)
        slow_ema = self._calculate_ema(price_values, self.slow_period)
        
        if fast_ema is None or slow_ema is None or slow_ema == 0:
            return None
        
        # Calculate previous EMAs (one tick back) to detect crossover
        prev_price_values = price_values[:-1]
        if len(prev_price_values) < self.slow_period:
            return None
        
        prev_fast_ema = self._calculate_ema(prev_price_values, self.fast_period)
        prev_slow_ema = self._calculate_ema(prev_price_values, self.slow_period)
        
        if prev_fast_ema is None or prev_slow_ema is None:
            return None
        
        # Check for crossover
        separation_pct = abs((fast_ema - slow_ema) / slow_ema) * 100.0
        
        # Bullish crossover: fast crosses above slow
        if prev_fast_ema <= prev_slow_ema and fast_ema > slow_ema:
            if separation_pct >= self.min_separation_pct:
                confidence = min(0.85, 0.65 + (separation_pct / 2.0))
                return Signal(
                    symbol=symbol,
                    signal="BUY",
                    strategy=self.name,
                    confidence=confidence,
                    details={
                        "fast_ema": round(fast_ema, 6),
                        "slow_ema": round(slow_ema, 6),
                        "separation_pct": round(separation_pct, 4),
                        "fast_period": self.fast_period,
                        "slow_period": self.slow_period,
                    }
                )
        
        # Bearish crossover: fast crosses below slow
        elif prev_fast_ema >= prev_slow_ema and fast_ema < slow_ema:
            if separation_pct >= self.min_separation_pct:
                confidence = min(0.85, 0.65 + (separation_pct / 2.0))
                return Signal(
                    symbol=symbol,
                    signal="SELL",
                    strategy=self.name,
                    confidence=confidence,
                    details={
                        "fast_ema": round(fast_ema, 6),
                        "slow_ema": round(slow_ema, 6),
                        "separation_pct": round(separation_pct, 4),
                        "fast_period": self.fast_period,
                        "slow_period": self.slow_period,
                    }
                )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "min_separation_pct": self.min_separation_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "fast_period" in params:
            self.fast_period = int(params["fast_period"])
        if "slow_period" in params:
            self.slow_period = int(params["slow_period"])
        if "min_separation_pct" in params:
            self.min_separation_pct = float(params["min_separation_pct"])
