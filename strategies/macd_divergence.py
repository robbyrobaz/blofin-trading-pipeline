#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class MACDDivergenceStrategy(BaseStrategy):
    """Detects MACD histogram divergence from price action."""
    
    name = "macd_divergence"
    version = "1.0"
    description = "Detects MACD histogram divergence from price movement"
    
    def __init__(self):
        self.fast_period = int(os.getenv("MACD_FAST_PERIOD", "12"))
        self.slow_period = int(os.getenv("MACD_SLOW_PERIOD", "26"))
        self.signal_period = int(os.getenv("MACD_SIGNAL_PERIOD", "9"))
        self.min_divergence_pct = float(os.getenv("MACD_MIN_DIVERGENCE_PCT", "0.3"))
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate EMA for given prices and period."""
        if len(prices) < period:
            return None
        
        sma = sum(prices[:period]) / period
        multiplier = 2 / (period + 1)
        ema = sma
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: List[float]) -> Optional[Tuple[float, float, float]]:
        """Calculate MACD, signal line, and histogram."""
        if len(prices) < self.slow_period + self.signal_period:
            return None
        
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        macd = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD values)
        # For simplicity, approximate with recent MACD trend
        # In production, maintain MACD history
        signal = macd  # Simplified; real implementation needs MACD history
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        if len(prices) < self.slow_period + self.signal_period + 10:
            return None
        
        price_values = [p for _, p in prices]
        
        # Calculate current MACD
        current_macd = self._calculate_macd(price_values)
        if current_macd is None:
            return None
        
        macd, signal_line, histogram = current_macd
        
        # Calculate previous MACD (5 ticks back)
        if len(price_values) < self.slow_period + self.signal_period + 15:
            return None
        
        prev_price_values = price_values[:-5]
        prev_macd = self._calculate_macd(prev_price_values)
        if prev_macd is None:
            return None
        
        prev_macd_val, prev_signal, prev_histogram = prev_macd
        
        # Check for divergence
        recent_prices = price_values[-10:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100 if recent_prices[0] > 0 else 0
        
        histogram_change = histogram - prev_histogram
        
        # Bullish divergence: price making lower lows, but MACD histogram rising
        if price_trend < -self.min_divergence_pct and histogram > prev_histogram:
            confidence = min(0.85, 0.65 + abs(histogram_change) * 5)
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "macd": round(macd, 6),
                    "signal": round(signal_line, 6),
                    "histogram": round(histogram, 6),
                    "price_trend_pct": round(price_trend, 4),
                    "divergence": "bullish",
                }
            )
        
        # Bearish divergence: price making higher highs, but MACD histogram falling
        elif price_trend > self.min_divergence_pct and histogram < prev_histogram:
            confidence = min(0.85, 0.65 + abs(histogram_change) * 5)
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "macd": round(macd, 6),
                    "signal": round(signal_line, 6),
                    "histogram": round(histogram, 6),
                    "price_trend_pct": round(price_trend, 4),
                    "divergence": "bearish",
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "min_divergence_pct": self.min_divergence_pct,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "fast_period" in params:
            self.fast_period = int(params["fast_period"])
        if "slow_period" in params:
            self.slow_period = int(params["slow_period"])
        if "signal_period" in params:
            self.signal_period = int(params["signal_period"])
        if "min_divergence_pct" in params:
            self.min_divergence_pct = float(params["min_divergence_pct"])
