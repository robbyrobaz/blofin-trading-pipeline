#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class RSIDivergenceStrategy(BaseStrategy):
    """Detects RSI overbought/oversold conditions."""
    
    name = "rsi_divergence"
    version = "1.0"
    description = "Detects RSI overbought and oversold conditions"
    
    def __init__(self):
        self.window_seconds = int(os.getenv("RSI_WINDOW_SECONDS", "1200"))
        self.oversold = float(os.getenv("RSI_OVERSOLD", "20"))
        self.overbought = float(os.getenv("RSI_OVERBOUGHT", "80"))
    
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
        
        if len(window) < 14:
            return None
        
        # Calculate RSI
        price_values = [p for _, p in window]
        gains = []
        losses = []
        
        for i in range(1, len(price_values)):
            change = price_values[i] - price_values[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        elif avg_gain > 0:
            rsi = 100
        else:
            rsi = 50
        
        # Buy when oversold
        if rsi <= self.oversold:
            confidence = min(0.80, max(0.55, (self.oversold - rsi) / self.oversold))
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "window_s": self.window_seconds,
                    "rsi": round(rsi, 2),
                    "threshold": self.oversold,
                }
            )
        # Sell when overbought
        elif rsi >= self.overbought:
            confidence = min(0.80, max(0.55, (rsi - self.overbought) / (100 - self.overbought)))
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "window_s": self.window_seconds,
                    "rsi": round(rsi, 2),
                    "threshold": self.overbought,
                }
            )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "window_seconds" in params:
            self.window_seconds = int(params["window_seconds"])
        if "oversold" in params:
            self.oversold = float(params["oversold"])
        if "overbought" in params:
            self.overbought = float(params["overbought"])
