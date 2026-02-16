#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class Signal:
    symbol: str
    signal: str  # "BUY" or "SELL"
    strategy: str
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    name: str = "base"
    version: str = "1.0"
    description: str = "Base strategy class"
    
    @abstractmethod
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        """
        Given current tick + history windows, return a signal or None.
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            volume: Current volume
            ts_ms: Current timestamp in milliseconds
            prices: List of (timestamp_ms, price) tuples for historical data
            volumes: List of (timestamp_ms, volume) tuples for historical data
            
        Returns:
            Signal object if a signal is detected, None otherwise
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configurable parameters."""
        return {}
    
    def update_config(self, params: Dict[str, Any]) -> None:
        """Update parameters (for AI tuning)."""
        pass
    
    def _slice_window(
        self,
        data: List[Tuple[int, float]],
        ts_ms: int,
        lookback_seconds: int
    ) -> List[Tuple[int, float]]:
        """Helper to slice a time-series window."""
        cutoff = ts_ms - (lookback_seconds * 1000)
        return [(t, v) for (t, v) in data if t >= cutoff]
