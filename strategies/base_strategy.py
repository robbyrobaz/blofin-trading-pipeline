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
    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        """
        Given a list of OHLCV candles and the trading symbol, return a signal or None.

        Args:
            context_candles: List of candle dicts with keys:
                ts_ms, open, high, low, close, volume
            symbol: Trading pair symbol, e.g. "BTC-USDT"

        Returns:
            dict with at least {'signal': 'BUY'|'SELL', 'confidence': float}
            or None if no signal detected.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return current configurable parameters."""
        return {}

    def update_config(self, params: Dict[str, Any]) -> None:
        """Update parameters (for AI tuning)."""
        pass

    # ── Legacy helper kept for backward-compat (not used by backtester) ──
    def _slice_window(
        self,
        data: List[Tuple[int, float]],
        ts_ms: int,
        lookback_seconds: int
    ) -> List[Tuple[int, float]]:
        """Helper to slice a time-series window (tick-era utility)."""
        cutoff = ts_ms - (lookback_seconds * 1000)
        return [(t, v) for (t, v) in data if t >= cutoff]
