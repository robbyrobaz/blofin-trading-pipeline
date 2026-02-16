#!/usr/bin/env python3
import os
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class CandlePatternsStrategy(BaseStrategy):
    """Detects candlestick patterns: engulfing, hammer, shooting star, doji."""
    
    name = "candle_patterns"
    version = "1.0"
    description = "Detects candlestick patterns from price action"
    
    def __init__(self):
        self.candle_period_seconds = int(os.getenv("CANDLE_PERIOD_SECONDS", "60"))  # 1-minute candles
        self.doji_body_pct = float(os.getenv("CANDLE_DOJI_BODY_PCT", "0.1"))
        self.hammer_wick_ratio = float(os.getenv("CANDLE_HAMMER_WICK_RATIO", "2.0"))
    
    def _build_candle(self, window: List[Tuple[int, float]]) -> Optional[Dict[str, float]]:
        """Build OHLC candle from price window."""
        if not window:
            return None
        
        prices = [p for _, p in window]
        return {
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
        }
    
    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]]
    ) -> Optional[Signal]:
        # Build 2-3 recent candles for pattern detection
        candle_ms = self.candle_period_seconds * 1000
        
        # Current candle window
        current_start = (ts_ms // candle_ms) * candle_ms
        current_window = [(t, p) for t, p in prices if current_start <= t <= ts_ms]
        
        # Previous candle window
        prev_start = current_start - candle_ms
        prev_window = [(t, p) for t, p in prices if prev_start <= t < current_start]
        
        if len(current_window) < 3 or len(prev_window) < 3:
            return None
        
        current_candle = self._build_candle(current_window)
        prev_candle = self._build_candle(prev_window)
        
        if current_candle is None or prev_candle is None:
            return None
        
        # Bullish Engulfing: current green candle engulfs previous red candle
        if (prev_candle["close"] < prev_candle["open"] and
            current_candle["close"] > current_candle["open"] and
            current_candle["open"] <= prev_candle["close"] and
            current_candle["close"] >= prev_candle["open"]):
            
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=0.75,
                details={
                    "pattern": "bullish_engulfing",
                    "prev_candle": prev_candle,
                    "current_candle": current_candle,
                }
            )
        
        # Bearish Engulfing: current red candle engulfs previous green candle
        if (prev_candle["close"] > prev_candle["open"] and
            current_candle["close"] < current_candle["open"] and
            current_candle["open"] >= prev_candle["close"] and
            current_candle["close"] <= prev_candle["open"]):
            
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=0.75,
                details={
                    "pattern": "bearish_engulfing",
                    "prev_candle": prev_candle,
                    "current_candle": current_candle,
                }
            )
        
        # Hammer: long lower wick, small body at top
        c = current_candle
        body = abs(c["close"] - c["open"])
        lower_wick = min(c["open"], c["close"]) - c["low"]
        upper_wick = c["high"] - max(c["open"], c["close"])
        candle_range = c["high"] - c["low"]
        
        if candle_range > 0:
            body_pct = (body / candle_range) * 100
            
            # Hammer pattern (bullish)
            if (lower_wick > body * self.hammer_wick_ratio and
                upper_wick < body and
                body_pct <= 33):
                
                return Signal(
                    symbol=symbol,
                    signal="BUY",
                    strategy=self.name,
                    confidence=0.70,
                    details={
                        "pattern": "hammer",
                        "candle": current_candle,
                        "lower_wick": round(lower_wick, 6),
                        "body": round(body, 6),
                    }
                )
            
            # Shooting Star pattern (bearish)
            if (upper_wick > body * self.hammer_wick_ratio and
                lower_wick < body and
                body_pct <= 33):
                
                return Signal(
                    symbol=symbol,
                    signal="SELL",
                    strategy=self.name,
                    confidence=0.70,
                    details={
                        "pattern": "shooting_star",
                        "candle": current_candle,
                        "upper_wick": round(upper_wick, 6),
                        "body": round(body, 6),
                    }
                )
            
            # Doji: very small body (indecision, potential reversal)
            if body_pct <= self.doji_body_pct:
                # Look at trend context to determine signal direction
                if len(prices) >= 10:
                    recent_trend = prices[-10][1] - prices[-1][1]
                    
                    if recent_trend > 0:  # Downtrend, doji signals potential reversal up
                        return Signal(
                            symbol=symbol,
                            signal="BUY",
                            strategy=self.name,
                            confidence=0.60,
                            details={
                                "pattern": "doji_reversal",
                                "candle": current_candle,
                                "body_pct": round(body_pct, 4),
                                "context": "downtrend",
                            }
                        )
                    elif recent_trend < 0:  # Uptrend, doji signals potential reversal down
                        return Signal(
                            symbol=symbol,
                            signal="SELL",
                            strategy=self.name,
                            confidence=0.60,
                            details={
                                "pattern": "doji_reversal",
                                "candle": current_candle,
                                "body_pct": round(body_pct, 4),
                                "context": "uptrend",
                            }
                        )
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "candle_period_seconds": self.candle_period_seconds,
            "doji_body_pct": self.doji_body_pct,
            "hammer_wick_ratio": self.hammer_wick_ratio,
        }
    
    def update_config(self, params: Dict[str, Any]) -> None:
        if "candle_period_seconds" in params:
            self.candle_period_seconds = int(params["candle_period_seconds"])
        if "doji_body_pct" in params:
            self.doji_body_pct = float(params["doji_body_pct"])
        if "hammer_wick_ratio" in params:
            self.hammer_wick_ratio = float(params["hammer_wick_ratio"])
