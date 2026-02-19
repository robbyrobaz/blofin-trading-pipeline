#!/usr/bin/env python3
"""
Candlestick Pattern Recognition — candle-based interface.

Detects classic candlestick patterns from OHLCV data.
Handles both real OHLCV candles (with intra-bar range) and flat candles
(where high==low==open==close), using close-price sequences for the latter.

Patterns detected:
  Real candles: engulfing, hammer, shooting star, morning/evening star, doji, marubozu
  Close-based: price-action engulfing, momentum bursts, reversal signals
  Mixed:        trend context for doji/inside-bar detection
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy, Signal


class CandlePatternsStrategy(BaseStrategy):
    """Candlestick pattern recognition on OHLCV candle data."""

    name = "candle_patterns"
    version = "2.2"
    description = "Candlestick patterns: real OHLCV + close-based fallback for flat candles"

    def __init__(self):
        self.min_candles = 20
        self.doji_body_pct = 20.0       # body < 20% of range = doji
        self.hammer_wick_ratio = 1.2    # lower wick >= 1.2x body = hammer
        self.trend_period = 10          # bars for trend context
        self.marubozu_wick_pct = 10.0   # wick < 10% of body = marubozu
        # Close-based pattern thresholds (for flat/synthetic candles)
        self.engulf_mult = 1.1          # current move must be >= 1.1x prior move
        self.reversal_bars = 3          # consecutive bars for momentum burst

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _body(c: dict) -> float:
        return abs(c['close'] - c['open'])

    @staticmethod
    def _rng(c: dict) -> float:
        return c['high'] - c['low']

    @staticmethod
    def _is_bullish_candle(c: dict) -> bool:
        return c['close'] >= c['open']

    @staticmethod
    def _is_bearish_candle(c: dict) -> bool:
        return c['close'] <= c['open']

    @staticmethod
    def _upper_wick(c: dict) -> float:
        return c['high'] - max(c['open'], c['close'])

    @staticmethod
    def _lower_wick(c: dict) -> float:
        return min(c['open'], c['close']) - c['low']

    @staticmethod
    def _is_flat(c: dict) -> bool:
        return c['high'] == c['low']

    def _trend_direction(self, candles: List[dict]) -> int:
        """Returns +1 uptrend, -1 downtrend, 0 neutral based on close prices."""
        if len(candles) < 4:
            return 0
        closes = [c['close'] for c in candles]
        n = len(closes)
        first_avg = sum(closes[:n//2]) / (n//2)
        second_avg = sum(closes[n//2:]) / (n - n//2)
        if second_avg > first_avg * 1.0001:
            return 1
        if second_avg < first_avg * 0.9999:
            return -1
        return 0

    def _avg_move(self, candles: List[dict]) -> float:
        """Average absolute close-to-close move."""
        moves = [abs(candles[i]['close'] - candles[i-1]['close'])
                 for i in range(1, len(candles))]
        return sum(moves) / len(moves) if moves else 0.0

    # ------------------------------------------------------------------
    # Close-based pattern detection (works for flat candles)
    # ------------------------------------------------------------------

    def _detect_close_based(self, candles: List[dict], trend: int) -> Optional[dict]:
        """Detect patterns from close price sequences only."""
        closes = [c['close'] for c in candles]
        n = len(closes)
        if n < 5:
            return None

        # Compute close-to-close moves
        moves = [closes[i] - closes[i-1] for i in range(1, n)]
        m0 = moves[-1]   # most recent move
        m1 = moves[-2]   # prior move
        m2 = moves[-3] if len(moves) >= 3 else 0.0
        m3 = moves[-4] if len(moves) >= 4 else 0.0

        avg_move = sum(abs(m) for m in moves[-20:]) / min(20, len(moves))
        if avg_move <= 0:
            return None

        # --- Close-based Engulfing ---
        # Prior move was down, current move is up and larger
        if m1 < 0 and m0 > 0 and m0 >= abs(m1) * self.engulf_mult:
            return {
                'signal': 'BUY', 'strategy': self.name, 'confidence': 0.65,
                'pattern': 'close_engulfing_bull', 'trend': trend,
            }
        # Prior move was up, current move is down and larger
        if m1 > 0 and m0 < 0 and abs(m0) >= m1 * self.engulf_mult:
            return {
                'signal': 'SELL', 'strategy': self.name, 'confidence': 0.65,
                'pattern': 'close_engulfing_bear', 'trend': trend,
            }

        # --- Momentum Burst (3 consecutive moves in same direction, last > avg) ---
        if m2 < 0 and m1 < 0 and m0 > 0 and m0 > avg_move * 1.0:
            return {
                'signal': 'BUY', 'strategy': self.name, 'confidence': 0.62,
                'pattern': 'exhaustion_reversal_bull', 'trend': trend,
            }
        if m2 > 0 and m1 > 0 and m0 < 0 and abs(m0) > avg_move * 1.0:
            return {
                'signal': 'SELL', 'strategy': self.name, 'confidence': 0.62,
                'pattern': 'exhaustion_reversal_bear', 'trend': trend,
            }

        # --- Outside reversal (current move reverses after 2+ same-direction moves) ---
        if m3 < 0 and m2 < 0 and m1 < 0 and m0 > 0:
            return {
                'signal': 'BUY', 'strategy': self.name, 'confidence': 0.64,
                'pattern': 'three_bar_reversal_bull', 'trend': trend,
            }
        if m3 > 0 and m2 > 0 and m1 > 0 and m0 < 0:
            return {
                'signal': 'SELL', 'strategy': self.name, 'confidence': 0.64,
                'pattern': 'three_bar_reversal_bear', 'trend': trend,
            }

        # --- Doji-equivalent: tiny move after trend ---
        if avg_move > 0 and abs(m0) < avg_move * 0.2:
            # Tiny move after directional context
            if trend < 0:
                return {
                    'signal': 'BUY', 'strategy': self.name, 'confidence': 0.58,
                    'pattern': 'close_doji', 'context': 'downtrend',
                }
            if trend > 0:
                return {
                    'signal': 'SELL', 'strategy': self.name, 'confidence': 0.58,
                    'pattern': 'close_doji', 'context': 'uptrend',
                }

        return None

    # ------------------------------------------------------------------
    # OHLCV-based pattern detection (works for real candles with range)
    # ------------------------------------------------------------------

    def _detect_ohlcv_based(self, non_flat: List[dict], trend: int,
                             avg_body: float) -> Optional[dict]:
        """Detect traditional candlestick patterns when range exists."""
        if len(non_flat) < 3:
            return None

        c0 = non_flat[-1]
        c1 = non_flat[-2]
        c2 = non_flat[-3]

        body0 = self._body(c0)
        rng0 = self._rng(c0)
        body1 = self._body(c1)
        body2 = self._body(c2)
        upper0 = self._upper_wick(c0)
        lower0 = self._lower_wick(c0)

        # --- Bullish Engulfing ---
        if (self._is_bearish_candle(c1) and self._is_bullish_candle(c0)
                and c0['open'] <= c1['close']
                and c0['close'] >= c1['open']):
            return {
                'signal': 'BUY', 'strategy': self.name, 'confidence': 0.74,
                'pattern': 'bullish_engulfing', 'trend': trend,
            }

        # --- Bearish Engulfing ---
        if (self._is_bullish_candle(c1) and self._is_bearish_candle(c0)
                and c0['open'] >= c1['close']
                and c0['close'] <= c1['open']):
            return {
                'signal': 'SELL', 'strategy': self.name, 'confidence': 0.74,
                'pattern': 'bearish_engulfing', 'trend': trend,
            }

        # --- Hammer ---
        if rng0 > 0 and body0 > 0:
            body_pct = body0 / rng0 * 100
            if (lower0 >= body0 * self.hammer_wick_ratio
                    and upper0 <= lower0 * 0.5
                    and body_pct <= 50):
                return {
                    'signal': 'BUY', 'strategy': self.name, 'confidence': 0.67,
                    'pattern': 'hammer', 'lower_wick': round(lower0, 6), 'trend': trend,
                }

        # --- Shooting Star ---
        if rng0 > 0 and body0 > 0:
            body_pct = body0 / rng0 * 100
            if (upper0 >= body0 * self.hammer_wick_ratio
                    and lower0 <= upper0 * 0.5
                    and body_pct <= 50):
                return {
                    'signal': 'SELL', 'strategy': self.name, 'confidence': 0.67,
                    'pattern': 'shooting_star', 'upper_wick': round(upper0, 6), 'trend': trend,
                }

        # --- Marubozu ---
        if rng0 > 0 and body0 > 0 and avg_body > 0:
            upper_pct = upper0 / body0 * 100 if body0 > 0 else 100
            lower_pct = lower0 / body0 * 100 if body0 > 0 else 100
            if (body0 > avg_body * 1.1
                    and upper_pct < self.marubozu_wick_pct
                    and lower_pct < self.marubozu_wick_pct):
                if self._is_bullish_candle(c0):
                    return {'signal': 'BUY', 'strategy': self.name, 'confidence': 0.70,
                            'pattern': 'bullish_marubozu', 'trend': trend}
                else:
                    return {'signal': 'SELL', 'strategy': self.name, 'confidence': 0.70,
                            'pattern': 'bearish_marubozu', 'trend': trend}

        # --- Morning Star ---
        if (self._is_bearish_candle(c2) and body2 > 0
                and body1 < body2 * 0.6
                and self._is_bullish_candle(c0)
                and c0['close'] > c2['close'] + body2 * 0.3):
            return {'signal': 'BUY', 'strategy': self.name, 'confidence': 0.72,
                    'pattern': 'morning_star'}

        # --- Evening Star ---
        if (self._is_bullish_candle(c2) and body2 > 0
                and body1 < body2 * 0.6
                and self._is_bearish_candle(c0)
                and c0['close'] < c2['close'] - body2 * 0.3):
            return {'signal': 'SELL', 'strategy': self.name, 'confidence': 0.72,
                    'pattern': 'evening_star'}

        # --- Doji with trend ---
        if rng0 > 0:
            body_pct = body0 / rng0 * 100
            if body_pct <= self.doji_body_pct:
                if trend < 0:
                    return {'signal': 'BUY', 'strategy': self.name, 'confidence': 0.61,
                            'pattern': 'doji', 'context': 'downtrend'}
                if trend > 0:
                    return {'signal': 'SELL', 'strategy': self.name, 'confidence': 0.61,
                            'pattern': 'doji', 'context': 'uptrend'}

        return None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        trend = self._trend_direction(context_candles[-self.trend_period:])

        # Separate flat from real candles
        non_flat = [c for c in context_candles if c['high'] > c['low']]
        flat_ratio = 1.0 - len(non_flat) / len(context_candles)

        # If mostly flat data, use close-based detection
        if flat_ratio >= 0.5 or len(non_flat) < 5:
            return self._detect_close_based(context_candles, trend)

        # Real candle data: use OHLCV patterns first
        avg_body = sum(self._body(c) for c in non_flat[-20:]) / min(20, len(non_flat))
        result = self._detect_ohlcv_based(non_flat, trend, avg_body)
        if result:
            return result

        # Fall back to close-based if OHLCV found nothing
        return self._detect_close_based(context_candles, trend)

    def get_config(self) -> Dict[str, Any]:
        return {
            'doji_body_pct': self.doji_body_pct,
            'hammer_wick_ratio': self.hammer_wick_ratio,
            'trend_period': self.trend_period,
            'marubozu_wick_pct': self.marubozu_wick_pct,
            'engulf_mult': self.engulf_mult,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        if 'doji_body_pct' in params:
            self.doji_body_pct = float(params['doji_body_pct'])
        if 'hammer_wick_ratio' in params:
            self.hammer_wick_ratio = float(params['hammer_wick_ratio'])
        if 'trend_period' in params:
            self.trend_period = int(params['trend_period'])
        if 'marubozu_wick_pct' in params:
            self.marubozu_wick_pct = float(params['marubozu_wick_pct'])
        if 'engulf_mult' in params:
            self.engulf_mult = float(params['engulf_mult'])
