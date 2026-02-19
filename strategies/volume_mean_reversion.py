#!/usr/bin/env python3
"""
Volume-Weighted Mean Reversion for Low-Volatility Ranging Markets

Design rationale:
- Learns from vwap_reversion (top performer): uses VWAP as anchor + mean reversion logic
- Avoids bb_squeeze failure patterns: does NOT trade breakouts; instead requires low volatility
  as a *confirmation* for mean reversion (tight range = price reverts to mean)
- Fills portfolio gap: volume-weighted regime filter ensures we only trade when volume confirms
  the ranging condition (avoids thin-volume fakeouts that killed bb_squeeze)
- Adapts to ranging_low_volatility regime: explicitly detects and targets this regime
"""
import os
import math
from typing import Optional, Dict, Any, List, Tuple

from .base_strategy import BaseStrategy, Signal


class VolumeMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy optimized for low-volatility ranging markets.

    Core idea: When volatility is low and volume is stable (ranging market),
    price oscillates around VWAP. Trade deviations from VWAP only when:
      1. Volatility is below threshold (ranging regime confirmed)
      2. Volume is not spiking (no breakout incoming)
      3. Price deviates enough from VWAP to offer edge
      4. RSI confirms oversold/overbought within the range

    Anti-patterns from bb_squeeze avoided:
      - No breakout trading (bb_squeeze failed on false breakouts)
      - Requires volume to be STABLE, not spiking
      - Uses adaptive thresholds based on recent volatility
    """

    name = "volume_mean_reversion"
    version = "1.0"
    description = "Volume-weighted mean reversion for low-volatility ranging markets"

    def __init__(self):
        # VWAP calculation window
        self.vwap_lookback_seconds = int(os.getenv("VMR_VWAP_LOOKBACK", "1800"))
        # Volatility measurement window
        self.vol_lookback_seconds = int(os.getenv("VMR_VOL_LOOKBACK", "900"))
        # Maximum volatility % to consider market "ranging" (band width / mean * 100)
        self.max_volatility_pct = float(os.getenv("VMR_MAX_VOL_PCT", "1.5"))
        # Minimum VWAP deviation % to trigger signal
        self.min_deviation_pct = float(os.getenv("VMR_MIN_DEVIATION", "0.4"))
        # Maximum volume ratio (current / average) — reject spikes
        self.max_volume_ratio = float(os.getenv("VMR_MAX_VOL_RATIO", "2.0"))
        # Minimum volume ratio — reject thin markets
        self.min_volume_ratio = float(os.getenv("VMR_MIN_VOL_RATIO", "0.3"))
        # RSI period (number of data points)
        self.rsi_period = int(os.getenv("VMR_RSI_PERIOD", "14"))
        # RSI oversold/overbought thresholds
        self.rsi_oversold = float(os.getenv("VMR_RSI_OVERSOLD", "35"))
        self.rsi_overbought = float(os.getenv("VMR_RSI_OVERBOUGHT", "65"))
        # Minimum data points required
        self.min_data_points = int(os.getenv("VMR_MIN_DATA", "30"))

    def detect(
        self,
        symbol: str,
        price: float,
        volume: float,
        ts_ms: int,
        prices: List[Tuple[int, float]],
        volumes: List[Tuple[int, float]],
    ) -> Optional[Signal]:
        # Slice windows
        price_window = self._slice_window(prices, ts_ms, self.vwap_lookback_seconds)
        vol_window = self._slice_window(volumes, ts_ms, self.vwap_lookback_seconds)
        short_prices = self._slice_window(prices, ts_ms, self.vol_lookback_seconds)

        if len(price_window) < self.min_data_points or len(vol_window) < self.min_data_points:
            return None

        # --- 1. Compute VWAP ---
        total_pv = 0.0
        total_v = 0.0
        min_len = min(len(price_window), len(vol_window))
        for i in range(min_len):
            p = price_window[i][1]
            v = vol_window[i][1]
            total_pv += p * v
            total_v += v

        if total_v <= 0:
            return None

        vwap = total_pv / total_v
        if vwap <= 0:
            return None

        deviation_pct = ((price - vwap) / vwap) * 100.0

        # --- 2. Check volatility is LOW (ranging regime) ---
        if len(short_prices) < 10:
            return None

        sp_values = [p for _, p in short_prices]
        sp_mean = sum(sp_values) / len(sp_values)
        if sp_mean <= 0:
            return None

        sp_var = sum((p - sp_mean) ** 2 for p in sp_values) / len(sp_values)
        sp_std = math.sqrt(sp_var) if sp_var > 0 else 0
        volatility_pct = (sp_std / sp_mean) * 100.0

        # REJECT if volatility is too high — not a ranging market
        if volatility_pct > self.max_volatility_pct:
            return None

        # --- 3. Check volume is STABLE (no spike = no breakout) ---
        vol_values = [v for _, v in vol_window]
        avg_volume = sum(vol_values) / len(vol_values) if vol_values else 0
        if avg_volume <= 0:
            return None

        volume_ratio = volume / avg_volume

        # Reject volume spikes (breakout risk — bb_squeeze lesson)
        if volume_ratio > self.max_volume_ratio:
            return None
        # Reject thin volume (unreliable price action)
        if volume_ratio < self.min_volume_ratio:
            return None

        # --- 4. Compute RSI for confirmation ---
        rsi = self._compute_rsi(price_window)

        # --- 5. Generate signal ---
        if abs(deviation_pct) < self.min_deviation_pct:
            return None

        # BUY: price below VWAP + RSI confirms oversold
        if deviation_pct <= -self.min_deviation_pct and (rsi is None or rsi < self.rsi_oversold):
            confidence = self._compute_confidence(
                abs(deviation_pct), volatility_pct, volume_ratio, rsi, is_buy=True
            )
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=confidence,
                details={
                    "vwap": round(vwap, 6),
                    "deviation_pct": round(deviation_pct, 4),
                    "volatility_pct": round(volatility_pct, 4),
                    "volume_ratio": round(volume_ratio, 4),
                    "rsi": round(rsi, 2) if rsi is not None else None,
                    "regime": "ranging_low_volatility",
                },
            )

        # SELL: price above VWAP + RSI confirms overbought
        if deviation_pct >= self.min_deviation_pct and (rsi is None or rsi > self.rsi_overbought):
            confidence = self._compute_confidence(
                abs(deviation_pct), volatility_pct, volume_ratio, rsi, is_buy=False
            )
            return Signal(
                symbol=symbol,
                signal="SELL",
                strategy=self.name,
                confidence=confidence,
                details={
                    "vwap": round(vwap, 6),
                    "deviation_pct": round(deviation_pct, 4),
                    "volatility_pct": round(volatility_pct, 4),
                    "volume_ratio": round(volume_ratio, 4),
                    "rsi": round(rsi, 2) if rsi is not None else None,
                    "regime": "ranging_low_volatility",
                },
            )

        return None

    def _compute_rsi(self, price_window: List[Tuple[int, float]]) -> Optional[float]:
        """Compute RSI from price window."""
        if len(price_window) < self.rsi_period + 1:
            return None

        values = [p for _, p in price_window[-(self.rsi_period + 1):]]
        gains = []
        losses = []
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            if delta > 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(delta))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_confidence(
        self,
        deviation: float,
        volatility_pct: float,
        volume_ratio: float,
        rsi: Optional[float],
        is_buy: bool,
    ) -> float:
        """
        Adaptive confidence based on multiple factors.

        Higher confidence when:
        - Deviation from VWAP is larger (stronger reversion expected)
        - Volatility is lower (more predictable range)
        - Volume is closer to average (stable market)
        - RSI confirms extreme
        """
        # Base confidence from deviation magnitude (0.55 - 0.75)
        dev_score = min(1.0, deviation / (self.min_deviation_pct * 3))
        base = 0.55 + (dev_score * 0.20)

        # Volatility bonus: lower vol = more confidence (0 to +0.08)
        vol_score = max(0.0, 1.0 - (volatility_pct / self.max_volatility_pct))
        base += vol_score * 0.08

        # Volume stability bonus: closer to 1.0 ratio = more confidence (0 to +0.05)
        vol_ratio_score = 1.0 - min(1.0, abs(volume_ratio - 1.0))
        base += vol_ratio_score * 0.05

        # RSI confirmation bonus (0 to +0.07)
        if rsi is not None:
            if is_buy:
                rsi_score = max(0.0, (self.rsi_oversold - rsi) / self.rsi_oversold)
            else:
                rsi_score = max(0.0, (rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought))
            base += rsi_score * 0.07

        return round(min(0.90, max(0.50, base)), 4)

    def get_config(self) -> Dict[str, Any]:
        return {
            "vwap_lookback_seconds": self.vwap_lookback_seconds,
            "vol_lookback_seconds": self.vol_lookback_seconds,
            "max_volatility_pct": self.max_volatility_pct,
            "min_deviation_pct": self.min_deviation_pct,
            "max_volume_ratio": self.max_volume_ratio,
            "min_volume_ratio": self.min_volume_ratio,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "min_data_points": self.min_data_points,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        config_map = {
            "vwap_lookback_seconds": ("vwap_lookback_seconds", int),
            "vol_lookback_seconds": ("vol_lookback_seconds", int),
            "max_volatility_pct": ("max_volatility_pct", float),
            "min_deviation_pct": ("min_deviation_pct", float),
            "max_volume_ratio": ("max_volume_ratio", float),
            "min_volume_ratio": ("min_volume_ratio", float),
            "rsi_period": ("rsi_period", int),
            "rsi_oversold": ("rsi_oversold", float),
            "rsi_overbought": ("rsi_overbought", float),
            "min_data_points": ("min_data_points", int),
        }
        for key, (attr, typ) in config_map.items():
            if key in params:
                setattr(self, attr, typ(params[key]))
