#!/usr/bin/env python3
"""
Strategy 025: Vol-Volume Mean Reversion — candle-based interface.
Type: volatility/volume-based mean-reversion hybrid, Timeframe: 5m

Mean-reversion to VWMA with volatility expansion and volume surge confirmation.
Avoids false breakouts by requiring volatility confirmation and volume anomalies.
"""
from typing import Optional, Dict, Any, List

from .base_strategy import BaseStrategy


class VolVolumeReversionStrategy(BaseStrategy):
    """VWMA mean-reversion with vol expansion and volume surge confirmation."""

    name = "vol_volume_reversion"
    version = "2.0"
    description = "VWMA mean-reversion with volatility expansion + volume surge"

    def __init__(self):
        self.vwma_period             = 20
        self.vol_ma_period           = 14
        # Lowered from 1.3 → 1.1: vol expansion >1.3 only 4x per 2016 candles
        self.vol_expansion_threshold = 1.1
        # Lowered from 1.5 → 1.2: tick-based volume spikes are moderate
        self.volume_surge_threshold  = 1.2
        self.rsi_period              = 14
        # Lowered from 2 → 1: requiring 2 consecutive spikes was too restrictive
        self.min_vol_spike_bars      = 1
        self.min_candles             = self.vwma_period + 5

    def _calc_vwma(self, candles: list) -> float:
        period = min(self.vwma_period, len(candles))
        recent = candles[-period:]
        denom = sum(c['volume'] for c in recent)
        if denom == 0:
            return candles[-1]['close']
        return sum(c['close'] * c['volume'] for c in recent) / denom

    def _calc_volatility(self, candles: list) -> float:
        """Standard deviation of close returns."""
        if len(candles) < 2:
            return 0.0
        returns = []
        for i in range(1, len(candles)):
            prev = candles[i-1]['close']
            if prev > 0:
                returns.append((candles[i]['close'] - prev) / prev)
        if not returns:
            return 0.0
        mean_r = sum(returns) / len(returns)
        return (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5

    def _calc_rsi(self, closes: List[float], period: int) -> float:
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0.0))
            losses.append(max(-d, 0.0))
        ag = sum(gains[:period]) / period
        al = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0 if ag > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    def detect(self, context_candles: list, symbol: str) -> Optional[dict]:
        if len(context_candles) < self.min_candles:
            return None

        current = context_candles[-1]

        vwma = self._calc_vwma(context_candles)
        price = current['close']
        price_above_vwma = price > vwma
        if vwma <= 0:
            return None
        price_distance_ratio = abs(price - vwma) / vwma

        # Volatility expansion: recent vs older window
        recent_candles = context_candles[-self.vol_ma_period:]
        older_candles  = context_candles[-2 * self.vol_ma_period:-self.vol_ma_period]
        recent_vol = self._calc_volatility(recent_candles)
        older_vol  = self._calc_volatility(older_candles)
        vol_ratio = recent_vol / (older_vol + 1e-6)

        # Volume analysis
        avg_volume = sum(c['volume'] for c in context_candles[-20:]) / 20
        volume_spike_count = sum(
            1 for c in context_candles[-self.min_vol_spike_bars:]
            if avg_volume > 0 and c['volume'] > avg_volume * self.volume_surge_threshold
        )

        closes = [c['close'] for c in context_candles]
        rsi = self._calc_rsi(closes, self.rsi_period)

        # BUY: price below VWMA + vol expanding + volume surging + RSI oversold
        # RSI threshold loosened from 35 → 42 to match data distribution
        if (not price_above_vwma and
                vol_ratio > self.vol_expansion_threshold and
                volume_spike_count >= self.min_vol_spike_bars and
                rsi < 42 and
                price_distance_ratio < 0.05):
            # Confidence scoring
            c = 0.0
            c += min((vol_ratio - 1.0) / max(self.vol_expansion_threshold - 1.0, 0.01), 1.0) * 0.25
            c += min(current['volume'] / (avg_volume * self.volume_surge_threshold), 1.0) * 0.25
            c += max(abs(rsi - 50) / 50, 0) * 0.25
            c += max(1.0 - price_distance_ratio * 20, 0) * 0.25
            return {
                'signal': 'BUY',
                'strategy': self.name,
                'confidence': round(max(0.50, min(c, 1.0)), 4),
                'rsi': round(rsi, 2),
                'vol_ratio': round(vol_ratio, 3),
                'price_vs_vwma': round((price - vwma) / vwma * 100, 4),
            }

        # SELL: price above VWMA + vol expanding + volume surging + RSI overbought
        # RSI threshold loosened from 65 → 58 to match data distribution
        if (price_above_vwma and
                vol_ratio > self.vol_expansion_threshold and
                volume_spike_count >= self.min_vol_spike_bars and
                rsi > 58 and
                price_distance_ratio < 0.05):
            c = 0.0
            c += min((vol_ratio - 1.0) / max(self.vol_expansion_threshold - 1.0, 0.01), 1.0) * 0.25
            c += min(current['volume'] / (avg_volume * self.volume_surge_threshold), 1.0) * 0.25
            c += max(abs(rsi - 50) / 50, 0) * 0.25
            c += max(1.0 - price_distance_ratio * 20, 0) * 0.25
            return {
                'signal': 'SELL',
                'strategy': self.name,
                'confidence': round(max(0.50, min(c, 1.0)), 4),
                'rsi': round(rsi, 2),
                'vol_ratio': round(vol_ratio, 3),
                'price_vs_vwma': round((price - vwma) / vwma * 100, 4),
            }

        return None

    def get_config(self) -> Dict[str, Any]:
        return {
            'vwma_period': self.vwma_period,
            'vol_ma_period': self.vol_ma_period,
            'vol_expansion_threshold': self.vol_expansion_threshold,
            'volume_surge_threshold': self.volume_surge_threshold,
        }

    def update_config(self, params: Dict[str, Any]) -> None:
        for key in ('vwma_period', 'vol_ma_period', 'min_vol_spike_bars', 'rsi_period'):
            if key in params:
                setattr(self, key, int(params[key]))
        for key in ('vol_expansion_threshold', 'volume_surge_threshold'):
            if key in params:
                setattr(self, key, float(params[key]))
