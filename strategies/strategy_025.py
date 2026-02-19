#!/usr/bin/env python3
"""
Strategy Name: Vol-Volume Mean Reversion
Type: volatility/volume-based mean-reversion hybrid
Timeframe: 5m
Description: Detects volatility expansion with volume surge in ranging markets. 
Uses mean-reversion to price bands anchored by volume-weighted moving averages. 
Avoids false breakouts by requiring volatility confirmation and volume anomalies.
"""

class Strategy:
    def __init__(self):
        self.name = "vol_volume_reversion"
        self.params = {
            "vwma_period": 20,
            "vol_ma_period": 14,
            "vol_expansion_threshold": 1.3,
            "volume_surge_threshold": 1.5,
            "band_width": 2.0,
            "rsi_period": 14,
            "min_vol_spike_bars": 2,
        }
    
    def analyze(self, candles, indicators):
        """
        Analyze market data and generate signal.
        
        Args:
            candles: List of recent candles [{open, high, low, close, volume}, ...]
            indicators: Dict of pre-calculated indicators {rsi, macd, bbands, etc.}
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if len(candles) < self.params["vwma_period"] + 5:
            return 'HOLD'
        
        current = candles[-1]
        prev = candles[-2]
        
        # Calculate volume-weighted moving average
        vwma = self._calculate_vwma(candles)
        
        # Calculate volatility expansion
        recent_volatility = self._calculate_volatility(candles[-self.params["vol_ma_period"]:])
        older_volatility = self._calculate_volatility(candles[-2*self.params["vol_ma_period"]:-self.params["vol_ma_period"]])
        vol_ratio = recent_volatility / (older_volatility + 1e-6)
        
        # Calculate volume analysis
        avg_volume = sum([c['volume'] for c in candles[-20:]]) / 20
        current_volume_ratio = current['volume'] / (avg_volume + 1e-6)
        
        # Count recent volume spikes
        volume_spike_count = sum(1 for c in candles[-self.params["min_vol_spike_bars"]:] 
                                  if c['volume'] > avg_volume * self.params["volume_surge_threshold"])
        
        # Price position relative to VWMA
        price_above_vwma = current['close'] > vwma
        price_distance_ratio = abs(current['close'] - vwma) / vwma
        
        # RSI for overbought/oversold confirmation
        rsi = indicators.get('rsi', 50)
        
        # Signal generation logic
        # BUY: Price below VWMA + volatility expanding + volume surge + RSI oversold
        if (not price_above_vwma and 
            vol_ratio > self.params["vol_expansion_threshold"] and
            volume_spike_count >= self.params["min_vol_spike_bars"] and
            rsi < 35 and
            price_distance_ratio < 0.05):
            return 'BUY'
        
        # SELL: Price above VWMA + volatility expanding + volume surge + RSI overbought
        if (price_above_vwma and 
            vol_ratio > self.params["vol_expansion_threshold"] and
            volume_spike_count >= self.params["min_vol_spike_bars"] and
            rsi > 65 and
            price_distance_ratio < 0.05):
            return 'SELL'
        
        return 'HOLD'
    
    def get_confidence(self, candles, indicators):
        """Return confidence score 0-1 for current signal."""
        if len(candles) < self.params["vwma_period"] + 5:
            return 0.0
        
        current = candles[-1]
        vwma = self._calculate_vwma(candles)
        
        recent_volatility = self._calculate_volatility(candles[-self.params["vol_ma_period"]:])
        older_volatility = self._calculate_volatility(candles[-2*self.params["vol_ma_period"]:-self.params["vol_ma_period"]])
        vol_ratio = recent_volatility / (older_volatility + 1e-6)
        
        avg_volume = sum([c['volume'] for c in candles[-20:]]) / 20
        current_volume_ratio = current['volume'] / (avg_volume + 1e-6)
        
        volume_spike_count = sum(1 for c in candles[-self.params["min_vol_spike_bars"]:] 
                                  if c['volume'] > avg_volume * self.params["volume_surge_threshold"])
        
        price_distance_ratio = abs(current['close'] - vwma) / vwma
        rsi = indicators.get('rsi', 50)
        
        # Confidence based on alignment of multiple factors
        confidence = 0.0
        
        # Volatility expansion contribution (0-0.25)
        vol_confidence = min((vol_ratio - 1.0) / (self.params["vol_expansion_threshold"] - 1.0), 1.0) * 0.25
        confidence += max(0, vol_confidence)
        
        # Volume surge contribution (0-0.25)
        vol_surge_confidence = (current_volume_ratio / self.params["volume_surge_threshold"]) * 0.25
        confidence += min(vol_surge_confidence, 0.25)
        
        # RSI extremeness contribution (0-0.25)
        rsi_confidence = max(
            abs(rsi - 50) / 50,
            0
        ) * 0.25
        confidence += min(rsi_confidence, 0.25)
        
        # Price proximity to VWMA contribution (0-0.25)
        proximity_confidence = max(1.0 - (price_distance_ratio * 20), 0) * 0.25
        confidence += min(proximity_confidence, 0.25)
        
        return min(confidence, 1.0)
    
    def _calculate_vwma(self, candles):
        """Calculate Volume Weighted Moving Average."""
        period = self.params["vwma_period"]
        if len(candles) < period:
            return candles[-1]['close']
        
        recent = candles[-period:]
        numerator = sum(c['close'] * c['volume'] for c in recent)
        denominator = sum(c['volume'] for c in recent)
        return numerator / (denominator + 1e-6)
    
    def _calculate_volatility(self, candles):
        """Calculate standard deviation of returns."""
        if len(candles) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(candles)):
            ret = (candles[i]['close'] - candles[i-1]['close']) / (candles[i-1]['close'] + 1e-6)
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return variance ** 0.5