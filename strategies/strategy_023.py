#!/usr/bin/env python3
"""
Strategy Name: Volatility Expansion Volume Breakout
Type: volatility/breakout
Timeframe: 5m
Description: Detects ultra-low volatility regimes and confirms breakouts with volume surge. Works in ranging markets transitioning to directional moves.
"""

class Strategy:
    def __init__(self):
        self.name = "volatility_expansion_volume_breakout"
        self.params = {
            "volatility_window": 20,
            "volatility_percentile": 15,
            "volume_multiplier": 1.8,
            "atr_period": 14,
            "breakout_threshold": 0.015,
            "min_volume_bars": 3,
        }
    
    def analyze(self, candles, indicators):
        """
        Analyze market data and generate signal.
        
        Args:
            candles: List of recent candles [{open, high, low, close, volume}, ...]
            indicators: Dict of pre-calculated indicators {rsi, macd, bbands, atr, etc.}
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if len(candles) < self.params["volatility_window"] + 5:
            return 'HOLD'
        
        recent = candles[-1]
        prev = candles[-2]
        
        atr = indicators.get('atr', {}).get('value', 0)
        if atr == 0:
            return 'HOLD'
        
        # Calculate volatility (normalized ATR)
        current_volatility = atr / recent['close']
        
        # Historical volatility average
        volatility_history = []
        for i in range(self.params["volatility_window"]):
            idx = -(self.params["volatility_window"] - i)
            if idx >= -len(candles):
                c = candles[idx]
                high_low = c['high'] - c['low']
                volatility_history.append(high_low / c['close'])
        
        if not volatility_history:
            return 'HOLD'
        
        volatility_history.sort()
        vol_threshold = volatility_history[int(len(volatility_history) * self.params["volatility_percentile"] / 100)]
        
        # Ultra-low volatility detection
        is_ultra_low_vol = current_volatility < vol_threshold
        
        # Volume surge detection
        avg_volume = sum(c['volume'] for c in candles[-self.params["min_volume_bars"]:-1]) / (self.params["min_volume_bars"] - 1)
        volume_surge = recent['volume'] > avg_volume * self.params["volume_multiplier"]
        
        # Price breakout detection
        price_change = abs(recent['close'] - prev['close']) / prev['close']
        is_breakout = price_change > self.params["breakout_threshold"]
        
        # Signal generation
        if is_ultra_low_vol and volume_surge and is_breakout:
            if recent['close'] > prev['close']:
                return 'BUY'
            else:
                return 'SELL'
        
        return 'HOLD'
    
    def get_confidence(self, candles, indicators):
        """Return confidence score 0-1 for current signal."""
        if len(candles) < self.params["volatility_window"] + 5:
            return 0.0
        
        recent = candles[-1]
        prev = candles[-2]
        atr = indicators.get('atr', {}).get('value', 0)
        
        if atr == 0:
            return 0.0
        
        current_volatility = atr / recent['close']
        volatility_history = []
        for i in range(self.params["volatility_window"]):
            idx = -(self.params["volatility_window"] - i)
            if idx >= -len(candles):
                c = candles[idx]
                high_low = c['high'] - c['low']
                volatility_history.append(high_low / c['close'])
        
        if not volatility_history:
            return 0.0
        
        volatility_history.sort()
        vol_threshold = volatility_history[int(len(volatility_history) * self.params["volatility_percentile"] / 100)]
        
        vol_score = max(0, 1 - (current_volatility / vol_threshold))
        
        avg_volume = sum(c['volume'] for c in candles[-self.params["min_volume_bars"]:-1]) / (self.params["min_volume_bars"] - 1)
        vol_ratio = recent['volume'] / avg_volume if avg_volume > 0 else 1
        volume_score = min(1, (vol_ratio - 1) / (self.params["volume_multiplier"] - 1))
        
        price_change = abs(recent['close'] - prev['close']) / prev['close']
        breakout_score = min(1, price_change / self.params["breakout_threshold"])
        
        confidence = (vol_score * 0.3 + volume_score * 0.4 + breakout_score * 0.3)
        
        return max(0, min(1, confidence))