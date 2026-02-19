#!/usr/bin/env python3
"""
Strategy Name: Volatility Expansion Breakout
Type: volatility
Timeframe: 5m
Description: Detects Bollinger Band squeezes in low-volatility regimes and trades volatility expansion breakouts. Enters on ATR expansion with confirmation from price movement outside bands.
"""

class Strategy:
    def __init__(self):
        self.name = "volatility_expansion_breakout"
        self.params = {
            "bb_period": 20,
            "bb_std_dev": 1.5,
            "atr_period": 14,
            "atr_expansion_threshold": 1.3,
            "squeeze_lookback": 5,
            "min_squeeze_bars": 3,
            "volume_filter": 0.8,
            "profit_target_atr_mult": 2.0,
            "stop_loss_atr_mult": 1.0,
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
        if len(candles) < self.params["bb_period"] + self.params["squeeze_lookback"]:
            return 'HOLD'
        
        current = candles[-1]
        prev = candles[-2]
        
        bbands = indicators.get('bbands', {})
        atr = indicators.get('atr', 0)
        
        if not bbands or atr == 0:
            return 'HOLD'
        
        upper_band = bbands.get('upper', 0)
        lower_band = bbands.get('lower', 0)
        mid_band = bbands.get('mid', 0)
        bb_width = upper_band - lower_band
        
        if bb_width == 0:
            return 'HOLD'
        
        # Calculate historical band width to detect squeeze
        historical_widths = []
        for i in range(len(candles) - self.params["squeeze_lookback"], len(candles) - 1):
            c = candles[i]
            # Simplified: use close deviation from mid as proxy
            historical_widths.append(abs(c['close'] - mid_band))
        
        if not historical_widths:
            return 'HOLD'
        
        avg_historical_width = sum(historical_widths) / len(historical_widths)
        squeeze_ratio = bb_width / (avg_historical_width * 2) if avg_historical_width > 0 else 1.0
        
        is_in_squeeze = squeeze_ratio < 0.6
        
        # Calculate previous ATR for comparison
        prev_atr_estimate = indicators.get('atr_prev', atr)
        atr_expansion = (atr / prev_atr_estimate) if prev_atr_estimate > 0 else 1.0
        
        volume = current.get('volume', 1)
        prev_volume = prev.get('volume', 1)
        volume_ratio = volume / prev_volume if prev_volume > 0 else 1.0
        
        # BUY Signal: Squeeze detected, then ATR expands with price breakout above upper band
        if is_in_squeeze and atr_expansion > self.params["atr_expansion_threshold"]:
            if current['close'] > upper_band and volume_ratio > self.params["volume_filter"]:
                return 'BUY'
        
        # SELL Signal: Squeeze detected, then ATR expands with price breakout below lower band
        if is_in_squeeze and atr_expansion > self.params["atr_expansion_threshold"]:
            if current['close'] < lower_band and volume_ratio > self.params["volume_filter"]:
                return 'SELL'
        
        return 'HOLD'
    
    def get_confidence(self, candles, indicators):
        """Return confidence score 0-1 for current signal."""
        if len(candles) < 2:
            return 0.0
        
        bbands = indicators.get('bbands', {})
        atr = indicators.get('atr', 0)
        
        if not bbands or atr == 0:
            return 0.0
        
        current = candles[-1]
        upper_band = bbands.get('upper', 0)
        lower_band = bbands.get('lower', 0)
        bb_width = upper_band - lower_band
        
        if bb_width == 0:
            return 0.0
        
        # Confidence based on:
        # 1. How far price is from bands (stronger breakout = higher confidence)
        # 2. ATR expansion magnitude
        # 3. Volume confirmation
        
        price_distance = max(
            abs(current['close'] - upper_band),
            abs(current['close'] - lower_band)
        )
        band_confidence = min(price_distance / (bb_width * 0.5), 1.0)
        
        atr_expansion = indicators.get('atr_ratio', 1.0)
        atr_confidence = min((atr_expansion - 1.0) / 0.5, 1.0)
        
        volume_ratio = current.get('volume', 1) / candles[-2].get('volume', 1) if len(candles) > 1 else 1.0
        volume_confidence = min(volume_ratio, 1.0)
        
        combined_confidence = (band_confidence * 0.4 + atr_confidence * 0.35 + volume_confidence * 0.25)
        
        return max(0.0, min(combined_confidence, 1.0))