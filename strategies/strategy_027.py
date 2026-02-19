#!/usr/bin/env python3
"""
Strategy Name: volatility_adaptive_mean_reversion
Type: volatility/mean-reversion
Timeframe: 5m
Description: Mean-reversion strategy with volatility adaptation for ranging markets. Uses Bollinger Bands for entry signals, ATR for volatility-based confirmation, and RSI for overbought/oversold validation. Reduces position size during low volatility to preserve capital in choppy conditions.
"""

class Strategy:
    def __init__(self):
        self.name = "volatility_adaptive_mean_reversion"
        self.params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'atr_period': 14,
            'atr_threshold': 0.5,
            'volatility_threshold': 1.0,
            'min_atr_pct': 0.15,
            'max_atr_pct': 2.0,
        }
    
    def analyze(self, candles, indicators):
        """
        Volatility-adaptive mean-reversion strategy for ranging markets.
        
        Args:
            candles: List of recent candles [{open, high, low, close, volume}, ...]
            indicators: Dict of pre-calculated indicators {rsi, macd, bbands, atr, etc.}
        
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if len(candles) < self.params['bb_period']:
            return 'HOLD'
        
        current_close = candles[-1]['close']
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', 0)
        bbands = indicators.get('bbands', {})
        
        bb_upper = bbands.get('upper', 0)
        bb_middle = bbands.get('middle', 0)
        bb_lower = bbands.get('lower', 0)
        
        if bb_upper == 0 or bb_lower == 0 or atr == 0:
            return 'HOLD'
        
        bb_range = bb_upper - bb_lower
        price_position = (current_close - bb_lower) / bb_range if bb_range > 0 else 0.5
        atr_pct = (atr / current_close) * 100 if current_close > 0 else 0
        
        volatility_score = atr_pct / self.params['max_atr_pct']
        volatility_multiplier = max(0.5, min(volatility_score, 1.0))
        
        effective_atr_threshold = self.params['atr_threshold'] * volatility_multiplier
        
        buy_signal = (
            current_close <= bb_lower + (bb_range * effective_atr_threshold) and
            rsi < self.params['rsi_oversold'] and
            atr_pct >= self.params['min_atr_pct']
        )
        
        sell_signal = (
            current_close >= bb_upper - (bb_range * effective_atr_threshold) and
            rsi > self.params['rsi_overbought'] and
            atr_pct >= self.params['min_atr_pct']
        )
        
        if buy_signal:
            return 'BUY'
        elif sell_signal:
            return 'SELL'
        
        return 'HOLD'
    
    def get_confidence(self, candles, indicators):
        """Return confidence score 0-1 for current signal."""
        if len(candles) < self.params['bb_period']:
            return 0.0
        
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', 0)
        bbands = indicators.get('bbands', {})
        
        current_close = candles[-1]['close']
        bb_upper = bbands.get('upper', 0)
        bb_lower = bbands.get('lower', 0)
        
        if bb_upper == 0 or bb_lower == 0 or current_close == 0:
            return 0.0
        
        bb_range = bb_upper - bb_lower
        price_position = abs((current_close - (bb_upper + bb_lower) / 2) / (bb_range / 2))
        rsi_extremeness = max(0, abs(rsi - 50) - 15) / 35.0
        atr_pct = (atr / current_close) * 100
        
        volatility_score = min(atr_pct / self.params['max_atr_pct'], 1.0)
        
        confidence = (price_position * 0.4 + rsi_extremeness * 0.3 + volatility_score * 0.3)
        
        return min(confidence, 1.0)