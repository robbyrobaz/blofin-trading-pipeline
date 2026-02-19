#!/usr/bin/env python3
"""
Strategy Name: Volume Volatility Mean Reversion
Type: volatility/volume + mean-reversion hybrid
Timeframe: 5m
Description: Exploits volatility expansion with volume confirmation in ranging markets. 
Trades mean reversion pullbacks when volatility increases and volume supports the reversal.
Avoids pure Bollinger Band squeeze patterns by using ATR and volume as primary filters.
"""

class Strategy:
    def __init__(self):
        self.name = "volume_volatility_mean_reversion"
        self.params = {
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "volume_ma_period": 20,
            "volume_threshold": 1.2,
            "rsi_period": 14,
            "rsi_buy_threshold": 35,
            "rsi_sell_threshold": 65,
            "min_candles": 30,
            "volatility_expansion_threshold": 1.3
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
        if len(candles) < self.params["min_candles"]:
            return 'HOLD'
        
        current_candle = candles[-1]
        prev_candle = candles[-2]
        
        # Get current indicators
        current_atr = indicators.get('atr', [0])[-1] if indicators.get('atr') else 0
        atr_prev = indicators.get('atr', [0])[-2] if len(indicators.get('atr', [])) > 1 else 0
        
        current_rsi = indicators.get('rsi', [50])[-1] if indicators.get('rsi') else 50
        current_volume = current_candle.get('volume', 0)
        
        # Calculate volume MA
        volumes = [c.get('volume', 0) for c in candles[-self.params["volume_ma_period"]:]]
        volume_ma = sum(volumes) / len(volumes) if volumes else 0
        
        # Volatility expansion condition
        volatility_expanding = current_atr > atr_prev * self.params["volatility_expansion_threshold"] if atr_prev > 0 else False
        
        # Volume confirmation
        volume_confirmed = current_volume > volume_ma * self.params["volume_threshold"]
        
        # BUY Signal: Volatility expanding + Volume surge + RSI oversold + Price pullback
        if (volatility_expanding and 
            volume_confirmed and 
            current_rsi < self.params["rsi_buy_threshold"] and
            current_candle.get('close', 0) < prev_candle.get('close', 0)):
            return 'BUY'
        
        # SELL Signal: Volatility expanding + Volume surge + RSI overbought + Price bounce
        if (volatility_expanding and 
            volume_confirmed and 
            current_rsi > self.params["rsi_sell_threshold"] and
            current_candle.get('close', 0) > prev_candle.get('close', 0)):
            return 'SELL'
        
        return 'HOLD'
    
    def get_confidence(self, candles, indicators):
        """Return confidence score 0-1 for current signal."""
        if len(candles) < self.params["min_candles"]:
            return 0.0
        
        current_candle = candles[-1]
        current_atr = indicators.get('atr', [0])[-1] if indicators.get('atr') else 0
        atr_prev = indicators.get('atr', [0])[-2] if len(indicators.get('atr', [])) > 1 else 0
        current_rsi = indicators.get('rsi', [50])[-1] if indicators.get('rsi') else 50
        current_volume = current_candle.get('volume', 0)
        
        # Calculate volume MA
        volumes = [c.get('volume', 0) for c in candles[-self.params["volume_ma_period"]:]]
        volume_ma = sum(volumes) / len(volumes) if volumes else 0
        
        confidence = 0.5
        
        # Boost confidence on volatility expansion
        if current_atr > atr_prev * self.params["volatility_expansion_threshold"] and atr_prev > 0:
            confidence += 0.15
        
        # Boost confidence on volume spike
        if current_volume > volume_ma * self.params["volume_threshold"]:
            confidence += 0.15
        
        # RSI extremes add confidence
        if current_rsi < 30 or current_rsi > 70:
            confidence += 0.1
        
        return min(confidence, 1.0)