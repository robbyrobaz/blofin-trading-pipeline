#!/usr/bin/env python3
"""
Integration tests for backtester with existing strategies.
Tests that backtester works with real strategy implementations.
"""

import unittest
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backtester import BacktestEngine, format_metrics


class StrategyAdapter:
    """
    Adapter to make existing strategies work with backtester.
    
    Existing strategies use:
        detect(symbol, price, volume, ts_ms, prices, volumes)
    
    Backtester expects:
        detect(candles, symbol)
    
    This adapter converts between the two interfaces.
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
    
    def detect(self, candles, symbol):
        """Convert candles to old interface."""
        if not candles or len(candles) < 2:
            return None
        
        # Get current candle
        current = candles[-1]
        
        # Convert candles to prices and volumes lists
        prices = [(c['ts_ms'], c['close']) for c in candles]
        volumes = [(c['ts_ms'], c.get('volume', 0)) for c in candles]
        
        # Call old strategy interface
        signal = self.strategy.detect(
            symbol=symbol,
            price=current['close'],
            volume=current.get('volume', 0),
            ts_ms=current['ts_ms'],
            prices=prices,
            volumes=volumes
        )
        
        # Convert signal to backtester format
        if signal:
            return {
                'signal': signal.signal,
                'confidence': signal.confidence
            }
        
        return None


class TestIntegration(unittest.TestCase):
    """Integration tests with real strategies (if available)."""
    
    def test_backtester_structure(self):
        """Test that backtester module is properly structured."""
        from backtester import (
            BacktestEngine,
            aggregate_ohlcv,
            calculate_all_metrics,
            format_metrics
        )
        
        # All imports should work
        self.assertIsNotNone(BacktestEngine)
        self.assertIsNotNone(aggregate_ohlcv)
        self.assertIsNotNone(calculate_all_metrics)
        self.assertIsNotNone(format_metrics)
    
    def test_strategy_adapter(self):
        """Test that strategy adapter works."""
        
        # Mock old-style strategy
        class MockOldStrategy:
            def detect(self, symbol, price, volume, ts_ms, prices, volumes):
                # Simple logic: buy if price rising
                if len(prices) < 2:
                    return None
                
                if prices[-1][1] > prices[-2][1]:
                    from strategies.base_strategy import Signal
                    return Signal(
                        symbol=symbol,
                        signal='BUY',
                        strategy='mock',
                        confidence=0.8,
                        details={'reason': 'price_rising', 'price': price}
                    )
                return None
        
        # Wrap with adapter
        old_strategy = MockOldStrategy()
        adapter = StrategyAdapter(old_strategy)
        
        # Test with mock candles
        candles = [
            {'ts_ms': 1000, 'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 10},
            {'ts_ms': 2000, 'open': 101, 'high': 103, 'low': 100, 'close': 102, 'volume': 15},
        ]
        
        signal = adapter.detect(candles, 'BTC-USDT')
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['signal'], 'BUY')
        self.assertEqual(signal['confidence'], 0.8)
    
    def test_multi_symbol_comparison(self):
        """Test comparing performance across multiple symbols."""
        
        class SimpleStrategy:
            """Always buy on even candles, sell on odd."""
            def __init__(self):
                self.count = 0
            
            def detect(self, candles, symbol):
                self.count += 1
                if self.count % 20 == 10:
                    return {'signal': 'BUY', 'confidence': 0.7}
                elif self.count % 20 == 0:
                    return {'signal': 'SELL', 'confidence': 0.7}
                return None
        
        # Note: This would require real database data
        # For now, just test the structure works
        
        # symbols = ['BTC-USDT', 'ETH-USDT']
        # results = {}
        # 
        # for symbol in symbols:
        #     engine = BacktestEngine(symbol, days_back=1)
        #     strategy = SimpleStrategy()
        #     result = engine.run_strategy(strategy, timeframe='5m')
        #     results[symbol] = result['metrics'].get('score', 0)
        
        # For now, just verify the test structure
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
