#!/usr/bin/env python3
"""
Demonstration of backtester usage.
Shows how to run strategies and models on historical data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtester import BacktestEngine, format_metrics


class SimpleMovingAverageStrategy:
    """
    Simple Moving Average crossover strategy.
    Buy when fast MA crosses above slow MA, sell when crosses below.
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def detect(self, candles, symbol):
        """Detect trading signals based on MA crossover."""
        if len(candles) < self.slow_period:
            return None
        
        # Calculate moving averages
        fast_ma = sum(c['close'] for c in candles[-self.fast_period:]) / self.fast_period
        slow_ma = sum(c['close'] for c in candles[-self.slow_period:]) / self.slow_period
        
        # Previous candles for crossover detection
        if len(candles) < self.slow_period + 1:
            return None
        
        prev_fast_ma = sum(c['close'] for c in candles[-self.fast_period-1:-1]) / self.fast_period
        prev_slow_ma = sum(c['close'] for c in candles[-self.slow_period-1:-1]) / self.slow_period
        
        # Detect crossover
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            return {'signal': 'BUY', 'confidence': 0.7}
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            return {'signal': 'SELL', 'confidence': 0.7}
        
        return None


class SimpleMomentumModel:
    """
    Simple momentum-based ML model.
    Predicts price will go up if recent momentum is positive.
    """
    
    def predict(self, candles, symbol):
        """Predict next price movement based on momentum."""
        if len(candles) < 10:
            return 0.5
        
        # Calculate momentum (rate of change over last 10 candles)
        price_10_ago = candles[-10]['close']
        current_price = candles[-1]['close']
        momentum = (current_price - price_10_ago) / price_10_ago
        
        # Convert momentum to probability (sigmoid-like)
        # Positive momentum -> higher probability
        prob = 0.5 + (momentum * 10)  # Scale momentum
        prob = max(0, min(1, prob))  # Clamp to [0, 1]
        
        return prob


def run_demo():
    """Run backtesting demo."""
    print("=" * 60)
    print("Backtester Demonstration")
    print("=" * 60)
    
    # Test multiple symbols
    symbols = ['BTC-USDT', 'ETH-USDT']
    
    for symbol in symbols:
        print(f"\n{'=' * 60}")
        print(f"Testing {symbol}")
        print(f"{'=' * 60}")
        
        try:
            # Initialize engine
            engine = BacktestEngine(
                symbol=symbol,
                days_back=7,
                db_path='data/blofin_monitor.db'
            )
            
            print(f"\nLoaded {len(engine.ticks)} ticks")
            print(f"Generated {len(engine.ohlcv_1m)} 1-minute candles")
            
            # Test strategy backtesting
            print(f"\n{'-' * 60}")
            print("Testing Simple Moving Average Strategy")
            print(f"{'-' * 60}")
            
            strategy = SimpleMovingAverageStrategy(fast_period=5, slow_period=20)
            
            for timeframe in ['5m', '15m', '1h']:
                print(f"\nTimeframe: {timeframe}")
                
                results = engine.run_strategy(
                    strategy,
                    timeframe=timeframe,
                    stop_loss_pct=2.0,
                    take_profit_pct=5.0
                )
                
                print(f"Candles analyzed: {results.get('num_candles', 0)}")
                
                if results.get('trades'):
                    print(format_metrics(results['metrics']))
                else:
                    print("No trades executed")
            
            # Test model backtesting
            print(f"\n{'-' * 60}")
            print("Testing Simple Momentum Model")
            print(f"{'-' * 60}")
            
            model = SimpleMomentumModel()
            
            for timeframe in ['1m', '5m']:
                print(f"\nTimeframe: {timeframe}")
                
                results = engine.run_model(model, timeframe=timeframe)
                
                print(f"Predictions: {results.get('num_predictions', 0)}")
                print(f"Accuracy: {results.get('accuracy', 0):.2%}")
                print(f"Precision: {results.get('precision', 0):.2%}")
                print(f"Recall: {results.get('recall', 0):.2%}")
                print(f"F1 Score: {results.get('f1_score', 0):.4f}")
        
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    run_demo()
