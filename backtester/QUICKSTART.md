# Backtester - Quick Start Guide

Get started with backtesting in 5 minutes.

## 1. Basic Usage

```python
from backtester import BacktestEngine

# Initialize
engine = BacktestEngine(
    symbol='BTC-USDT',
    days_back=7,
    db_path='data/blofin_monitor.db'
)

# Get OHLCV data
candles_5m = engine.get_ohlcv('5m')
print(f"Loaded {len(candles_5m)} 5-minute candles")
```

## 2. Create a Simple Strategy

```python
class MyStrategy:
    """Buy when price crosses above MA, sell when crosses below."""
    
    def detect(self, candles, symbol):
        if len(candles) < 20:
            return None
        
        # Calculate 20-period moving average
        ma = sum(c['close'] for c in candles[-20:]) / 20
        current_price = candles[-1]['close']
        
        # Simple crossover logic
        if current_price > ma * 1.01:  # 1% above MA
            return {'signal': 'BUY', 'confidence': 0.7}
        elif current_price < ma * 0.99:  # 1% below MA
            return {'signal': 'SELL', 'confidence': 0.7}
        
        return None
```

## 3. Run Backtest

```python
from backtester import BacktestEngine, format_metrics

# Initialize
engine = BacktestEngine('BTC-USDT', days_back=7)

# Run strategy
results = engine.run_strategy(
    MyStrategy(),
    timeframe='5m',
    stop_loss_pct=2.0,      # 2% stop loss
    take_profit_pct=5.0     # 5% take profit
)

# Show results
print(format_metrics(results['metrics']))
```

**Output:**
```
Trades: 12 (8W / 4L)
Win Rate: 66.67%
Avg P&L: 1.45%
Total P&L: 17.40%
Sharpe: 1.82
Max Drawdown: 3.20%
Score: 72.3/100
```

## 4. Test ML Model

```python
class MyModel:
    """Predict price movement based on momentum."""
    
    def predict(self, candles, symbol):
        if len(candles) < 10:
            return 0.5
        
        # Calculate momentum
        price_10_ago = candles[-10]['close']
        current_price = candles[-1]['close']
        momentum = (current_price - price_10_ago) / price_10_ago
        
        # Convert to probability
        prob = 0.5 + momentum * 5  # Scale factor
        return max(0, min(1, prob))  # Clamp [0, 1]
```

```python
# Run model backtest
results = engine.run_model(MyModel(), timeframe='1m')

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"F1 Score: {results['f1_score']:.4f}")
```

## 5. Multi-Symbol Testing

```python
symbols = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT']
strategy = MyStrategy()

for symbol in symbols:
    engine = BacktestEngine(symbol, days_back=7)
    results = engine.run_strategy(strategy, timeframe='5m')
    
    score = results['metrics'].get('score', 0)
    print(f"{symbol}: {score:.1f}/100")
```

**Output:**
```
BTC-USDT: 72.3/100
ETH-USDT: 65.8/100
XRP-USDT: 58.2/100
```

## 6. Multi-Timeframe Analysis

```python
engine = BacktestEngine('BTC-USDT', days_back=7)
strategy = MyStrategy()

for timeframe in ['1m', '5m', '15m', '1h']:
    results = engine.run_strategy(strategy, timeframe=timeframe)
    
    metrics = results['metrics']
    print(f"{timeframe:4s}: Score={metrics['score']:.1f}, "
          f"Trades={metrics['num_trades']}, "
          f"WinRate={metrics['win_rate']:.1%}")
```

**Output:**
```
1m  : Score=45.2, Trades=89, WinRate=52%
5m  : Score=72.3, Trades=12, WinRate=67%
15m : Score=68.5, Trades=5, WinRate=60%
1h  : Score=55.1, Trades=2, WinRate=50%
```

## 7. Understanding the Score

The composite score (0-100) is calculated as:

```
score = (win_rate × 40) + (avg_pnl_pct × 30) + (sharpe × 20) - (max_drawdown × 10)
```

**Interpretation:**
- **80-100**: 🟢 Excellent - Deploy immediately
- **60-80**: 🟡 Good - Consider for production
- **40-60**: 🟠 Mediocre - Needs improvement
- **20-40**: 🔴 Poor - Major issues
- **0-20**: ⛔ Very poor - Don't use

## 8. Working with Features

Assuming you have the feature manager from Agent #1:

```python
from features import FeatureManager
from backtester import BacktestEngine

features = FeatureManager()
engine = BacktestEngine('BTC-USDT', days_back=7)

class FeatureStrategy:
    def detect(self, candles, symbol):
        # Get features
        feat = features.get_features(candles, symbol)
        
        # Use features for decision
        if feat['rsi'] < 30 and feat['macd_signal'] == 'BUY':
            return {'signal': 'BUY', 'confidence': 0.8}
        elif feat['rsi'] > 70 and feat['macd_signal'] == 'SELL':
            return {'signal': 'SELL', 'confidence': 0.8}
        
        return None

results = engine.run_strategy(FeatureStrategy(), timeframe='5m')
```

## 9. Accessing Raw Data

```python
engine = BacktestEngine('BTC-USDT', days_back=7)

# Raw ticks
print(f"Ticks: {len(engine.ticks)}")

# 1-minute candles
print(f"1m candles: {len(engine.ohlcv_1m)}")

# Custom timeframe
candles_15m = engine.get_ohlcv('15m')
candles_4h = engine.get_ohlcv('4h')
candles_1d = engine.get_ohlcv('1d')
```

## 10. Extracting Trade Details

```python
results = engine.run_strategy(strategy, timeframe='5m')

# All trades
for trade in results['trades']:
    print(f"Trade: {trade['side']} "
          f"@ ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}, "
          f"P&L: {trade['pnl_pct']:.2f}%, "
          f"Reason: {trade['reason']}")

# Equity curve
equity = results['equity_curve']
final_capital = results['final_capital']
print(f"Starting: $10,000 → Final: ${final_capital:.2f}")
```

## Common Pitfalls

❌ **Don't**: Use too short lookback (< 3 days)  
✅ **Do**: Use 7 days minimum for meaningful results

❌ **Don't**: Test only on 1 symbol  
✅ **Do**: Test on BTC, ETH, XRP minimum

❌ **Don't**: Ignore max drawdown  
✅ **Do**: Consider risk-adjusted returns (Sharpe, drawdown)

❌ **Don't**: Optimize on same data you test on  
✅ **Do**: Use walk-forward validation

## Next Steps

1. Read full [README.md](README.md) for detailed API docs
2. Check [demo_backtest.py](demo_backtest.py) for complete examples
3. Run tests: `python3 backtester/tests/test_backtest.py -v`
4. Integrate with feature manager (Agent #1)
5. Connect to ML pipeline (Agent #3)

## Questions?

- Check the tests: `backtester/tests/test_backtest.py`
- Review architecture: `../ARCHITECTURE.md`
- See implementation plan: `../IMPLEMENTATION_PLAN.md`

---

**Happy backtesting! 📈**
