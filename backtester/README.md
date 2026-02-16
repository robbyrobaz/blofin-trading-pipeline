# Backtester Module

Complete backtesting engine for strategies and ML models. Replays historical data from the database and calculates performance metrics.

## Features

✅ **Data Loading**: Efficiently loads tick data from SQLite database  
✅ **OHLCV Aggregation**: Converts ticks → 1m → 5m → 15m → 1h candles  
✅ **Strategy Backtesting**: Execute strategies on historical data with realistic trade simulation  
✅ **ML Model Backtesting**: Test model predictions and calculate accuracy/F1  
✅ **Performance Metrics**: Win rate, Sharpe ratio, max drawdown, composite score  
✅ **Multi-Symbol Support**: Backtest same strategy on BTC, ETH, XRP, etc.  
✅ **Multi-Timeframe**: Test on 1m, 5m, 15m, 1h, 4h, 1d timeframes  

## Quick Start

```python
from backtester import BacktestEngine

# Initialize engine
engine = BacktestEngine(
    symbol='BTC-USDT',
    days_back=7,
    db_path='data/blofin_monitor.db'
)

# Run strategy backtest
results = engine.run_strategy(
    my_strategy,
    timeframe='5m',
    stop_loss_pct=2.0,
    take_profit_pct=5.0
)

print(results['metrics'])
# {
#   'win_rate': 0.65,
#   'avg_pnl_pct': 1.2,
#   'total_pnl_pct': 8.4,
#   'sharpe_ratio': 1.5,
#   'max_drawdown_pct': 3.2,
#   'score': 68.5,
#   'num_trades': 7
# }
```

## Module Structure

```
backtester/
├── __init__.py              # Public API exports
├── backtest_engine.py       # Core backtesting engine
├── aggregator.py            # OHLCV timeframe conversion
├── metrics.py               # Performance calculation
├── demo_backtest.py         # Usage examples
├── tests/
│   └── test_backtest.py     # Comprehensive unit tests
└── README.md                # This file
```

## Components

### BacktestEngine

Core engine for running backtests.

**Methods:**

- `__init__(symbol, days_back=7, db_path, initial_capital=10000)` - Initialize
- `get_ohlcv(timeframe='1m')` - Get OHLCV data for timeframe
- `run_strategy(strategy, timeframe='5m', ...)` - Run strategy backtest
- `run_model(model, timeframe='1m', ...)` - Run ML model backtest

**Example:**

```python
engine = BacktestEngine('BTC-USDT', days_back=7)

# Get different timeframes
candles_1m = engine.get_ohlcv('1m')
candles_5m = engine.get_ohlcv('5m')
candles_1h = engine.get_ohlcv('1h')
```

### Aggregator Functions

Timeframe conversion utilities.

**Functions:**

- `aggregate_ohlcv(candles_1m, target_minutes)` - Aggregate to higher timeframe
- `aggregate_ticks_to_1m_ohlcv(ticks)` - Ticks → 1m candles
- `fast_aggregate_numpy(candles, target_minutes)` - Fast numpy-based aggregation
- `timeframe_to_minutes(timeframe)` - Convert '5m' → 5

**Example:**

```python
from backtester import aggregate_ohlcv

# Convert 1m → 5m
candles_5m = aggregate_ohlcv(candles_1m, 5)

# Convert 1m → 1h
candles_1h = aggregate_ohlcv(candles_1m, 60)
```

### Metrics Functions

Performance calculation utilities.

**Functions:**

- `calculate_win_rate(trades)` - Win rate (0-1)
- `calculate_avg_pnl_pct(trades)` - Average P&L %
- `calculate_total_pnl_pct(trades)` - Total P&L %
- `calculate_sharpe_ratio(returns)` - Sharpe ratio
- `calculate_max_drawdown(equity_curve)` - Max drawdown %
- `calculate_score(metrics)` - Composite score (0-100)
- `calculate_all_metrics(trades, equity_curve)` - All metrics at once
- `format_metrics(metrics)` - Human-readable string

**Example:**

```python
from backtester import calculate_all_metrics, format_metrics

metrics = calculate_all_metrics(trades, equity_curve)
print(format_metrics(metrics))
```

### Score Formula

Composite score (0-100):

```
score = (win_rate × 40) + (avg_pnl_pct × 30) + (sharpe × 20) - (max_drawdown × 10)
```

**Interpretation:**

- **80-100**: Excellent strategy
- **60-80**: Good strategy
- **40-60**: Mediocre strategy
- **20-40**: Poor strategy
- **0-20**: Very poor strategy

## Strategy Interface

Strategies must implement a `detect()` method:

```python
class MyStrategy:
    def detect(self, candles, symbol):
        """
        Detect trading signals.
        
        Args:
            candles: List of OHLCV candles (most recent last)
            symbol: Trading symbol (e.g., 'BTC-USDT')
        
        Returns:
            Signal dict or None:
            {
                'signal': 'BUY' or 'SELL',
                'confidence': 0.0 - 1.0
            }
        """
        # Your logic here
        if should_buy:
            return {'signal': 'BUY', 'confidence': 0.8}
        elif should_sell:
            return {'signal': 'SELL', 'confidence': 0.7}
        return None
```

## ML Model Interface

Models must implement a `predict()` method:

```python
class MyModel:
    def predict(self, candles, symbol):
        """
        Predict next price movement.
        
        Args:
            candles: List of OHLCV candles (most recent last)
            symbol: Trading symbol
        
        Returns:
            Probability (0-1) that price will go up
        """
        # Your logic here
        features = extract_features(candles)
        probability = self.model.predict_proba(features)[0][1]
        return probability
```

## Testing

Run comprehensive tests:

```bash
# Activate virtualenv
source .venv/bin/activate

# Run tests
python3 backtester/tests/test_backtest.py -v
```

**Test Coverage:**

- ✅ OHLCV aggregation (1m → 5m → 60m)
- ✅ Tick → 1m conversion
- ✅ Fast numpy aggregation
- ✅ Metrics calculation (all functions)
- ✅ Strategy backtesting
- ✅ ML model backtesting
- ✅ Edge cases (empty trades, zero volatility, etc.)

**18 tests, all passing.**

## Performance

**Benchmarks (1 day of data, ~171k ticks):**

- Load ticks from DB: ~0.5s
- Generate 1m candles: ~1.5s
- Aggregate to 5m: ~0.1s
- Run strategy (5m): ~0.2s
- Calculate metrics: <0.01s

**Total backtest time: ~2-3 seconds per symbol**

## Multi-Symbol Testing

Test same strategy on multiple symbols:

```python
symbols = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT']

for symbol in symbols:
    engine = BacktestEngine(symbol, days_back=7)
    results = engine.run_strategy(my_strategy, timeframe='5m')
    
    print(f"{symbol}: Score = {results['metrics']['score']:.1f}")
```

## Multi-Timeframe Analysis

Compare strategy performance across timeframes:

```python
engine = BacktestEngine('BTC-USDT', days_back=7)

for tf in ['1m', '5m', '15m', '1h']:
    results = engine.run_strategy(my_strategy, timeframe=tf)
    print(f"{tf}: {results['metrics']['score']:.1f}")
```

## Known Limitations

- **Data Requirements**: Requires at least 7 days of tick data in database
- **Memory Usage**: Large datasets (>1M ticks) may require significant RAM
- **Execution Time**: 7 days of tick data takes ~30-60s to process
- **No Slippage Model**: Uses close price for entry/exit (optimistic)
- **No Commission Model**: P&L doesn't account for fees
- **Single Position**: Only supports one position at a time

## Future Enhancements

Potential improvements:

- [ ] Multi-position support (portfolio backtesting)
- [ ] Slippage and commission modeling
- [ ] Order book replay (bid/ask spread)
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Risk management (position sizing, max drawdown limits)
- [ ] Report generation (HTML/PDF reports)
- [ ] Parallel backtesting (multiple symbols at once)

## Integration

Works seamlessly with other Blofin stack components:

```python
# With feature manager (Agent #1)
from features import FeatureManager
from backtester import BacktestEngine

features = FeatureManager()
engine = BacktestEngine('BTC-USDT', days_back=7)

class FeatureBasedStrategy:
    def detect(self, candles, symbol):
        feat = features.get_features(candles, symbol)
        if feat['rsi'] < 30:
            return {'signal': 'BUY', 'confidence': 0.8}
        return None

results = engine.run_strategy(FeatureBasedStrategy())
```

## Support

For issues or questions:

1. Check test cases in `tests/test_backtest.py`
2. Run demo: `python3 backtester/demo_backtest.py`
3. Review ARCHITECTURE.md in root directory

---

**Built by Agent #2 as part of the Blofin AI Trading Pipeline.**
