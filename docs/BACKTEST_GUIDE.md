# Backtest Guide

## Overview

Backtesting replays the last 7 days of historical data and executes strategy/model logic to calculate performance metrics **without touching live data**.

Why backtest first?
- Eliminate bad strategies before going live
- Detect overfitting (backtest ≠ live = problem)
- Iterate fast (7-day cycles vs waiting for real money)
- Lower risk (no real trades until we're confident)

---

## Backtester Architecture

```
backtester/
├── backtest_engine.py      # Core: data replay + metrics
├── aggregator.py           # 1min → 5min → 60min
└── metrics.py              # Score, Sharpe, drawdown, etc.
```

### Data Preparation

The backtester uses the last 7 days of 1-minute candlesticks from `blofin_monitor.db`:

```sql
SELECT ts_ms, symbol, open, high, low, close, volume 
FROM ticks 
WHERE ts_ms >= (SELECT MAX(ts_ms) - 7*24*60*60*1000 FROM ticks)
ORDER BY ts_ms
```

Then aggregates to 5min and 60min internally.

---

## Backtesting a Strategy

### Manual Run

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate

python -c "
from backtester.backtest_engine import BacktestEngine
from strategies.momentum import MomentumStrategy

# Create engine for BTC-USDT, last 7 days
engine = BacktestEngine(symbol='BTC-USDT', days_back=7)

# Load strategy
strategy = MomentumStrategy()

# Run backtest on 5m timeframe
results = engine.run_strategy(strategy, timeframe='5m')

# Print results
print(f'Trades: {results[\"total_trades\"]}')
print(f'Win Rate: {results[\"win_rate\"]:.2%}')
print(f'Sharpe: {results[\"sharpe\"]:.2f}')
print(f'Max Drawdown: {results[\"max_drawdown\"]:.2%}')
print(f'Score: {results[\"score\"]:.1f}')
"
```

### Using the Backtester API

```python
from backtester.backtest_engine import BacktestEngine
from backtester.metrics import PerformanceMetrics
from features.feature_manager import FeatureManager

class StrategyBacktester:
    def __init__(self, symbol='BTC-USDT', days_back=7):
        self.engine = BacktestEngine(symbol=symbol, days_back=days_back)
        self.fm = FeatureManager()
    
    def run(self, strategy, timeframe='5m'):
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Strategy instance (must have detect() method)
            timeframe: '1m', '5m', or '60m'
        
        Returns:
            {
                'total_trades': int,
                'win_rate': float,
                'avg_pnl_pct': float,
                'sharpe': float,
                'max_drawdown': float,
                'score': float,
                'trades': [...]  # detailed trade list
            }
        """
        trades = []
        equity = 1.0
        max_equity = 1.0
        
        for candle_idx, candle in enumerate(self.engine.data):
            # Get features for this candle
            df = self.fm.get_features(
                symbol=self.engine.symbol,
                timeframe=timeframe,
                feature_list=strategy.required_features,
                lookback_bars=candle_idx + 1
            )
            
            # Ask strategy if it wants to trade
            signal = strategy.detect(
                symbol=self.engine.symbol,
                current_price=candle['close'],
                ts_ms=candle['ts_ms'],
                df=df
            )
            
            if signal:
                trade = {
                    'ts_ms': candle['ts_ms'],
                    'signal': signal.signal,
                    'entry_price': candle['close'],
                    'confidence': signal.confidence,
                    'details': signal.details
                }
                
                # Exit logic (example: next candle or stop loss)
                next_candle = self.engine.data[candle_idx + 1]
                exit_price = next_candle['close']
                pnl_pct = (exit_price - candle['close']) / candle['close']
                
                trade['exit_price'] = exit_price
                trade['pnl_pct'] = pnl_pct
                trade['win'] = pnl_pct > 0
                
                trades.append(trade)
                equity *= (1 + pnl_pct)
                max_equity = max(max_equity, equity)
        
        # Calculate metrics
        metrics = PerformanceMetrics(trades, starting_equity=1.0, final_equity=equity)
        
        return {
            'total_trades': len(trades),
            'win_rate': metrics.win_rate,
            'avg_pnl_pct': metrics.avg_pnl_pct,
            'total_pnl_pct': metrics.total_pnl_pct,
            'sharpe': metrics.sharpe,
            'max_drawdown': metrics.max_drawdown,
            'score': metrics.calculate_score(),
            'trades': trades
        }
```

---

## Backtesting an ML Model

### Manual Run

```bash
python -c "
from backtester.backtest_engine import BacktestEngine
from models.common.predictor import load_model

# Load trained model
model = load_model('model_001')

# Create engine
engine = BacktestEngine(symbol='BTC-USDT', days_back=7)

# Run backtest
results = engine.run_model(model, timeframe='1m')

# Print results
print(f'Accuracy: {results[\"accuracy\"]:.2%}')
print(f'Precision: {results[\"precision\"]:.2%}')
print(f'Recall: {results[\"recall\"]:.2%}')
print(f'F1: {results[\"f1\"]:.2f}')
"
```

### Full API

```python
from backtester.backtest_engine import BacktestEngine
from models.common.predictor import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelBacktester:
    def __init__(self, symbol='BTC-USDT', days_back=7):
        self.engine = BacktestEngine(symbol=symbol, days_back=days_back)
    
    def run(self, model, timeframe='1m'):
        """
        Backtest ML model on historical data.
        
        Returns:
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'predictions': [...],
                'actuals': [...]
            }
        """
        predictions = []
        actuals = []
        
        for i in range(len(self.engine.data) - 1):
            # Get features for next bar
            df = self.fm.get_features(
                symbol=self.engine.symbol,
                timeframe=timeframe,
                feature_list=model.features,
                lookback_bars=i + 1
            )
            
            # Predict
            pred = model.predict(df.iloc[-1:])
            predictions.append(pred[0])
            
            # Actual (did price go up?)
            next_candle = self.engine.data[i + 1]
            actual = 1 if next_candle['close'] > self.engine.data[i]['close'] else 0
            actuals.append(actual)
        
        return {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions),
            'recall': recall_score(actuals, predictions),
            'f1': f1_score(actuals, predictions),
            'predictions': predictions,
            'actuals': actuals
        }
```

---

## Multi-Timeframe Testing

Strategies often perform differently on different timeframes. Test all three:

```python
from backtester.backtest_engine import BacktestEngine

engine = BacktestEngine(symbol='BTC-USDT', days_back=7)
strategy = MyStrategy()

for timeframe in ['1m', '5m', '60m']:
    results = engine.run_strategy(strategy, timeframe=timeframe)
    print(f"{timeframe}: score={results['score']:.1f}, win_rate={results['win_rate']:.1%}")

# Output:
# 1m: score=45.2, win_rate=42%
# 5m: score=52.1, win_rate=48%
# 60m: score=38.9, win_rate=35%
```

---

## Interpreting Results

### Key Metrics

| Metric | Good Range | What It Means |
|--------|-----------|--------------|
| **Trades** | 50+ | Enough data to trust the stats |
| **Win Rate** | 50%+ | Winning more than losing (but not always) |
| **Avg PnL %** | +0.5%+ | Average profit per trade |
| **Sharpe** | 1.0+ | Risk-adjusted returns (higher=better) |
| **Max Drawdown** | <20% | Worst peak-to-trough loss |
| **Score** | 40+ | Composite metric (pass threshold) |

### Example Results

```
GOOD: win_rate=52%, avg_pnl=+0.75%, sharpe=1.4, max_dd=12%, score=54.3
  → Likely to work live

MEDIOCRE: win_rate=48%, avg_pnl=+0.02%, sharpe=0.3, max_dd=25%, score=35.1
  → Borderline, needs tuning

BAD: win_rate=45%, avg_pnl=-0.50%, sharpe=-0.2, max_dd=35%, score=18.7
  → Archiv and replace
```

---

## Detecting Overfitting

Overfitting = strategy optimized to past data but doesn't work live.

**Signs:**
- Backtest score is 60+, live score is 35
- Win rate: backtest 65%, live 45%
- Backtest uses 50+ parameters with no clear logic
- Strategy specific to one coin/timeframe

**Prevention:**
- Keep strategy logic simple
- Test on multiple timeframes
- Validate on live data (7-day comparison)
- Use reasonable parameter ranges

**Validation:**

```python
# After running live for 7 days:
def check_overfitting(strategy_name):
    backtest_results = db.query('backtest', strategy_name)
    live_results = db.query('live_trades', strategy_name, last_7_days=True)
    
    bt_score = backtest_results['score']
    live_score = live_results['score']
    drift = (bt_score - live_score) / bt_score
    
    if drift > 0.2:  # 20%+ drift
        print(f"⚠️ Overfitting detected: {drift:.1%} drift")
        # Mark for investigation or replacement
```

---

## Batch Backtesting (All Strategies)

```python
from backtester.backtest_engine import BacktestEngine
import concurrent.futures

def backtest_all_strategies(strategies, timeframes=['1m', '5m', '60m']):
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        
        for strategy in strategies:
            for timeframe in timeframes:
                engine = BacktestEngine(symbol='BTC-USDT', days_back=7)
                future = executor.submit(
                    engine.run_strategy,
                    strategy,
                    timeframe
                )
                futures[(strategy.name, timeframe)] = future
        
        for (name, tf), future in futures.items():
            results[f"{name}_{tf}"] = future.result()
    
    return results

# Run and save results
results = backtest_all_strategies(active_strategies)
db.save_backtest_results(results)
```

---

## Saving & Analyzing Results

```python
import json
from datetime import datetime

# Save individual result
result = {
    'strategy': 'strategy_042',
    'timestamp': datetime.now().isoformat(),
    'timeframe': '5m',
    'backtest_window': '7d',
    'total_trades': 152,
    'win_rate': 0.524,
    'avg_pnl_pct': 0.0087,
    'sharpe': 1.34,
    'max_drawdown': 0.158,
    'score': 48.3,
    'status': 'active'
}

filename = f"data/backtest_results/strategies/strategy_042_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(result, f, indent=2)

# Query and compare
import pandas as pd

results_df = pd.read_json('data/backtest_results/strategies/*.json')
top_10 = results_df.nlargest(10, 'score')
print(top_10[['strategy', 'timeframe', 'win_rate', 'sharpe', 'score']])
```

---

## Common Issues

### "Not enough trades"
- Strategy too conservative
- Timeframe too large (try 1m instead of 60m)
- Conditions too strict
- Solution: Loosen entry conditions or reduce lookback

### "High win rate, low PnL"
- Winning trades too small
- Losing trades too big
- Solution: Add risk management (stop loss, take profit scaling)

### "Good backtest, poor live"
- Likely overfitting
- Market regime changed (trending → ranging)
- Slippage/fees not accounted for
- Solution: Simplify, test on multiple symbols, add regime detection

### "Score 50+ but still losing money"
- Score formula may not match your goals
- Look at actual PnL, not just metrics
- Solution: Adjust score weights or focus on Sharpe ratio instead

---

## Performance Optimization

Backtesting is fast, but can optimize further:

```python
# Use multiprocessing for multiple symbols
from multiprocessing import Pool

symbols = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT']
with Pool(processes=3) as pool:
    results = pool.map(
        lambda sym: engine.run_strategy(strategy, symbol=sym),
        symbols
    )

# Cache features to avoid recalculating
feature_cache = {}
def get_cached_features(symbol, timeframe, features):
    key = (symbol, timeframe, tuple(sorted(features)))
    if key not in feature_cache:
        feature_cache[key] = fm.get_features(symbol, timeframe, features)
    return feature_cache[key]
```

---

## Next Steps

1. Build `backtester/backtest_engine.py`
2. Implement aggregator (1m → 5m → 60m)
3. Implement metrics.py (Sharpe, drawdown, etc.)
4. Refactor existing strategies for backtest mode
5. Run backtest suite on all strategies
6. Integrate with daily orchestration

The backtest engine is the foundation of rapid iteration. Get it right first.
