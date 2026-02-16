# Integration Guide for Agents #1-3

This guide explains how to integrate your components with the orchestration layer.

## Agent #1: Strategy Manager Integration

### Required Interface

The orchestration layer expects a strategy scoring module at the project root level.

**Location:** `strategy_manager.py` or create `orchestration/strategy_scorer.py`

**Expected Interface:**
```python
def score_all_strategies(db_path: str, window: str = '7d') -> Dict[str, Any]:
    """
    Score all active strategies and update database.
    
    Args:
        db_path: Path to SQLite database
        window: Time window for scoring ('7d', '30d', etc.)
    
    Returns:
        {
            'scored_count': int,
            'avg_score': float,
            'top_strategy': str,
            'duration_seconds': float
        }
    """
    pass
```

### Integration Point

Edit `orchestration/daily_runner.py`, replace the stub in `step_score_strategies()`:

```python
def step_score_strategies(self) -> Dict[str, Any]:
    self._log_step('score_strategies', 'started')
    start_time = datetime.utcnow()
    
    try:
        # Import your module
        from strategy_manager import score_all_strategies
        
        # Call it
        result = score_all_strategies(str(self.db_path), window='7d')
        
        self._log_step('score_strategies', 'success', result)
        return result
        
    except Exception as e:
        self._log_step('score_strategies', 'failure', {'error': str(e)})
        return {'error': str(e)}
```

### Database Schema

Your module should write to these tables:
- `strategy_scores` - Current scores (already exists)
- `strategy_backtest_results` - Detailed backtest metrics (newly added)

Example:
```python
con.execute('''
    INSERT INTO strategy_backtest_results 
    (ts_ms, ts_iso, strategy, symbol, backtest_window_days, 
     total_trades, win_rate, sharpe_ratio, max_drawdown_pct, 
     total_pnl_pct, avg_pnl_pct, score, config_json, metrics_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (...))
```

---

## Agent #2: ML Pipeline Integration

### Required Interface

**Location:** Create `ml_pipeline/trainer.py`

**Expected Interface:**
```python
def train_all_models(db_path: str, features_config: Dict) -> Dict[str, Any]:
    """
    Train all ML models and ensembles.
    
    Args:
        db_path: Path to SQLite database
        features_config: Feature engineering configuration
    
    Returns:
        {
            'models_trained': int,
            'ensembles_tested': int,
            'best_model': str,
            'best_f1': float,
            'duration_seconds': float
        }
    """
    pass
```

### Integration Point

Edit `orchestration/daily_runner.py`, replace the stub in `step_train_ml_models()`:

```python
def step_train_ml_models(self) -> Dict[str, Any]:
    self._log_step('train_ml_models', 'started')
    start_time = datetime.utcnow()
    
    try:
        # Import your module
        from ml_pipeline.trainer import train_all_models
        
        # Call it
        features_config = {
            'window_sizes': [5, 10, 20],
            'indicators': ['rsi', 'macd', 'bbands', 'volume']
        }
        result = train_all_models(str(self.db_path), features_config)
        
        self._log_step('train_ml_models', 'success', result)
        return result
        
    except Exception as e:
        self._log_step('train_ml_models', 'failure', {'error': str(e)})
        return {'error': str(e)}
```

### Database Schema

Your module should write to these tables:
- `ml_model_results` - Individual model results
- `ml_ensembles` - Ensemble configurations and results

Example:
```python
# Save model results
con.execute('''
    INSERT INTO ml_model_results 
    (ts_ms, ts_iso, model_name, model_type, symbol, features_json,
     train_accuracy, test_accuracy, f1_score, precision_score, 
     recall_score, roc_auc, config_json, metrics_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (...))

# Save ensemble results
con.execute('''
    INSERT INTO ml_ensembles 
    (ts_ms, ts_iso, ensemble_name, model_ids_json, symbol,
     test_accuracy, f1_score, voting_method, config_json, metrics_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (...))
```

---

## Agent #3: Backtesting Integration

### Required Interface

**Location:** Create `backtesting/runner.py` or extend existing backtest code

**Expected Interface:**
```python
def backtest_strategies(strategy_names: List[str], db_path: str, 
                       window_days: int = 30) -> Dict[str, Any]:
    """
    Backtest a list of strategies.
    
    Args:
        strategy_names: List of strategy names to backtest
        db_path: Path to SQLite database
        window_days: Number of days to backtest
    
    Returns:
        {
            'backtested_count': int,
            'results': [
                {
                    'strategy': str,
                    'total_trades': int,
                    'win_rate': float,
                    'sharpe_ratio': float,
                    'score': float
                },
                ...
            ],
            'duration_seconds': float
        }
    """
    pass
```

### Integration Point

Edit `orchestration/daily_runner.py`, replace the stub in `step_backtest_new_strategies()`:

```python
def step_backtest_new_strategies(self, designed_strategies: List[str]) -> Dict[str, Any]:
    self._log_step('backtest_strategies', 'started')
    start_time = datetime.utcnow()
    
    try:
        # Import your module
        from backtesting.runner import backtest_strategies
        
        # Call it
        result = backtest_strategies(
            designed_strategies, 
            str(self.db_path), 
            window_days=30
        )
        
        self._log_step('backtest_strategies', 'success', result)
        return result
        
    except Exception as e:
        self._log_step('backtest_strategies', 'failure', {'error': str(e)})
        return {'error': str(e)}
```

### Database Schema

Your module should write to:
- `strategy_backtest_results` - Detailed backtest metrics

---

## Testing Integration

### 1. Create Mock Data

Before integrating, ensure you have test data:

```python
import sqlite3
from datetime import datetime

con = sqlite3.connect('data/blofin_monitor.db')

# Add test strategy scores
ts_ms = int(datetime.utcnow().timestamp() * 1000)
ts_iso = datetime.utcnow().isoformat() + 'Z'

con.execute('''
    INSERT INTO strategy_scores 
    (ts_ms, ts_iso, strategy, symbol, window, trades, wins, losses, 
     win_rate, avg_pnl_pct, total_pnl_pct, sharpe_ratio, score, enabled)
    VALUES (?, ?, 'test_strategy', 'BTC-USDT', '7d', 100, 60, 40, 
            60.0, 0.5, 10.0, 1.5, 0.75, 1)
''', (ts_ms, ts_iso))

con.commit()
con.close()
```

### 2. Test Individual Modules

```bash
# Test your strategy scorer
python3 strategy_manager.py

# Test your ML trainer
python3 ml_pipeline/trainer.py

# Test your backtester
python3 backtesting/runner.py
```

### 3. Test Orchestration

```bash
# Dry run (with stubs)
python3 orchestration/daily_runner.py

# Check logs
tail -f data/pipeline.log
```

### 4. Integration Checklist

- [ ] My module writes to the correct database tables
- [ ] My module returns the expected dictionary structure
- [ ] My module handles errors gracefully
- [ ] My module logs progress
- [ ] My module is importable from daily_runner.py
- [ ] Integration point in daily_runner.py is updated
- [ ] End-to-end test passes

---

## Communication Between Components

### Shared Database

All components share the same SQLite database (`data/blofin_monitor.db`). Use transactions:

```python
con = sqlite3.connect(db_path, timeout=30)
try:
    # Do work
    con.execute(...)
    con.commit()
finally:
    con.close()
```

### File-Based Communication

Strategy designer outputs files to `strategies/strategy_NNN.py`. Your backtester should:

1. Load strategy file
2. Instantiate strategy class
3. Run backtest
4. Save results to database

### Configuration

Store configuration in database tables:

- `strategy_configs` - Strategy parameters
- `knowledge_entries` - Learnings and insights

---

## Parallel Execution

The orchestration layer runs some tasks in parallel:

```python
parallel_tasks = [
    ('design', self.step_design_strategies, ()),
    ('tune', self.step_tune_underperformers, ()),
    ('ml_train', self.step_train_ml_models, ())
]
```

**Important:** Your modules must be thread-safe or use separate database connections.

---

## Next Steps

1. Review your module's interface
2. Ensure database writes match the schema
3. Test your module standalone
4. Update the integration point in `daily_runner.py`
5. Run full pipeline test
6. Monitor logs for errors

Questions? Check the orchestration README or examine existing modules (ranker, reporter) as examples.
