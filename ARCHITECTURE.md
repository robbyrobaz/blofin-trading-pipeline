# Blofin AI Trading Pipeline — Architecture v3

## Overview

Fully automated, AI-driven trading evolution system. **No live trading yet** — pure backtest mode to rapidly iterate strategy and ML candidates.

Core principle: **Backtest everything first**, rank by performance (no hard thresholds), keep top performers, design replacements for underperformers, compose ensembles as needed.

---

## System Structure

```
blofin-stack/
├── features/                    # Feature library (ML + strategies tap here)
│   ├── __init__.py
│   ├── price_features.py       # OHLC, returns, momentum
│   ├── volume_features.py      # Volume spikes, VWAP, imbalance
│   ├── technical_indicators.py # RSI, MACD, Bollinger, ATR, etc.
│   ├── volatility_features.py  # Std dev, percentile ranges
│   ├── market_regime.py        # Trending, ranging, volatile detection
│   ├── custom_features.py      # User-defined combinations
│   └── feature_manager.py      # Central API: get_features(symbol, timeframe, feature_list)
│
├── backtester/                  # Backtesting engine
│   ├── __init__.py
│   ├── backtest_engine.py      # Core: replay data, execute logic, calc metrics
│   ├── aggregator.py           # Convert 1min → 5min → 60min
│   └── metrics.py              # Score, Sharpe, win_rate, drawdown
│
├── strategies/                  # Strategy plugins (backtest-ready)
│   ├── base_strategy.py        # Updated: detect() works in backtest mode
│   ├── [existing strategies]
│   ├── strategy_backtester.py  # NEW: Run single strategy backtest
│   └── strategy_validator.py   # NEW: Validate backtest vs live data
│
├── models/                      # ML models (ensemble-ready, separate folders)
│   ├── model_001/
│   │   ├── model.pkl           # Trained model (sklearn/xgb/torch)
│   │   ├── config.json         # Features used, hyperparams
│   │   ├── metadata.json       # Performance, creation_ts, status
│   │   └── README.md           # What this model does
│   ├── model_002/
│   ├── model_003/
│   ├── common/
│   │   ├── base_model.py       # Abstract: train(), predict(), serialize()
│   │   ├── trainer.py          # Generic training loop
│   │   └── predictor.py        # Load + predict from any model folder
│   └── ensembles/
│       ├── ensemble_001.json   # {"models": ["model_001", "model_002"], "weights": [0.6, 0.4], "method": "weighted_avg"}
│       ├── ensemble_002.json
│       └── ensemble_001_results.json  # Performance metrics
│
├── ml_pipeline/                 # ML training & tuning
│   ├── __init__.py
│   ├── train.py                # Build models on rolling 7-day windows
│   ├── validate.py             # Backtest models, calc accuracy/precision/F1
│   ├── tune.py                 # Adjust hyperparams if performance drifting
│   └── feature_selection.py    # Analyze feature importance, suggest dropouts
│
├── orchestration/               # Central coordination
│   ├── __init__.py
│   ├── daily_runner.py         # Main entry point (called by cron)
│   ├── strategy_cycle.py       # Design → backtest → validate → rank (48h)
│   ├── ml_cycle.py             # Build → backtest → validate → rank (12h)
│   ├── reporter.py             # Generate human-readable daily report
│   └── ranker.py               # Keep top X (dynamic, no hard thresholds)
│
├── data/
│   ├── blofin_monitor.db       # Live data + historical (existing)
│   ├── backtest_results/
│   │   ├── strategies/
│   │   │   ├── strategy_001_20260215.json
│   │   │   └── ...
│   │   └── models/
│   │       ├── model_001_20260215.json
│   │       └── ...
│   ├── model_history/          # Snapshots of model performance over time
│   ├── reports/                # Daily human reports
│   └── ai_reviews/             # Design prompts, decisions
│
└── docs/
    ├── ARCHITECTURE.md         # This file
    ├── FEATURE_LIBRARY.md      # How to add features
    ├── BACKTEST_GUIDE.md       # Running backtests
    ├── ML_PIPELINE.md          # Training, validation, tuning
    ├── ENSEMBLE_GUIDE.md       # Combining models
    └── STRATEGY_DESIGN.md      # Creating new strategies
```

---

## Data Flow

### Strategy Evolution (48h cycle)

```
1. SCORE EXISTING STRATEGIES (Haiku, 2 min)
   └─ Pull backtest results from last 7 days
   └─ Rank by composite score
   └─ Identify bottom 20% (candidates for replacement)

2. TUNE UNDERPERFORMERS (Sonnet, per strategy, 10 min each)
   └─ Analyze last backtest: what failed?
   └─ Suggest parameter adjustments
   └─ Run backtest with new params
   └─ If improved: keep new params, else discard

3. DESIGN NEW STRATEGIES (Opus, 15 min)
   └─ For each slot to fill: analyze what failed
   └─ Market regime analysis (trending? ranging? volatile?)
   └─ Design new strategy (return Python code)
   └─ Implement + register

4. BACKTEST NEW + TUNED (Sonnet, parallel, 20 min each)
   └─ Run backtest on last 7 days (1min, 5min, 60min aggregations)
   └─ Calculate metrics
   └─ Store results

5. VALIDATE vs LIVE (Haiku, 5 min)
   └─ For strategies running > 7 days: compare backtest ≠ live
   └─ Flag if drift > 10% (indicates overfitting or regime change)

6. RANK & UPDATE (Haiku, 2 min)
   └─ Keep top 20 active strategies
   └─ Archive bottom performers
   └─ Update database

```

### ML Pipeline (12h cycle)

```
1. BUILD MODELS (Sonnet, 30 min)
   └─ Get features from feature_manager (1min, 5min, 60min)
   └─ Train multiple models in parallel:
      ├─ XGBoost (direction prediction)
      ├─ Random Forest (risk scoring)
      ├─ Neural Net (price prediction)
      └─ SVM (classification)
   └─ Save each to models/model_XXX/

2. VALIDATE MODELS (Sonnet, 20 min)
   └─ Backtest each model on historical data (not used in training)
   └─ Calculate accuracy, precision, recall, F1
   └─ Store results

3. TEST ENSEMBLES (Sonnet, 15 min)
   └─ Try combinations of top models
   └─ Weight by individual performance
   └─ Backtest ensemble on historical data
   └─ Keep promising combinations as configs

4. TUNE DRIFTING MODELS (Sonnet, if needed, 15 min)
   └─ If model accuracy dropping: retrain with new data
   └─ If drift persistent: suggest feature redesign (Opus, 10 min)

5. RANK & DEPLOY (Haiku, 2 min)
   └─ Keep top 5 models active
   └─ Archive bottom performers
   └─ Update database
```

### Daily Orchestration (Cron)

```
Every 24 hours at 00:00 UTC:

1. Score all 20 strategies + 5 models (Haiku, 2 min)
   └─ Pull backtest results from last 7 days
   └─ Rank by score

2. [PARALLEL] Design new strategies + backtest (Opus + Sonnet, 45 min)
   ├─ Opus: Design 2-3 new candidates (analyze failures, market regime)
   ├─ Sonnet: Backtest on 1m/5m/60m (execute detect(), calc metrics)
   └─ Haiku: Validate vs live data (< 10% drift acceptable)

3. [PARALLEL] Tune underperformers (Sonnet, 20 min)
   └─ Analyze worst performers
   └─ Suggest parameter adjustments
   └─ Backtest with new params
   └─ Keep if improved

4. [PARALLEL] Build ML models (Sonnet, 50 min)
   ├─ Train 5 models in parallel (direction, risk, price, momentum, volatility)
   ├─ Backtest each on holdout data
   └─ Test ensemble combinations

5. Rank & Update (Haiku, 2 min)
   ├─ Keep top 20 strategies
   ├─ Keep top 5 models
   ├─ Keep top 3 ensembles
   └─ Archive bottom performers

6. Generate Report (Haiku, 5 min)
   ├─ What changed (strategies, models, ensembles)
   ├─ Performance metrics (top 5, performance trends)
   ├─ Who got replaced and why
   └─ Save to data/reports/YYYY-MM-DD.json

7. AI Review (Opus, 10 min)
   ├─ Read daily report
   ├─ Analyze trends
   ├─ Recommend improvements
   └─ Log to ai_reviews/

Total time: ~2.5 hours (mostly parallel)
Sleep: 21.5 hours until next cycle
```

---

## Feature Library Design

**Goal:** Universal, modular, reusable features for both strategies and ML.

```python
# features/feature_manager.py

class FeatureManager:
    """
    Central API for all features.
    Strategies and ML models request features, not implement them.
    """
    
    def get_features(self, symbol, timeframe, feature_list, lookback_bars=100):
        """
        symbol: 'BTC-USDT'
        timeframe: '1m', '5m', '60m'
        feature_list: ['close', 'rsi_14', 'macd_signal', 'volume_sma_20', ...]
        lookback_bars: how many candles of history
        
        Returns: pandas DataFrame with all requested features
        """
    
    def list_available_features(self):
        """Return all registered feature names + descriptions"""
    
    def get_feature_group(self, group_name):
        """Get pre-grouped sets: 'momentum', 'volatility', 'volume', etc."""
```

**Feature categories:**

| Category | Examples |
|----------|----------|
| **Price** | close, open, high, low, returns, log_returns, hl2, hlc3 |
| **Volume** | volume, volume_sma_20, volume_ema_9, volume_ratio |
| **Momentum** | rsi_14, rsi_7, momentum_10, rate_of_change_12 |
| **Trend** | ema_9, ema_21, sma_50, sma_200, trend_direction |
| **Volatility** | atr_14, std_dev_20, bbands_width, keltner_width |
| **MACD** | macd_12_26, macd_signal_9, macd_histogram |
| **Oscillators** | stoch_k, stoch_d, cci_20, williams_r |
| **Volume Weighted** | vwap, volume_weighted_momentum |
| **Regime** | is_trending_up, is_ranging, volatility_regime |
| **Custom** | Any combo defined by user |

---

## Backtester Design

**Key principle:** Replay last 7 days of data, execute strategy/model logic, calculate metrics.

```python
# backtester/backtest_engine.py

class BacktestEngine:
    """
    Replay historical data, execute strategy logic, calculate P&L.
    """
    
    def __init__(self, symbol, start_date, end_date):
        self.data = self._load_data(symbol, start_date, end_date)
        self.trades = []
        self.equity = 1.0  # Start with 1 unit
    
    def run_strategy(self, strategy, feature_manager):
        """
        Execute strategy on historical data.
        - For each candle: get features → call strategy.detect() → record trade
        - Calculate P&L, Sharpe, win_rate, max_drawdown
        """
    
    def run_model(self, model, feature_manager):
        """
        Execute ML model predictions on historical data.
        - For each candle: get features → call model.predict() → record prediction
        - Compare vs actual (if available) → calc accuracy, precision, recall
        """
    
    def calculate_metrics(self):
        """Return: score, sharpe, win_rate, max_drawdown, avg_pnl, total_pnl"""
```

---

## Ranking System (No Hard Thresholds)

Instead of "pass/fail", keep top performers:

```python
# orchestration/ranker.py

class StrategyRanker:
    def get_top_strategies(self, count=20):
        """
        Return top N strategies by composite score.
        Always keep top performers, always retire bottom.
        Dynamic, no fixed thresholds.
        """
    
    def get_to_replace(self, count=5):
        """Return strategies to remove (bottom 25%)"""

class ModelRanker:
    def get_top_models(self, count=5):
        """Keep top 5 models"""
    
    def get_to_replace(self, count=2):
        """Replace bottom 2"""
```

**Composite score formula (same for strategies & models):**
```
score = (win_rate * 40) + (avg_pnl_pct * 30) + (sharpe * 20) - (max_drawdown * 10)
```

---

## Database Schema

```sql
-- Existing (keep as-is)
strategy_scores, knowledge_entries, strategy_configs

-- NEW: Strategy backtest history
CREATE TABLE strategy_backtest_results (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT,
    timestamp TEXT,
    backtest_period TEXT,  -- "7d", "last_7_days"
    win_rate REAL,
    avg_pnl_pct REAL,
    sharpe REAL,
    max_drawdown REAL,
    total_trades INTEGER,
    score REAL,
    status TEXT,  -- "active", "archived", "tuning"
    tuning_attempt INTEGER,  -- 1, 2, 3
    design_prompt TEXT,  -- What Opus was asked to design
    UNIQUE(strategy_name, timestamp)
);

-- NEW: ML model history
CREATE TABLE ml_model_results (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    model_type TEXT,  -- "xgboost", "random_forest", "neural_net"
    timestamp TEXT,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    features_used TEXT,  -- JSON list
    hyperparams TEXT,    -- JSON
    status TEXT,  -- "active", "archived", "tuning"
    train_loss REAL,
    UNIQUE(model_name, timestamp)
);

-- NEW: Ensemble configs & results
CREATE TABLE ml_ensembles (
    id INTEGER PRIMARY KEY,
    ensemble_name TEXT,
    timestamp TEXT,
    model_list TEXT,     -- JSON: ["model_001", "model_002"]
    weights TEXT,        -- JSON: [0.6, 0.4]
    combination_method TEXT,  -- "weighted_avg", "voting", "stacking"
    accuracy REAL,
    f1 REAL,
    backtest_sharpe REAL,
    status TEXT,  -- "active", "archived"
    UNIQUE(ensemble_name, timestamp)
);

-- NEW: Daily reports
CREATE TABLE daily_reports (
    id INTEGER PRIMARY KEY,
    report_date TEXT,
    report_json TEXT,  -- Full report as JSON
    strategy_changes JSON,
    model_changes JSON,
    ensemble_changes JSON,
    top_strategies TEXT,  -- JSON list of top 5
    top_models TEXT,      -- JSON list of top 5
    ai_recommendations TEXT,
    timestamp TEXT
);
```

---

## Reporting Format

**Daily Report (human-readable + AI-readable JSON):**

```json
{
  "report_date": "2026-02-16",
  "cycle": "strategy_day",
  "strategy_changes": {
    "designed": [
      {
        "name": "strategy_042",
        "reason": "failed bb_squeeze pattern; tried divergence-based entry",
        "backtest_score": 42.3
      }
    ],
    "tuned": [
      {
        "name": "rsi_divergence",
        "params_changed": {"oversold_threshold": "30→28"},
        "improvement": "+2.1%"
      }
    ],
    "archived": [
      {
        "name": "strategy_035",
        "reason": "3 tuning attempts, no improvement"
      }
    ]
  },
  "ml_changes": {
    "trained": [
      {
        "name": "model_015",
        "type": "xgboost",
        "features": 25,
        "accuracy": 0.56
      }
    ],
    "ensembles_tested": [
      {
        "name": "ensemble_003",
        "models": ["model_012", "model_013"],
        "f1": 0.58
      }
    ]
  },
  "performance": {
    "top_5_strategies": [...],
    "top_5_models": [...],
    "backtest_vs_live": {
      "drift_detected": false,
      "avg_accuracy": 0.54
    }
  },
  "ai_next_steps": "Focus on mean-reversion patterns; add volatility scaling to ML features"
}
```

---

## Execution Flow (Single Cron Job)

```bash
# /usr/local/bin/blofin-ai-pipeline
#!/bin/bash

cd /home/rob/.openclaw/workspace/blofin-stack

# Every 12 hours
if [ $(($(date +%s) / 43200)) % 2 -eq 0 ]; then
    # Strategy cycle (48h = every other run)
    .venv/bin/python orchestration/daily_runner.py --mode strategy
else
    echo "Strategy cycle skipped (not strategy day)"
fi

# Always run ML cycle
.venv/bin/python orchestration/daily_runner.py --mode ml

# Generate report
.venv/bin/python orchestration/daily_runner.py --mode report

# Log completion
echo "Blofin AI pipeline completed at $(date)" >> data/pipeline.log
```

**Cron:**
```
0 */12 * * * /usr/local/bin/blofin-ai-pipeline >> /home/rob/.openclaw/workspace/data/pipeline.log 2>&1
```

---

## Implementation Checklist

- [ ] Build feature_manager.py (central feature API)
- [ ] Implement backtester (strategy + model execution)
- [ ] Create base strategy + base model abstractions
- [ ] Build strategy tuner (Sonnet prompt + implementation)
- [ ] Build strategy designer (Opus prompt + code generation)
- [ ] Build ML trainer (train multiple models in parallel)
- [ ] Build ML validator (accuracy/precision/F1 calculation)
- [ ] Build ranker (keep top X, no hard thresholds)
- [ ] Build reporter (human-readable + JSON output)
- [ ] Build orchestrator (daily_runner.py)
- [ ] Update database schema
- [ ] Write feature library docs
- [ ] Write backtest guide
- [ ] Set up cron job

---

## File Size & Performance Notes

- Backtesting 7 days of 1min/5min/60min data: ~2-5 min per strategy
- Training 1 ML model: ~5-10 min
- Parallel execution: run 5 strategies + 4 models in parallel = ~10 min total
- Feature caching: optional, will speed up repeated feature requests
- Daily cost: ~$2-5 (Opus designs, Sonnet backtests/trains, Haiku coordinates)

---

## Next Steps

1. Build feature_manager.py (feature library)
2. Build backtest_engine.py (replay + metrics)
3. Refactor existing strategies for backtest mode
4. Build ML training pipeline
5. Build orchestration scripts
6. Update all docs
7. Test on sample data
8. Deploy cron job

Ready to build?
