# Blofin AI Trading Pipeline — Implementation Plan

## Overview

Complete AI-driven trading system with continuous strategy design and ML model evolution. **No live trading yet** — pure backtest mode for rapid iteration.

Status: **Design complete, ready for implementation**

---

## Project Goals

1. ✅ **Design phase complete** (ARCHITECTURE.md)
2. 🔄 **Build feature library** (single source of truth for all features)
3. 🔄 **Build backtester** (replay 7 days of data, calc metrics)
4. 🔄 **Build ML pipeline** (train, validate, tune models)
5. 🔄 **Build orchestration** (daily cron job coordinating everything)
6. 🔄 **Build reporter** (human-readable + AI-readable daily reports)
7. 🔄 **Deploy and monitor** (cron job, log analysis)

---

## Phase 1: Foundation (Week 1)

### 1.1 Feature Library

**Deliverable:** `features/` directory with all feature calculations

```
features/
├── feature_manager.py          # Central API
├── price_features.py           # OHLC, returns, momentum
├── volume_features.py          # Volume indicators
├── technical_indicators.py     # RSI, MACD, Bollinger, ATR, etc.
├── volatility_features.py      # Volatility measures
├── market_regime.py            # Trending, ranging, volatile
├── market_microstructure.py    # (Optional: spreads, imbalance)
└── tests/test_features.py      # Unit tests
```

**Tasks:**

- [ ] Implement `feature_manager.py` with `get_features()` API
- [ ] Implement each feature module (price, volume, indicators, volatility, regime)
- [ ] Add 50+ feature combinations
- [ ] Write unit tests for each feature
- [ ] Document all available features
- [ ] Create feature grouping (momentum, volatility, trend, etc.)

**Estimated time:** 2-3 days
**Model:** Sonnet (code generation)

### 1.2 Backtester

**Deliverable:** `backtester/` with data replay engine

```
backtester/
├── backtest_engine.py         # Core: data replay + metrics
├── aggregator.py              # 1min → 5min → 60min
└── metrics.py                 # Score, Sharpe, drawdown, etc.
```

**Tasks:**

- [ ] Build `backtest_engine.py` (load 7 days of data, replay, execute logic)
- [ ] Implement OHLCV aggregator (1min → 5min → 60min)
- [ ] Implement metrics calculation (win_rate, sharpe, max_drawdown, score)
- [ ] Build strategy backtester (run strategy on historical data)
- [ ] Build model backtester (run model on historical data, calc accuracy/F1)
- [ ] Implement result saving (JSON to `data/backtest_results/`)
- [ ] Write integration tests

**Estimated time:** 3-4 days
**Model:** Sonnet (algorithm design + code)

### 1.3 Database Schema Updates

**Deliverable:** Updated `db.py` with new tables

**Tables to add:**
- `strategy_backtest_results` (strategy_name, timestamp, score, win_rate, status, tuning_attempt)
- `ml_model_results` (model_name, model_type, timestamp, accuracy, f1, features_used)
- `ml_ensembles` (ensemble_name, timestamp, model_list, weights, accuracy)
- `daily_reports` (report_date, report_json, strategy_changes, model_changes)

**Tasks:**

- [ ] Extend `db.py` with new table schemas
- [ ] Add helper functions (save_backtest_result, get_top_strategies, etc.)
- [ ] Add query functions for ranking and retrieval
- [ ] Add archival functions (mark as archived)

**Estimated time:** 1 day
**Model:** Haiku (schema design)

---

## Phase 2: ML Pipeline (Week 2)

### 2.1 Model Framework

**Deliverable:** `models/` with base classes and trainers

```
models/
├── common/
│   ├── base_model.py          # Abstract base class
│   ├── trainer.py             # Generic training loop
│   └── predictor.py           # Load + predict from any model
├── model_001/
│   ├── model.pkl
│   ├── config.json
│   └── metadata.json
└── ensembles/
    ├── ensemble_001.json
    └── ensemble_001_results.json
```

**Tasks:**

- [ ] Create `base_model.py` (abstract interface: train(), predict(), save(), load())
- [ ] Create `trainer.py` (training loop for sklearn/xgboost/torch)
- [ ] Create `predictor.py` (load model from folder, predict)
- [ ] Create ensemble predictor (weighted avg, voting, stacking)
- [ ] Implement model folder naming convention

**Estimated time:** 2 days
**Model:** Sonnet (architecture + code)

### 2.2 ML Training Pipeline

**Deliverable:** `ml_pipeline/train.py` (builds 5 models in parallel)

**Models to implement:**
1. Direction Predictor (XGBoost) — UP/DOWN
2. Risk Scorer (Random Forest) — risk level 0-100
3. Price Predictor (Neural Net) — price prediction
4. Momentum Classifier (SVM) — momentum direction
5. Volatility Regressor (Gradient Boosting) — volatility prediction

**Tasks:**

- [ ] Implement training loop
- [ ] Add data preparation (feature selection, normalization)
- [ ] Implement each model type
- [ ] Add hyperparameter definitions
- [ ] Add cross-validation
- [ ] Implement parallel training (4 workers)
- [ ] Add model persistence (save to models/model_XXX/)

**Estimated time:** 3-4 days
**Model:** Sonnet (model implementation)

### 2.3 ML Validation Pipeline

**Deliverable:** `ml_pipeline/validate.py` (backtest models)

**Tasks:**

- [ ] Implement model backtester
- [ ] Calculate accuracy, precision, recall, F1
- [ ] Test on holdout data (not used in training)
- [ ] Test on multiple symbols
- [ ] Generate validation report
- [ ] Implement ensemble validation

**Estimated time:** 2 days
**Model:** Sonnet (backtesting logic)

### 2.4 ML Tuning Pipeline

**Deliverable:** `ml_pipeline/tune.py` (detect drift, retrain)

**Tasks:**

- [ ] Implement live accuracy monitoring
- [ ] Detect drifting models (accuracy drop > 5%)
- [ ] Implement retraining logic
- [ ] Implement feature importance analysis
- [ ] Implement feature selection (drop unimportant features)
- [ ] Log all retraining decisions

**Estimated time:** 2 days
**Model:** Sonnet (monitoring + tuning)

---

## Phase 3: Strategy Evolution (Week 2-3)

### 3.1 Strategy Framework Updates

**Deliverable:** Updated strategies with backtest support

**Tasks:**

- [ ] Update `base_strategy.py` for backtest mode
- [ ] Add `required_features` attribute to each strategy
- [ ] Update `detect()` method to work with feature DataFrames
- [ ] Refactor existing strategies to use feature_manager
- [ ] Test all strategies in backtest mode

**Estimated time:** 1-2 days
**Model:** Sonnet (refactoring)

### 3.2 Strategy Tuner

**Deliverable:** `ml_pipeline/strategy_tuner.py` (Sonnet designs parameter adjustments)

**Tasks:**

- [ ] Create Sonnet prompt for parameter tuning
- [ ] Parse Sonnet output (parameter changes)
- [ ] Apply changes to strategy code
- [ ] Backtest with new parameters
- [ ] Compare old vs new
- [ ] Save tuning history to knowledge base

**Estimated time:** 1 day
**Model:** Opus (prompt design), Sonnet (execution)

### 3.3 Strategy Designer

**Deliverable:** `ml_pipeline/strategy_designer.py` (Opus creates new strategies)

**Tasks:**

- [ ] Create Opus prompt for strategy design
  - Analyze failed strategies (what patterns to avoid?)
  - Market regime (trending? ranging? volatile?)
  - Gaps in current portfolio (need more mean-reversion?)
  
- [ ] Parse Opus output (strategy code)
- [ ] Generate Python strategy file
- [ ] Register in strategy manager
- [ ] Backtest automatically
- [ ] Save design prompt + results to ai_reviews

**Estimated time:** 2 days
**Model:** Opus (prompt design), Sonnet (prompt refinement)

### 3.4 Strategy Ranking

**Deliverable:** `orchestration/strategy_ranker.py`

**Tasks:**

- [ ] Implement ranking by composite score
- [ ] Keep top 20 strategies
- [ ] Archive bottom performers
- [ ] No hard thresholds (always keep top X)
- [ ] Track ranking history

**Estimated time:** 1 day
**Model:** Haiku (ranking logic)

---

## Phase 4: Orchestration & Reporting (Week 3)

### 4.1 Daily Orchestrator

**Deliverable:** `orchestration/daily_runner.py` (main entry point)

**Execution flow:**
```
Every 12 hours:

1. Score strategies + models (Haiku, 2 min)
2. If strategy day (48h cycle):
   a. Design new strategies (Opus, 15 min)
   b. Backtest new + tune existing (Sonnet, 30 min parallel)
   c. Rank and update active pool
3. Build ML models (Sonnet, 30 min parallel)
4. Backtest models (auto, 20 min parallel)
5. Generate daily report (Haiku, 5 min)
6. AI review (Opus, 10 min, async)
```

**Tasks:**

- [ ] Create main orchestrator script
- [ ] Implement strategy cycle (48h)
- [ ] Implement ML cycle (12h)
- [ ] Add timing + logging
- [ ] Add error handling + recovery
- [ ] Implement parallel execution (use ThreadPoolExecutor)

**Estimated time:** 2 days
**Model:** Sonnet (orchestration)

### 4.2 Reporter

**Deliverable:** `orchestration/reporter.py`

**Report includes:**
- What changed (strategies designed, tuned, archived)
- What changed (models trained, ensembles tested)
- Performance metrics (top 5 strategies, top 5 models)
- AI recommendations for next cycle
- JSON + human-readable text

**Tasks:**

- [ ] Create report template (JSON structure)
- [ ] Implement strategy report generation
- [ ] Implement model report generation
- [ ] Add ensemble metrics to report
- [ ] Format for human readability
- [ ] Save to `data/reports/`

**Estimated time:** 1 day
**Model:** Sonnet (report generation)

### 4.3 Cron Setup

**Deliverable:** Systemd timer to run daily_runner.py

**Tasks:**

- [ ] Create `/usr/local/bin/blofin-ai-pipeline` script
- [ ] Set to run every 12 hours (00:00, 12:00 UTC)
- [ ] Add logging
- [ ] Test execution
- [ ] Verify cron trigger

**Estimated time:** 0.5 day

---

## Phase 5: Documentation & Testing (Week 3-4)

### 5.1 Documentation (Done!)

- ✅ ARCHITECTURE.md
- ✅ FEATURE_LIBRARY.md
- ✅ BACKTEST_GUIDE.md
- ✅ ML_PIPELINE.md
- ✅ ENSEMBLE_GUIDE.md
- [ ] STRATEGY_DESIGN.md (how to add new strategies)
- [ ] TROUBLESHOOTING.md (common issues + fixes)

**Remaining tasks:**

- [ ] Write STRATEGY_DESIGN.md
- [ ] Write TROUBLESHOOTING.md
- [ ] Create quick start guide
- [ ] Record demo video (backtest run)

**Estimated time:** 2 days

### 5.2 Testing

**Tasks:**

- [ ] Unit tests for feature_manager (all features work)
- [ ] Unit tests for backtester (metrics calculation)
- [ ] Integration tests (strategy backtest → report)
- [ ] Integration tests (model training → backtest)
- [ ] Load test (what happens with 100 strategies?)
- [ ] Error recovery test (what if Opus API fails?)

**Estimated time:** 2-3 days

### 5.3 Performance Benchmarking

**Tasks:**

- [ ] Time backtesting 1 strategy (target: 2-5 min)
- [ ] Time training 1 model (target: 5-10 min)
- [ ] Measure memory usage (target: <2GB)
- [ ] Measure cost per day (target: $2-5)
- [ ] Optimize if needed

**Estimated time:** 1 day

---

## Phase 6: Deployment & Monitoring (Week 4)

### 6.1 Pre-Launch Checklist

- [ ] All components built + tested
- [ ] Cron job running successfully
- [ ] Daily reports generating correctly
- [ ] Backtests completing in time budget
- [ ] No API failures or timeouts
- [ ] Database queries optimized
- [ ] Disk space sufficient (estimate: 5GB for backtest data)

### 6.2 Go-Live Steps

1. **Week 1:** Run backtests only (no new strategy design yet)
   - Monitor backtest quality
   - Fix any bugs
   - Optimize performance

2. **Week 2:** Enable strategy design (Opus)
   - Start designing new strategies
   - Monitor design quality
   - Collect feedback

3. **Week 3:** Enable ML pipeline
   - Start training models
   - Test ensemble combinations
   - Validate model accuracy

4. **Week 4:** Full automation
   - All cycles running
   - Daily reports + AI review
   - Monitor performance

5. **Week 5+:** Observe + iterate
   - Track strategy/model performance
   - Refine tuning thresholds
   - Adjust feature library based on learnings
   - **Only after 4+ weeks:** Consider small live trades

### 6.3 Monitoring

**Create dashboard to track:**
- Number of active strategies
- Strategy lifecycle (designed, tuned, archived)
- Top 5 strategies by score
- Number of active models
- Model accuracy trends
- Daily report generation status
- Cost per day

**Log files:**
- `/home/rob/.openclaw/workspace/data/pipeline.log` (execution)
- `/home/rob/.openclaw/workspace/data/backtest_results/` (results)
- `/home/rob/.openclaw/workspace/data/reports/` (daily reports)

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Foundation | 1 week | Feature library + backtester |
| Phase 2: ML Pipeline | 1 week | Model training + validation |
| Phase 3: Strategy Evolution | 1 week | Strategy designer + tuner |
| Phase 4: Orchestration | 1 week | Daily runner + reporter |
| Phase 5: Documentation & Testing | 1 week | Tests + docs |
| Phase 6: Deployment & Monitoring | 2+ weeks | Go-live + monitoring |
| **Total** | **6 weeks** | **Full system running** |

---

## Cost Estimates

### Development Phase

| Task | Models | Estimated Cost | Notes |
|------|--------|-----------------|-------|
| Feature library | Sonnet | $50 | Code generation |
| Backtester | Sonnet | $100 | Algorithm design + code |
| ML training | Sonnet | $150 | Model implementations |
| ML validation | Sonnet | $50 | Backtesting logic |
| ML tuning | Sonnet | $50 | Monitoring + tuning |
| Strategy tuner | Sonnet | $50 | Prompt + integration |
| Strategy designer | Opus | $300 | Prompt design (most expensive) |
| Orchestration | Sonnet | $100 | Coordination script |
| Reporter | Sonnet | $50 | Report generation |
| **Development Total** | — | **$900** | |

### Operating Phase (Per Day, After Launch)

| Task | Frequency | Cost |
|------|-----------|------|
| Score strategies | 2x/day (Haiku) | $0.10 |
| Design strategies | 1x/day (Opus) | $0.50 |
| Backtest strategies | 1x/day (Sonnet) | $0.30 |
| Build ML models | 2x/day (Sonnet) | $0.40 |
| Backtest models | 2x/day (Sonnet) | $0.30 |
| Generate reports | 2x/day (Haiku) | $0.10 |
| AI review | 1x/day (Opus) | $0.50 |
| **Daily Operating Cost** | — | **~$2.20** |
| **Monthly Operating Cost** | — | **~$66** |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Opus API fails | Design halted | Fallback to strategy variations |
| Overfitting | Bad live performance | Validate vs live data, simplify logic |
| Market regime changes | Models become stale | Add regime detection, retrain frequently |
| Disk space full | System crashes | Monitor backtest data, archive old results |
| Slow backtesting | Can't iterate fast | Optimize aggregator, cache features |
| Feature library incomplete | ML can't find good features | Extensible design, add features as needed |
| Runaway costs | Budget exceeded | Set API limits, monitor spending |

---

## Success Metrics

After 4 weeks of operation, we should see:

- ✅ **20 active strategies** (continuously rotating)
- ✅ **5+ ML models** producing >55% accuracy
- ✅ **3+ ensembles** beating individual models
- ✅ **200+ designs tested** (archive rate ~90%)
- ✅ **Backtest scores improving** over time (downward bias dropping)
- ✅ **Cost < $100/month**
- ✅ **Daily reports** accurate + actionable
- ✅ **Zero manual intervention** required

---

## Next Steps

**Immediate:**
1. Review this plan with Rob
2. Get approval on approach
3. Confirm timeline
4. Assign any modeling/debugging tasks

**Then:**
1. Spawn Sonnet subagent to start Phase 1 (feature library)
2. Create feature_manager.py skeleton
3. Begin feature implementation
4. Daily progress updates

Ready to build?
