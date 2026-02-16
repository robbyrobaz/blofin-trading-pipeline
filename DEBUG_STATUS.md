# ML Pipeline Debug Status

**Started:** 2026-02-16 00:04 MST  
**Status:** In Progress (3 agents working in parallel)

---

## Problem Identified

**ML Models Return 0 Trained**

Root cause: `prepare_training_data()` returns empty because features DataFrame lacks **target columns** (labels for supervised learning).

Example:
```
Target generation missing:
  - target_direction (UP/DOWN for next 5 candles)
  - target_price (future price)
  - target_momentum (momentum direction)
  - target_volatility (future volatility)
```

---

## Fixes Applied

### 1. Target Generator (New File)
- **File:** `ml_pipeline/target_generator.py`
- **What:** Generates training targets from price data
- **How:** Shifts price forward by N candles to create labels

### 2. Universal Trainer (New File)
- **File:** `ml_pipeline/universal_trainer.py`
- **What:** Trains 5 ML models that work with ANY feature set
- **Why:** Previous trainer expected hardcoded column names that don't exist
- **Models:**
  - Direction classifier
  - Volatility regressor
  - Price predictor
  - Momentum classifier
  - Risk scorer

### 3. Updated TrainingPipeline
- **File:** `ml_pipeline/train.py`
- **Change:** Auto-generates targets if missing

### 4. Updated Orchestrator
- **File:** `orchestration/daily_runner.py`
- **Change:** Now uses universal_trainer instead of broken implementation

---

## Agents Working on This

### Agent #1: `debug-ml-pipeline` (Sonnet)
**Status:** Building/testing ML integration
**Task:** Verify all components and test with real data

### Agent #2: `blofin-hourly-monitor` (Haiku)
**Status:** Running pipeline hourly for testing
**Task:** Run every hour, check if models_trained > 0, stop when working

---

## Next Steps

1. **Wait for agents to complete** (~15-30 minutes)
2. **Check pipeline.log for results** (`grep models_trained`)
3. **Verify model files created** (`ls models/`)
4. **Run full pipeline manually** to confirm

---

## Expected Success Metrics

When working, you should see:
```
Training ML models...
✓ Direction: 85.23%
✓ Volatility: RMSE=0.0123
✓ Price: RMSE=234.56
✓ Momentum: 72.34%
✓ Risk scorer: 68.12%

Successfully trained 5 models
```

Instead of current:
```
Training ML models (STUB - will integrate with ml_pipeline)
✓ train_ml_models: {"models_trained": 0, ...}
```

---

## Files Changed

- ✅ `ml_pipeline/target_generator.py` - NEW (targets generation)
- ✅ `ml_pipeline/universal_trainer.py` - NEW (works with any features)
- ✅ `ml_pipeline/train.py` - MODIFIED (auto-generate targets)
- ✅ `orchestration/daily_runner.py` - MODIFIED (use universal trainer)

---

## Monitoring

Check progress with:
```bash
tail -100 data/pipeline.log | grep -i "model\|train"
```

When successful, should see:
```
Training 5 ML models...
  ✓ Direction: 85.23%
  ✓ Volatility: RMSE=0.0123
  ...
Successfully trained 5 models
```
