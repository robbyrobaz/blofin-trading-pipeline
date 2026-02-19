# ML Fix Plan

## Analysis Summary

### 1. Data Leakage in Existing Models

**direction_predictor.py** (lines 24-27):
- `returns` feature: If this is `pct_change()` (current close - prev close), it encodes directional movement of the CURRENT candle, not future. This is a subtle but real leakage because the target is `(close.shift(-5) > close)` — the current `returns` will be correlated with the near-future direction since momentum persists. However, the real data-leakage concern per the brief is that `returns` essentially encodes the label directly.
- **Fix**: Remove `returns` from `direction_predictor` feature list. Use only lagged/indicator-based features.

**momentum_classifier.py** (lines 26-28):
- `returns` is included. Same concern — also `acceleration` (derivative of returns) is included.
- **Fix**: Remove `returns` and `acceleration` from features.

**Other models** (price_predictor, risk_scorer, volatility_regressor):
- No direct leakage found. `close`, `high`, `low`, `volume` in price_predictor are raw inputs — predicting `target_price = close.shift(-5)` with current `close` as a feature is trivially high-accuracy (tomorrow's price ≈ today's price + noise). This is structural leakage.
- **Fix**: Remove `close`, `high`, `low` from price_predictor features. Keep computed indicators only.

**universal_trainer.py**:
- Uses `train_test_split` with `random_state=42` (shuffled). This shuffles time-series data, leaking future info.
- **Fix**: Replace with temporal split (no shuffle).

### 2. Entry Classifier (ML signal→outcome model)

**Training data available**:
- 35,903 closed paper trades
- Join: `paper_trades → confirmed_signals → signals` for strategy/features
- Labels: binary `pnl_pct > 0` (win/loss)
- Features at signal time T from `signals.details_json` + strategy metadata + confirmed_signals.score + side + symbol

**Features extractable**:
- `strategy` (categorical, one-hot or ordinal)
- `symbol` (categorical)
- `side` (BUY/SELL)
- `confidence` from signals table
- `score` from confirmed_signals table
- Strategy-specific fields from `details_json` (e.g., band_width_pct, rsi, deviation_pct, etc.)
- Time-of-day / day-of-week from ts_ms

**Label**: `1 if pnl_pct > 0 else 0`

**Temporal embargo**: 24h gap between train/test split (chronological, no shuffle).

### 3. Exit Timing Model

**Training data**:
- Closed trades with `opened_ts_ms`, `closed_ts_ms`, `entry_price`, `exit_price`, `pnl_pct`
- For each open trade state: time-in-trade, unrealized PnL, market conditions → should we hold or exit?

**Features**:
- `age_min` (minutes since open)
- `unrealized_pnl_pct`
- `side` (BUY/SELL)
- `symbol`
- Strategy of origin
- Time of day

**Label**: `1=CLOSE_NOW` if closing at that moment would yield > 0% PnL, else `0=HOLD`
- Use actual pnl_pct at close as ground truth.

### 4. Wire ML into Confirmation Pipeline

Modify `paper_engine.py::maybe_confirm_signals()` to:
1. Load entry classifier (if trained model exists on disk)
2. Build feature vector for each candidate signal
3. If `entry_classifier.predict_proba() > threshold`, allow confirmation
4. Add ML score to confirmed_signals rationale

## Implementation Plan

### Step 1: Fix data leakage in existing models
- `direction_predictor.py`: Remove `returns` from feature list
- `momentum_classifier.py`: Remove `returns` and `acceleration` from feature list
- `price_predictor.py`: Remove `close`, `high`, `low` (raw price levels) from features
- `universal_trainer.py`: Change `train_test_split` to temporal split

### Step 2: Build Entry Classifier
- Create `ml_pipeline/models/entry_classifier.py`
  - XGBoost binary classifier (win/loss prediction)
  - Temporal split with 24h embargo
  - Walk-forward validation

### Step 3: Build data loader for entry classifier
- Create `ml_pipeline/build_entry_dataset.py`
  - Queries `paper_trades JOIN confirmed_signals JOIN signals`
  - Extracts features from `details_json`
  - Returns sorted-by-time DataFrame
  - Applies 24h temporal embargo between train/test

### Step 4: Build Exit Timing Model
- Create `ml_pipeline/models/exit_classifier.py`
  - Binary classifier: hold vs close-now
  - Uses per-trade time series (age, unrealized PnL)

### Step 5: Wire ML into paper_engine.py
- Add optional ML scoring gate in `maybe_confirm_signals()`
- Load entry classifier model from disk (graceful fallback if not present)
- Include ML confidence in the `score` field saved to `confirmed_signals`

## Files to Create/Modify

### Modify:
- `ml_pipeline/models/direction_predictor.py` — remove `returns` from features
- `ml_pipeline/models/momentum_classifier.py` — remove `returns`, `acceleration`
- `ml_pipeline/models/price_predictor.py` — remove `close`, `high`, `low`
- `ml_pipeline/universal_trainer.py` — fix shuffle=True temporal split bug
- `paper_engine.py` — add ML scoring gate in `maybe_confirm_signals()`

### Create:
- `ml_pipeline/models/entry_classifier.py` — new ML entry classifier
- `ml_pipeline/build_entry_dataset.py` — data loader for entry classifier training
- `ml_pipeline/models/exit_classifier.py` — new exit timing model
- `ml_pipeline/train_entry_exit.py` — training script for both new models

## Risk / Scope Notes
- We have ~35k trades but only ~7 days of history (2026-02-11 to 2026-02-18). 24h embargo will use ~5k test samples.
- The `details_json` feature keys vary by strategy (bb_squeeze has band_width_pct, rsi_divergence has rsi, etc.). We'll use a flat feature vector with NaN-fill for missing strategy fields.
- Wire-in to paper_engine is read-only for now (model gate, not replacing the strategy-count confirmation logic). This ensures no breaking change if model file is missing.
