# Blofin Pipeline Debug Report

**Generated:** 2026-02-16 09:44 MST  
**Analyst:** Subagent (Debug Task)

---

## Executive Summary

Identified **4 critical issues** preventing the Blofin pipeline from working correctly:

1. ✗ **Strategy Tuner** - Empty Sonnet output (subprocess + parsing issues)
2. ✗ **Backtest Path Mismatch** - Empty strategy files (019-022)
3. ✗ **Feature Manager NaN** - Integer conversion error
4. ✗ **Ranker Issues** - Metric key mismatch (val_accuracy ≠ test_accuracy)

All issues have **identified root causes** and **proposed fixes**.

---

## Issue #1: Strategy Tuner - Empty Sonnet Output

### Symptoms
```
"strategies_tuned": 0
```

### Root Cause
**File:** `orchestration/strategy_tuner.py`

1. **Subprocess Call (Line 182-189):**
   ```python
   result = subprocess.run(
       ['openclaw', 'chat', '--model', 'sonnet', '--prompt', prompt],
       capture_output=True,
       text=True,
       timeout=120
   )
   return result.stdout.strip()
   ```
   - Calling `openclaw chat` via subprocess is fragile
   - Output may include ANSI codes, metadata, or formatting
   - Prompt may be too long for command-line argument

2. **Parsing Strategy (Line 194-252):**
   - Falls back to extracting JSON from text
   - If Sonnet wraps response in markdown or adds explanation, parsing fails
   - Returns `None` when no valid JSON found, but doesn't log the raw output

### Proposed Fix

**Option A: Use OpenClaw API directly (preferred)**
```python
def _call_sonnet(self, prompt: str) -> str:
    """Call Claude Sonnet via OpenClaw API."""
    try:
        import requests
        response = requests.post(
            'http://localhost:8780/api/chat',  # Or gateway URL
            json={
                'model': 'sonnet',
                'prompt': prompt,
                'temperature': 0.7
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        raise Exception(f"Failed to call Sonnet: {e}")
```

**Option B: Fix subprocess call**
```python
def _call_sonnet(self, prompt: str) -> str:
    """Call Claude Sonnet via OpenClaw CLI with file-based prompt."""
    import tempfile
    try:
        # Write prompt to temp file to avoid CLI length limits
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name
        
        result = subprocess.run(
            ['openclaw', 'chat', '--model', 'sonnet', '--file', prompt_file],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, 'NO_COLOR': '1'}  # Disable ANSI colors
        )
        
        os.unlink(prompt_file)
        
        # Log raw output for debugging
        if not result.stdout.strip():
            print(f"WARNING: Empty Sonnet output. stderr: {result.stderr}")
        
        return result.stdout.strip()
    except Exception as e:
        raise Exception(f"Failed to call Sonnet: {e}")
```

**Option C: Add logging + fallback**
```python
def _parse_tuning_suggestions(self, sonnet_output: str) -> Optional[Dict[str, Any]]:
    """Parse JSON suggestions with better logging."""
    if not sonnet_output or not sonnet_output.strip():
        print("ERROR: Empty Sonnet output")
        return None
    
    # Log first 500 chars for debugging
    print(f"DEBUG: Sonnet output preview: {sonnet_output[:500]}...")
    
    # [existing parsing logic]
    
    # If all parsing fails, log the full output
    print(f"ERROR: Failed to parse Sonnet output. Full output:\n{sonnet_output}")
    return None
```

---

## Issue #2: Backtest Path Mismatch - Empty Strategy Files

### Symptoms
```
[WARNING] Strategy strategy_019 not found, skipping backtest
[WARNING] Strategy strategy_020 not found, skipping backtest
```

### Evidence
```bash
$ ls -lh strategies/strategy_019.py
-rwxr-xr-x 1 rob rob 0 Feb 16 08:01 strategy_019.py
```

All `strategy_019.py` through `strategy_022.py` are **0 bytes**.

### Root Cause
**File:** `orchestration/strategy_designer.py`

1. **Opus Call Fails (Line 277-286):**
   - Same subprocess issue as Strategy Tuner
   - If Opus returns empty or malformed output, `_call_opus()` returns empty string

2. **Code Extraction (Line 288-301):**
   ```python
   def _extract_code(self, opus_output: str) -> str:
       """Extract Python code from Opus output."""
       code_match = re.search(r'```python\n(.*?)```', opus_output, re.DOTALL)
       if code_match:
           return code_match.group(1).strip()
       
       if opus_output.strip().startswith('#!/usr/bin/env python'):
           return opus_output.strip()
       
       # Last resort: assume entire output is code
       return opus_output.strip()  # ← Returns empty string if opus_output is empty!
   ```

3. **File Save (Line 312-320):**
   ```python
   def _save_strategy(self, code: str, strategy_num: int) -> Path:
       filepath = self.strategies_dir / f"strategy_{strategy_num:03d}.py"
       with open(filepath, 'w') as f:
           f.write(code)  # ← Writes empty string!
       filepath.chmod(0o755)
       return filepath
   ```

4. **No Validation:**
   - No check if `code` is empty before saving
   - No check if file was written successfully
   - Database registration succeeds even with empty file

### Proposed Fix

**Add validation + error handling:**

```python
def design_new_strategy(self) -> Optional[Dict[str, Any]]:
    """Design a new strategy using Opus."""
    con = self._connect()
    try:
        # [existing code for gathering intelligence and building prompt]
        
        print("Calling Opus to design new strategy...")
        opus_output = self._call_opus(prompt)
        
        # VALIDATE: Check if Opus returned anything
        if not opus_output or len(opus_output.strip()) < 100:
            print(f"ERROR: Opus returned insufficient output ({len(opus_output)} chars)")
            print(f"Output preview: {opus_output[:200]}")
            return None
        
        # Extract code
        code = self._extract_code(opus_output)
        
        # VALIDATE: Check if code extraction succeeded
        if not code or len(code) < 100:
            print(f"ERROR: Code extraction failed. Opus output length: {len(opus_output)}")
            print(f"Opus output:\n{opus_output}")
            return None
        
        # VALIDATE: Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            print(f"ERROR: Generated code has syntax errors: {e}")
            print(f"Code:\n{code}")
            return None
        
        # Save and register
        strategy_num = self._get_next_strategy_number()
        filepath = self._save_strategy(code, strategy_num)
        
        # VALIDATE: File was written
        if not filepath.exists() or filepath.stat().st_size == 0:
            print(f"ERROR: Strategy file was not written or is empty: {filepath}")
            return None
        
        print(f"✓ Strategy saved: {filepath} ({filepath.stat().st_size} bytes)")
        
        # [rest of existing code]
        
    except Exception as e:
        print(f"Strategy design failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        con.close()
```

**Alternative: Clean up empty files**

Add a post-processing step to the orchestrator:

```python
def _cleanup_empty_strategies(self):
    """Remove empty strategy files that failed to generate."""
    strategies_dir = self.workspace_dir / 'strategies'
    for py_file in strategies_dir.glob('strategy_*.py'):
        if py_file.stat().st_size == 0:
            print(f"Removing empty strategy file: {py_file}")
            py_file.unlink()
```

---

## Issue #3: Feature Manager NaN Errors

### Symptoms
```
[WARNING] Feature manager failed (Cannot convert float NaN to integer), using synthetic data
```

### Root Cause
**File:** `features/feature_manager.py`, Line 71-83

```python
def _load_ohlcv_from_ticks(self, symbol: str, timeframe: str = '1m', 
                            lookback_bars: int = 500) -> pd.DataFrame:
    # [query ticks from database]
    
    # Clean data: drop rows with NaN price, fill NaN volume with 0
    df = df.dropna(subset=['price'])
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')  # ← ERROR HERE!
```

**Problem:** `ts_ms` column may contain NaN values after cleaning, which `pd.to_datetime()` cannot convert to integers.

### Stack Trace
```
File "features/feature_manager.py", line 81, in _load_ohlcv_from_ticks
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
File "pandas/core/tools/datetimes.py", line 1067, in to_datetime
    values = convert_listlike(arg._values, format)
File "pandas/core/tools/datetimes.py", line 433, in _convert_listlike_datetimes
    return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)
File "pandas/core/tools/datetimes.py", line 467, in _array_strptime_with_fallback
    result, timezones = array_strptime(arg, format, exact=exact, errors=errors)
File "pandas/_libs/tslibs/strptime.pyx", line 530, in pandas._libs.tslibs.strptime.array_strptime
ValueError: Cannot convert float NaN to integer
```

### Proposed Fix

**Clean ts_ms before datetime conversion:**

```python
def _load_ohlcv_from_ticks(self, symbol: str, timeframe: str = '1m', 
                            lookback_bars: int = 500) -> pd.DataFrame:
    # [load ticks from database]
    
    if df.empty:
        raise ValueError(f"No tick data found for {symbol} in the specified time range")
    
    # Clean data BEFORE any conversions
    # 1. Drop rows with NaN in critical columns
    df = df.dropna(subset=['ts_ms', 'price'])
    
    # 2. Convert types explicitly
    df['ts_ms'] = pd.to_numeric(df['ts_ms'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    
    # 3. Drop any rows that failed conversion
    df = df.dropna(subset=['ts_ms', 'price'])
    
    # 4. Ensure ts_ms is integer type
    df['ts_ms'] = df['ts_ms'].astype('int64')
    
    # NOW safe to convert timestamp
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # [rest of existing code]
```

**Alternative: Add try-catch with better error message:**

```python
try:
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
except ValueError as e:
    # Log diagnostic info
    print(f"ERROR: Cannot convert ts_ms to datetime")
    print(f"  ts_ms dtype: {df['ts_ms'].dtype}")
    print(f"  ts_ms null count: {df['ts_ms'].isna().sum()}")
    print(f"  ts_ms sample: {df['ts_ms'].head()}")
    raise ValueError(f"Timestamp conversion failed: {e}")
```

---

## Issue #4: Ranker Issues - 0 Ensembles and 0 Top Models

### Symptoms
```
"top_models_count": 0
"top_ensembles_count": 0
```

Despite:
```
"models_trained": 5
"db_rows_saved": 5
```

### Evidence from Database
```
ML Models (archived=0): 40

Recent ML models:
  ('direction_predictor', 'classification', None, None, 0)
  ('price_predictor', 'regression', None, None, 0)
  ('volatility_regressor', 'regression', None, None, 0)
  ('risk_scorer', 'regression', None, None, 0)
  ('momentum_classifier', 'classification', None, None, 0)
```

**All test_accuracy and f1_score are NULL!**

### Root Cause
**File:** `ml_pipeline/db_connector.py`, Line 48-54

```python
# Extract metrics (handle both classification and regression)
train_accuracy = metrics.get("train_accuracy", metrics.get("train_r2"))
test_accuracy = metrics.get("test_accuracy", metrics.get("test_r2"))
f1_score = metrics.get("f1_score")
precision = metrics.get("precision")
recall = metrics.get("recall")
roc_auc = metrics.get("roc_auc")
```

**BUT** the model training code returns different keys:

**File:** `ml_pipeline/models/direction_predictor.py`, Line 96-99

```python
metrics = {
    "train_accuracy": train_acc,
    "val_accuracy": val_acc,  # ← NOT "test_accuracy"!
    "feature_importance": feature_importance,
    # No f1_score, precision, recall, or roc_auc!
}
```

**Mismatch:** 
- Model returns: `val_accuracy`
- DB expects: `test_accuracy`
- Result: NULL in database

**File:** `orchestration/ranker.py`, Line 66-72

```python
query = f'''
    SELECT model_name, model_type, /* ... */
    FROM ml_model_results
    WHERE archived = 0 AND {metric} IS NOT NULL  # ← Filters out ALL models!
    GROUP BY model_name
    ORDER BY metric_value DESC
'''
```

### Proposed Fix

**Option A: Standardize metric names in models (preferred)**

Update all model `train()` methods to return standard keys:

```python
# In direction_predictor.py, risk_scorer.py, etc.
def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
    # [training code]
    
    # Calculate additional metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    y_pred_train = self.model.predict(X_train)
    y_pred_val = self.model.predict(X_val)
    
    metrics = {
        # Standardized names
        "train_accuracy": train_acc,
        "test_accuracy": val_acc,  # Rename val_accuracy → test_accuracy
        "f1_score": f1_score(y_val, y_pred_val, average='binary'),
        "precision": precision_score(y_val, y_pred_val, average='binary'),
        "recall": recall_score(y_val, y_pred_val, average='binary'),
        # "roc_auc": roc_auc_score(y_val, y_pred_proba),  # If predict_proba available
        "feature_importance": feature_importance,
    }
    
    return metrics
```

**Option B: Update db_connector to handle both naming conventions**

```python
# In db_connector.py
def save_training_result(self, result: Dict[str, Any]) -> int:
    metrics = result.get("metrics", {})
    
    # Handle multiple naming conventions
    train_accuracy = (
        metrics.get("train_accuracy") or 
        metrics.get("train_r2") or 
        metrics.get("train_score")
    )
    
    test_accuracy = (
        metrics.get("test_accuracy") or 
        metrics.get("val_accuracy") or  # ← Add this!
        metrics.get("test_r2") or 
        metrics.get("val_r2")
    )
    
    f1_score = metrics.get("f1_score") or metrics.get("f1")
    # ... etc
```

**Option C: Update ranker query to use metrics_json**

```python
# In ranker.py
def keep_top_models(self, count: int = 5, metric: str = 'f1_score') -> List[Dict[str, Any]]:
    # Parse metric from JSON instead of relying on dedicated columns
    query = f'''
        SELECT 
            model_name,
            model_type,
            MAX(ts_ms) as latest_ts,
            json_extract(metrics_json, '$.{metric}') as metric_value,
            json_extract(metrics_json, '$.train_accuracy') as train_accuracy,
            json_extract(metrics_json, '$.val_accuracy') as test_accuracy,
            f1_score,
            test_accuracy as test_acc_col
        FROM ml_model_results
        WHERE archived = 0 
          AND metrics_json IS NOT NULL
          AND json_extract(metrics_json, '$.{metric}') IS NOT NULL
        GROUP BY model_name
        ORDER BY metric_value DESC
    '''
```

---

## Priority Recommendations

### Immediate (Critical)
1. **Fix metric key mismatch** (Issue #4) - Blocks ranker entirely
2. **Add validation to strategy designer** (Issue #2) - Prevents empty files
3. **Fix feature manager NaN** (Issue #3) - Forces synthetic data fallback

### High Priority
4. **Fix Sonnet/Opus subprocess calls** (Issues #1 & #2) - Core functionality
5. **Add comprehensive error logging** - Better debugging

### Optional Enhancements
6. **Add cleanup script** for empty strategy files
7. **Add database migration** to backfill metrics from metrics_json
8. **Add integration tests** for each component

---

## Testing Plan

After fixes are applied:

1. **Test Feature Manager:**
   ```bash
   cd /home/rob/.openclaw/workspace/blofin-stack
   python3 -c "
   from features.feature_manager import FeatureManager
   fm = FeatureManager()
   df = fm.get_features('BTC-USDT', '5m', lookback_bars=200)
   print(f'Success: {df.shape}')
   assert df.isna().sum().sum() == 0, 'NaN values present!'
   "
   ```

2. **Test ML Training + DB Connector:**
   ```bash
   python3 -c "
   from ml_pipeline.train import TrainingPipeline
   from ml_pipeline.db_connector import MLDatabaseConnector
   from features.feature_manager import FeatureManager
   
   fm = FeatureManager()
   df = fm.get_features('BTC-USDT', '1m', lookback_bars=1000)
   
   pipeline = TrainingPipeline()
   results = pipeline.train_all_models(df)
   
   db = MLDatabaseConnector('data/blofin_monitor.db')
   rows = db.save_all_results(results)
   
   print(f'Saved {len(rows)} models')
   
   # Verify metrics
   latest = db.get_latest_results(5)
   for row in latest:
       assert row['test_accuracy'] is not None, f'NULL test_accuracy for {row['model_name']}'
       print(f'{row['model_name']}: test_acc={row['test_accuracy']:.4f}')
   "
   ```

3. **Test Ranker:**
   ```bash
   python3 -c "
   from orchestration.ranker import Ranker
   
   ranker = Ranker('data/blofin_monitor.db')
   top_models = ranker.keep_top_models(count=5)
   
   print(f'Top models: {len(top_models)}')
   assert len(top_models) > 0, 'Ranker returned 0 models!'
   
   for m in top_models:
       print(f'{m['rank']}. {m['model_name']}: {m['metric_value']:.4f}')
   "
   ```

4. **Test Strategy Designer:**
   ```bash
   # Manual test (requires Opus API)
   python3 orchestration/strategy_designer.py
   
   # Check that new strategy file is NOT empty
   newest=$(ls -t strategies/strategy_*.py | head -1)
   size=$(wc -c < "$newest")
   echo "Newest strategy: $newest ($size bytes)"
   [ $size -gt 100 ] || echo "ERROR: Strategy file too small!"
   ```

---

## Files to Modify

1. `ml_pipeline/models/direction_predictor.py` - Add f1/precision/recall, rename val_accuracy
2. `ml_pipeline/models/risk_scorer.py` - Same
3. `ml_pipeline/models/price_predictor.py` - Same
4. `ml_pipeline/models/momentum_classifier.py` - Same
5. `ml_pipeline/models/volatility_regressor.py` - Same
6. `features/feature_manager.py` - Fix NaN handling in _load_ohlcv_from_ticks
7. `orchestration/strategy_designer.py` - Add validation for empty code
8. `orchestration/strategy_tuner.py` - Fix subprocess call / add logging

---

## Estimated Time to Fix

- **Issue #4 (Ranker):** 30 minutes - Update 5 model files
- **Issue #3 (Feature Manager):** 15 minutes - Add NaN handling
- **Issue #2 (Strategy Designer):** 20 minutes - Add validation
- **Issue #1 (Strategy Tuner):** 20 minutes - Same fix as designer

**Total:** ~1.5 hours

---

**End of Report**
