# Plan: Task #2 - Create strategy_registry and implement ghost strategies

## Overview

Create the `strategy_registry` table, migrate existing data, implement 3 ghost strategies as
real .py files, and add promotion/demotion logic.

## Current State

- Database: `data/blofin_monitor.db`
- Strategies directory has ~50 .py files
- `strategy_scores` table has paper-trade data for 10 real strategies
- 3 ghost strategies exist in `strategy_scores` but have no .py files:
  - `ml_gbt_5m`
  - `mtf_momentum_confirm`
  - `mtf_trend_align`
- All strategies follow the `BaseStrategy` ABC pattern with a `detect()` method

## Steps

### Step 1: Create `strategy_registry` table
- Run the CREATE TABLE SQL from the task spec in `data/blofin_monitor.db`
- Use `IF NOT EXISTS` to be idempotent

### Step 2: Migrate existing strategies into registry
- Scan `strategies/*.py` (excluding `base_strategy.py`, `__init__.py`, `__pycache__`)
- For each .py file, insert a row with `file_path` set, `tier=2` if the strategy has data in
  `strategy_scores`, else `tier=0`
- Pull `ft_win_rate`, `ft_sharpe`, `ft_pnl_pct`, `ft_max_dd`, `ft_trades` from `strategy_scores`
  (window='all', symbol IS NULL) for strategies that have paper trade data
- Strategies in strategy_scores with no .py file: insert as `archived=1`, `archive_reason='no_file'`

### Step 3: Implement 3 ghost strategies as real .py files

**`strategies/mtf_trend_align.py`**
- Class: `MTFTrendAlignStrategy`, name: `mtf_trend_align`
- 5m RSI + 1h EMA trend filter
- Use 5-minute window (300s) for RSI, 1-hour window (3600s) for EMA trend
- BUY when RSI oversold AND price is above 1h EMA (uptrend)
- SELL when RSI overbought AND price is below 1h EMA (downtrend)

**`strategies/ml_gbt_5m.py`**
- Class: `MLGbt5mStrategy`, name: `ml_gbt_5m`
- Pure ML entry strategy using GBT model (gradient boosted tree on 5m features)
- Features: 5m returns, RSI, Bollinger band position, volume ratio
- Loads model from file if available, falls back to heuristic scoring
- Since actual GBT model may not exist, implement a feature-based heuristic that mimics
  what a GBT would learn (multi-factor score)

**`strategies/mtf_momentum_confirm.py`**
- Class: `MTFMomentumConfirmStrategy`, name: `mtf_momentum_confirm`
- 15m momentum signal confirmed by 4h trend
- Use 15-minute window (900s) for momentum signal, 4-hour window (14400s) for trend filter
- BUY when 15m momentum is up AND 4h trend is positive (price > 4h EMA)
- SELL when 15m momentum is down AND 4h trend is negative (price < 4h EMA)

### Step 4: Register the 3 new ghost strategies in the registry
- Insert them at `tier=0` (library) with `source='librarian'`

### Step 5: Add promotion/demotion logic as a standalone module
- Create `strategies/strategy_promoter.py` with:
  - `check_promotion(strategy_name, registry_row, bt_metrics, ft_metrics) -> (bool, str)`
  - `check_demotion(strategy_name, registry_row, ft_metrics) -> (bool, str)`
  - Gates as specified in task:
    - Tier 0→1: file exists, imports cleanly, can generate ≥1 signal
    - Tier 1→2: ≥50 bt trades, WR ≥40%, Sharpe ≥0.5, max DD ≤30%, EEP ≥50, PF ≥1.1
    - Tier 2→3: ≥100 ft trades, WR ≥45%, Sharpe ≥1.0, convergence <20%, EEP ≥65, PF ≥1.3, 14+ days

### Step 6: Create migration script
- Create `migrate_strategy_registry.py` as a one-time migration script
- This is what actually executes steps 1, 2, and 4
- Run it to populate the DB

### Step 7: Register new strategies in `strategies/__init__.py`
- Import and add the 3 new strategy classes to `get_all_strategies()`

## Files to Create/Modify
- `data/blofin_monitor.db` - add strategy_registry table (via migration script)
- `strategies/mtf_trend_align.py` - new ghost strategy
- `strategies/ml_gbt_5m.py` - new ghost strategy
- `strategies/mtf_momentum_confirm.py` - new ghost strategy
- `strategies/strategy_promoter.py` - promotion/demotion logic module
- `migrate_strategy_registry.py` - one-time migration script
- `strategies/__init__.py` - add new strategy imports

## No Regressions
- All existing strategy files remain unchanged
- DB migration uses INSERT OR IGNORE / INSERT OR REPLACE to be safe
