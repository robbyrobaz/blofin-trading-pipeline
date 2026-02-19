# Plan: Task #4 - Fix Dashboard to Use strategy_registry Tiers and ML Metrics

## Current State

- Dashboard HTML: `/home/rob/.openclaw/workspace/blofin-dashboard/blofin-dashboard.html`
- Dashboard Flask server: `/home/rob/.openclaw/workspace/blofin-dashboard/server.py` (port 8888)
- Dashboard API calls: `/api/status`, `/api/live-data`, `/api/strategies`, `/api/models`,
  `/api/reports`, `/api/advanced_metrics`, `/api/backtest-comparison`
- `strategy_registry` table does NOT exist yet (being created by Task #2 teammate)

## Problems to Fix

1. **Top metrics / strategy table**: Shows all strategies averaged, no tier info
2. **Strategy table**: No tier column, no color coding, no ghost strategy filtering
3. **ML section**: Shows 100% accuracy numbers from data leakage (direction_predictor = 1.0)
4. **Advanced Trading Metrics**: Uses all strategies, not Tier 2 (forward-test) strategies
5. **Ghost strategies**: Strategies in DB but no .py file shown in dashboard

## Ghost Strategies (strategies in DB with no .py file)

From analysis:
- `ml_gbt_5m` - in strategy_scores, no standalone .py file exists yet
- `mtf_momentum_confirm` - in strategy_scores, no .py file
- `mtf_trend_align` - in strategy_scores, no .py file

Strategies with .py files (real): bb_squeeze, breakout, candle_patterns, ema_crossover,
momentum, reversal, rsi_divergence, support_resistance, volume_mean_reversion, vwap_reversion

## Plan

### Step 1: Add `/api/registry` endpoint to `server.py`

Add a new endpoint that:
- If `strategy_registry` table exists: JOIN with strategy_scores + strategy_backtest_results,
  return tier, bt_* and ft_* metrics per strategy
- If `strategy_registry` NOT yet created (blocking Task #2): fall back to strategy_scores
  data but filter ghost strategies (those without a .py file) by checking a hardcoded list
  or filesystem check
- Returns: `{strategies: [{name, tier, has_file, bt_win_rate, bt_sharpe, bt_pnl, bt_max_dd,
  ft_win_rate, ft_sharpe, ft_pnl, ft_max_dd, ft_trades, convergence_pct}]}`
- Includes Tier 2 aggregate metrics for top-metrics section

### Step 2: Fix ML section data in `/api/models` endpoint in `server.py`

The `direction_predictor` shows 100% accuracy (data leakage). Fix:
- Flag models with train_accuracy > 0.95 AND test_accuracy > 0.95 as "leakage suspected"
- Show out-of-sample accuracy (test_accuracy) as primary metric
- Add `leakage_flag` field when train_acc minus test_acc < 2% AND both > 95%
- Show a note in the dashboard when leakage is flagged
- If test_accuracy is genuinely available and != train_accuracy, prefer test

### Step 3: Update `/api/strategies` endpoint to filter ghost strategies

In `server.py`'s `api_strategies`:
- Get list of .py files from strategies directory
- Filter strategy_scores rows to only include strategies with existing .py files
- If strategy_registry is available, use tier from there

### Step 4: Update Dashboard HTML

#### 4a. Strategies section: Add tier column + color coding
- Add a "Tier" column to strategy cards
- Tier 2 (forward-test): green left border
- Tier 1 (backtest only): blue left border
- Tier 0 (library): gray left border
- Data source priority: strategy_registry tier if available, else infer from strategy_scores

#### 4b. Top metrics: Show Tier 2 averages
- Change "Performance Metrics" card to show only Tier 2 strategy averages
- Label it "Forward-Test Strategies (Tier 2)"
- Fallback: if no tier data, show strategies with paper trade data in strategy_scores

#### 4c. ML section: Show honest metrics
- Add warning badge when leakage is detected (accuracy > 95% train AND test match)
- Show "OOS Acc" (out-of-sample / test accuracy) as primary metric
- For known-leaky models (both train=1.0, test=1.0): show "Accuracy: N/A (leakage)" in red

#### 4d. Advanced Trading Metrics: Use Tier 2 data
- Pull from `/api/advanced_metrics` but use registry-filtered strategies
- Add `tier_filter` to the endpoint response (already data from Tier 2 strategies)

#### 4e. Ghost strategy removal
- Dashboard JS should skip rendering strategies flagged as `has_file: false`
- Or better: server.py filters them before returning

### Step 5: Graceful degradation

All changes must work BEFORE strategy_registry exists:
- `tier` defaults to:
  - tier=2 if strategy has paper trades AND win_rate not 100% (non-synthetic)
  - tier=1 if strategy has backtest data only (strategy_backtest_results)
  - tier=0 otherwise
- `has_file`: check filesystem at server startup, cache result
- Ghost filter: hardcode known ghost names as fallback

## Files to Modify

1. `/home/rob/.openclaw/workspace/blofin-dashboard/server.py`
   - Add `/api/registry` endpoint
   - Fix `/api/models` to flag leakage
   - Fix `/api/strategies` to filter ghosts
   - Update `/api/advanced_metrics` to prefer Tier 2 data

2. `/home/rob/.openclaw/workspace/blofin-dashboard/blofin-dashboard.html`
   - Add tier column + color coding to strategy cards
   - Fix ML section to show honest metrics with leakage warnings
   - Update top metrics to show Tier 2 averages
   - Update `updateStrategies()` to use `/api/registry`
   - Update `updateModels()` to show leakage warnings

## No New Files

All changes go into existing `server.py` and `blofin-dashboard.html`.

## Testing

After implementation, verify:
- [ ] Strategy cards show tier badges (T2 green, T1 blue, T0 gray)
- [ ] Ghost strategies not shown in dashboard
- [ ] ML section shows "Data Leakage Suspected" for 100% accuracy models
- [ ] Top metrics section shows Tier 2 averages (with fallback if registry missing)
- [ ] Works without strategy_registry table (graceful fallback)
