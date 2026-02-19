# Phase 2 ML Retrain Framework — Deployment Guide

## Overview

The Phase 2 ML Retrain Framework is a complete system for continuous model retraining and swapping based on live paper-trading performance.

### Architecture

```
Paper Trading (live, 24/7)
    ↓
[2-week + 75 trades?] → Gate Check Daily @ 06:00 MST
    ↓
[Gates pass?] → ML Retrain
    ├─ Regime diversity (volatility percentile 20th–80th)
    ├─ Sample ticks, build OHLCV candles
    ├─ Engineer features
    ├─ Walk-forward split (24h embargo)
    └─ Train 5 models (direction, risk, price, momentum, volatility)
    ↓
New Models Archived
    ├─ v1_backtest (baseline, kept for reference)
    └─ v2_paper (new, goes into A/B testing)
    ↓
[Daily @ 18:00 MST] → A/B Evaluation
    ├─ Compare Arm A (v1) vs Arm B (v2) on live trades
    ├─ Min 100 trades per arm
    ├─ Swap if: B Sharpe ≥ A Sharpe AND B expectancy > 0
    └─ Log decision to database
    ↓
Active Model Updated (or kept)
```

## Files & Components

### Core ML Modules

| File | Purpose |
|------|---------|
| `ml_retrain_phase2.py` | Gate checks, feature engineering, model training |
| `ab_test_models.py` | A/B evaluation, model swap logic |
| `execution_calibrator_v2.py` | EMA-weighted slippage, time-of-day & regime tracking |
| `execution_calibrator.py` | Legacy v1 (kept for backward-compat) |

### Cron Scripts & Scheduling

| File | Time | Purpose |
|------|------|---------|
| `cron_phase2_check.sh` | 06:00 MST (13:00 UTC) | Trigger Phase 2 retrain if gates pass |
| `cron_ab_evaluate.sh` | 18:00 MST (01:00 UTC next day) | A/B evaluation & model swap decision |

### Systemd Service Files

For persistent, reliable cron-like scheduling on Linux:

```bash
# Phase 2 Retrain (06:00 MST)
blofin-stack-phase2-retrain.service
blofin-stack-phase2-retrain.timer

# A/B Evaluation (18:00 MST)
blofin-stack-phase2-ab-eval.service
blofin-stack-phase2-ab-eval.timer
```

### Testing & Verification

| File | Purpose |
|------|---------|
| `smoke_test_phase2.py` | Comprehensive smoke test (6 checks) |
| `logs/phase2_retrain.log` | Main log file (shared with A/B) |
| `logs/smoke_test_phase2_results.log` | Smoke test results |

### Databases & Artifacts

| Location | Purpose |
|----------|---------|
| `data/blofin_monitor.db` | Main DB (contains paper_trades, phase2_*, ab_*) |
| `data/models/` | Saved v2_paper model artifacts |
| `data/execution_calibration.json` | Live execution calibration (read by backtester) |

## Installation & Deployment

### Prerequisites

- Python 3.10+
- Virtual environment activated: `.venv/bin/python3`
- Database initialized with paper_trades, ticks tables
- Systemd (for service scheduling) OR manual cron

### Step 1: Verify Code

```bash
cd /home/rob/.openclaw/workspace/blofin-stack

# Run smoke test
.venv/bin/python3 smoke_test_phase2.py
```

Expected output: **6/6 checks passed** ✅

### Step 2: Install Systemd Timers

Copy service & timer files to systemd user directory:

```bash
# Install to user timers (no sudo needed)
mkdir -p ~/.config/systemd/user
cp blofin-stack-phase2-*.service ~/.config/systemd/user/
cp blofin-stack-phase2-*.timer ~/.config/systemd/user/

# Reload systemd configuration
systemctl --user daemon-reload

# Enable timers (will run on boot)
systemctl --user enable blofin-stack-phase2-retrain.timer
systemctl --user enable blofin-stack-phase2-ab-eval.timer

# Start timers immediately
systemctl --user start blofin-stack-phase2-retrain.timer
systemctl --user start blofin-stack-phase2-ab-eval.timer

# Verify status
systemctl --user status blofin-stack-phase2-*.timer
systemctl --user list-timers
```

### Step 3: (Optional) Manual Cron Setup

If NOT using systemd, add to crontab:

```bash
# Edit user crontab
crontab -e

# Add these lines (adjust timezone as needed):
# Phase 2 Retrain Trigger Check — Daily @ 06:00 MST (13:00 UTC)
0 13 * * * /home/rob/.openclaw/workspace/blofin-stack/cron_phase2_check.sh

# A/B Model Evaluation — Daily @ 18:00 MST (01:00 UTC next day)
0 1 * * * /home/rob/.openclaw/workspace/blofin-stack/cron_ab_evaluate.sh
```

### Step 4: Monitor Logs

Watch the Phase 2 log file:

```bash
tail -f logs/phase2_retrain.log
```

### Step 5: Manual Testing

Test individual components:

```bash
# Check gates (dry-run, no training)
.venv/bin/python3 ml_retrain_phase2.py --dry-run

# Force a retrain (skip gate checks)
.venv/bin/python3 ml_retrain_phase2.py --force

# Smoke test (relaxed gates: 7d / 10 trades)
.venv/bin/python3 ml_retrain_phase2.py --smoke-test

# Check A/B status
.venv/bin/python3 ab_test_models.py --status

# Manually run A/B evaluation (dry-run)
.venv/bin/python3 ab_test_models.py --dry-run

# Run execution calibrator
.venv/bin/python3 execution_calibrator_v2.py
```

## Gate Logic

### Paper Age Gate

**Trigger:** Paper trading period ≥ 14 days

```
first_trade_opened_ts ... last_trade_closed_ts
|————————— duration ————————|
if duration ≥ 14 days → PASS ✅
```

### Trade Count Gate

**Trigger:** ≥ 75 closed paper trades

```
SELECT COUNT(*) FROM paper_trades WHERE status='CLOSED'
if count ≥ 75 → PASS ✅
```

### Regime Diversity Gate

**Trigger:** Volatility must span from ≤ 20th to ≥ 80th percentile

Samples ticks efficiently (last 10 days) → builds 5-min candles → computes HL range/close ratio → checks percentile distribution.

```
HL_ranges: [min_val, ... p20, ... p80, ... max_val]
if min_val ≤ p20 AND max_val ≥ p80 → PASS ✅ (diverse regime)
else → FAIL ❌ (regime too narrow)
```

## Model Training Details

### Feature Engineering

11-feature pipeline:
1. Mean return (20-period)
2. Volatility (σ)
3. Last return
4. 5-period mean return
5. EMA(9) deviation
6. EMA(21) deviation
7. RSI(14) normalized
8. Bollinger Band position
9. Volume spike ratio
10. HL range %
11. Range position

### Walk-Forward Split

```
Train (70%) | Embargo (5-10%, ≥24h) | Test (remainder)
```

The embargo ensures no look-ahead bias between training and test periods. Actual embargo duration reported in logs.

### Models Trained

All models are **GradientBoostingClassifier/Regressor** with:
- `n_estimators=100`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`

| Model | Type | Task |
|-------|------|------|
| `direction_predictor` | Classification | Up/down in next 5 candles |
| `risk_scorer` | Classification | Risk assessment |
| `price_predictor` | Regression | Price direction (continuous) |
| `momentum_classifier` | Classification | Momentum direction |
| `volatility_regressor` | Regression | Volatility forecast |

### Slippage Multiplier

Conservatively scales from **2.0× → 1.5×** as paper trades accumulate:

```python
mult = 2.0 - (2.0 - 1.5) * min(num_trades / 500, 1.0)
```

- At 0 trades: `2.0×` (very conservative)
- At 500+ trades: `1.5×` (less conservative)

## A/B Testing Logic

### Arm Assignment

- **Arm A (v1_backtest):** Trades BEFORE most recent Phase 2 retrain
- **Arm B (v2_paper):** Trades AFTER most recent Phase 2 retrain

### Decision Rules

```
if (Arm A trades < 100) OR (Arm B trades < 100):
    → Decision: "Wait for more data"
    → No swap
elif (Arm B Sharpe < Arm A Sharpe) OR (Arm B expectancy ≤ 0):
    → Decision: "Old model better or new model losing"
    → No swap
else:
    → Decision: "New model passes both gates"
    → SWAP to v2_paper ✅
```

### Metrics

- **Sharpe Ratio:** Mean PnL / Std Dev PnL (higher is better)
- **Expectancy:** Mean PnL per trade (must be > 0)
- **Win Rate:** % of profitable trades
- **Profit Factor:** Gross profit / Gross loss

## Execution Calibration (v2)

### EMA Weighting

Recent 50 trades weighted 3× vs older trades:

```python
# For each trade (oldest → newest):
if trade_index >= (total_trades - 50):
    alpha = 0.15 * 3.0  # Boosted
else:
    alpha = 0.15        # Standard

ema_next = alpha * slip + (1 - alpha) * ema_prev
```

### Time-of-Day Buckets (UTC)

| Bucket | Hours | Slippage | Pos Mult |
|--------|-------|----------|----------|
| 00-06 | 0-6h | EMA % | Derived |
| 06-12 | 6-12h | EMA % | Derived |
| 12-18 | 12-18h | EMA % | Derived |
| 18-24 | 18-24h | EMA % | Derived |

### Volatility Regimes

| Regime | |pnl|% | Slippage | Pos Mult |
|--------|---------|----------|----------|
| Ranging | < 0.3% | EMA % | Derived |
| Trending | 0.3–2.0% | EMA % | Derived |
| Volatile | ≥ 2.0% | EMA % | Derived |

Each bucket/regime outputs:
- EMA slippage/side
- Position size multiplier (0.5–1.2×)
- Trade count in that category

## Output Files

### Models

```
data/models/
├── direction_predictor_v2_paper/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── risk_scorer_v2_paper/
├── price_predictor_v2_paper/
├── momentum_classifier_v2_paper/
├── volatility_regressor_v2_paper/
└── [*_v2_paper_prev/]  ← Previous backup
```

### Calibration

```json
data/execution_calibration.json
{
  "generated_at": "2026-02-17T18:19:32Z",
  "trade_count": 35719,
  "win_rate": 0.520,
  "profit_factor": 1.234,
  "execution": {
    "ema_slippage_per_side_pct": 0.0600,
    "position_size_multiplier": 1.000,
    "recent_n_weighted": 50,
    "recent_weight_mult": 3.0
  },
  "time_of_day": {
    "00-06": { "count": 8929, "ema_slip": 0.0610, "pos_mult": 0.998 },
    "06-12": { "count": 8945, "ema_slip": 0.0590, "pos_mult": 1.003 },
    "12-18": { "count": 8923, "ema_slip": 0.0605, "pos_mult": 0.999 },
    "18-24": { "count": 8922, "ema_slip": 0.0595, "pos_mult": 1.004 }
  },
  "regime_slippage": {
    "ranging": { "count": 11800, "ema_slip": 0.0550, "pos_mult": 1.012 },
    "trending": { "count": 18500, "ema_slip": 0.0620, "pos_mult": 0.988 },
    "volatile": { "count": 5419, "ema_slip": 0.0680, "pos_mult": 0.963 }
  },
  "recommendations": [
    "✅ EMA slippage (0.0600%/side) better than assumed (0.0200%)",
    "✅ Profit factor 1.234 is healthy.",
    "✅ Execution parameters within normal range."
  ]
}
```

### Logs

```
logs/phase2_retrain.log          ← Shared by both cron jobs
logs/smoke_test_phase2_results.log ← Smoke test output
```

## Troubleshooting

### "Gates not met" on first run

**Expected.** You need:
- ≥ 14 days of paper trading
- ≥ 75 closed trades
- Volatility spanning 20th–80th percentile

Check gate details:

```bash
.venv/bin/python3 ml_retrain_phase2.py --dry-run
```

### "Insufficient feature rows" or "Insufficient OHLCV candles"

The tick-sampling strategy may need tuning if you have very sparse tick data. Adjust `TICK_SAMPLE_SIZE` in `ml_retrain_phase2.py` if needed (default: 10,000).

### Systemd timer not running

Check status:

```bash
systemctl --user status blofin-stack-phase2-retrain.timer
journalctl --user -u blofin-stack-phase2-retrain --follow
```

Manually trigger:

```bash
systemctl --user start blofin-stack-phase2-retrain.service
```

### A/B evaluation says "not ready"

You need ≥ 100 closed trades in EACH arm:

```bash
.venv/bin/python3 ab_test_models.py --status
```

Shows current state.

## Performance & Tuning

### Timing

- **Gate checks:** ~2–5 seconds (fast tick sampling)
- **Feature building:** ~10–30 seconds (depends on OHLCV size)
- **Model training (5 models):** ~30–60 seconds
- **A/B evaluation:** ~5–10 seconds
- **Total per retrain:** ~1–3 minutes

### Optimization Hints

1. **Reduce model complexity:** Lower `n_estimators` in `train_single_model()`
2. **Adjust embargo:** Modify `embargo_pct` in `walk_forward_split()`
3. **Tune slippage multiplier:** Adjust `INITIAL_SLIPPAGE_MULT` and `MIN_SLIPPAGE_MULT`
4. **Control EMA recency:** Change `RECENT_N` and `RECENT_WEIGHT_MULT` in execution_calibrator_v2.py

## Monitoring Checklist

Daily:
- [ ] Check logs: `tail -f logs/phase2_retrain.log`
- [ ] Verify model swap decision: `ab_test_models.py --status`
- [ ] Inspect execution calibration: `execution_calibrator_v2.py` output

Weekly:
- [ ] Review database: query `phase2_retrain_runs`, `ab_test_runs`
- [ ] Check model accuracy trends
- [ ] Audit gate logs: `phase2_gate_log` table

Monthly:
- [ ] Smoke test: `smoke_test_phase2.py`
- [ ] Review A/B history: are swaps improving performance?
- [ ] Adjust gates/parameters if needed

## Integration with Existing Systems

### Paper Engine

No changes needed. Paper engine continues to run live trades using the active model selected by A/B testing.

### Backtester

Reads `execution_calibration.json` for realistic slippage/friction. Position multipliers from TOD & regime buckets are available for strategy optimization.

### Strategy Manager

No direct dependency on Phase 2. Phase 2 retrains ML models; strategy generation is separate.

### Dashboard

A/B test results and execution calibration are logged to database. Dashboard can query these tables for visualization.

## Success Criteria

Phase 2 is **successfully deployed** when:

1. ✅ Smoke test passes (6/6 checks)
2. ✅ Systemd timers running (or cron jobs active)
3. ✅ Daily logs accumulating without errors
4. ✅ Phase 2 retrain triggered after gates pass
5. ✅ A/B models compared 12 hours later
6. ✅ Model swap executes (if conditions met)
7. ✅ Execution calibration updated daily
8. ✅ No spike in error logs

## Rollback

If issues arise:

```bash
# Disable timers
systemctl --user disable blofin-stack-phase2-retrain.timer
systemctl --user disable blofin-stack-phase2-ab-eval.timer

# Or remove cron lines:
crontab -e  # Delete two Phase 2 lines

# Revert to last good model
# (models are backed up in *_prev directories)
```

## Support & Debugging

For detailed logs:

```bash
# Phase 2 retrain log
logs/phase2_retrain.log

# Full debug output (last 50 runs):
sqlite3 data/blofin_monitor.db \
  "SELECT ts_iso, gate_pass, error_msg FROM phase2_retrain_runs ORDER BY ts_ms DESC LIMIT 50;"

# A/B test history:
sqlite3 data/blofin_monitor.db \
  "SELECT ts_iso, arm_a_trades, arm_b_trades, swap_recommended, swap_executed \
   FROM ab_test_runs ORDER BY ts_ms DESC LIMIT 20;"
```

---

**Deployment Date:** 2026-02-17
**Status:** ✅ Ready for production
**Last Smoke Test:** All 6/6 checks passed
