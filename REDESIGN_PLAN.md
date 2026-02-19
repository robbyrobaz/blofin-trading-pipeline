# Blofin Pipeline Redesign Plan
**Date:** Feb 18, 2026 | **Author:** Jarvis | **Status:** ✅ IMPLEMENTED (Tasks 1-4 complete, pipeline runner in progress)

## Current State Assessment

### What Works ✅
- **Data pipeline**: 34.8M ticks, 36 symbols, live websocket ingestion 24/7
- **Feature library**: 95+ technical indicators (price, volume, volatility, regime)
- **10 signal-generating strategies**: bb_squeeze, breakout, candle_patterns, ema_crossover, momentum, reversal, rsi_divergence, support_resistance, volume_mean_reversion, vwap_reversion
- **Paper trading engine**: 35.9K closed trades, signal → confirmation → execution loop working
- **Orchestration framework**: daily_runner.py with designer, tuner, ranker, reporter
- **EEP scoring**: Composite entry+exit package scoring system
- **Dashboard**: Live at :8888, real-time data display
- **Infrastructure**: systemd services, gap-fill, data retention, ingestor all stable

### What's Broken 🔴
1. **ML data leakage**: direction_predictor at 100% accuracy — `returns` feature directly encodes the label. All 5 models suspect.
2. **Ghost strategies**: Librarian top 3 (mtf_trend_align, ml_gbt_5m, mtf_momentum_confirm) exist only as DB records. No strategy files, can't generate signals.
3. **ML not integrated into live pipeline**: Models train during daily run but predictions aren't used for signal confirmation or trade decisions.
4. **Only 2 strategies generating signals** in last 24h (vwap_reversion, volume_mean_reversion). The other 8 are registered but barely firing.
5. **No strategy lifecycle management**: No tracking of strategy maturity (research → backtest → forward-test → live). Strategies are either "on" or "off."
6. **Backtester disconnected from live loop**: 22 backtest results total. Should be thousands.
7. **Daily pipeline runs once at midnight**: Should run components at different frequencies.
8. **Dashboard shows misleading metrics**: Top section averaged all garbage; bottom showed ghost strategies.

### What's Salvageable ♻️
- All the data (35M ticks, 36K trades) — gold
- Feature library — solid
- Strategy file structure and signal generation pattern
- EEP scoring framework
- Paper trading execution loop
- Orchestration skeleton (designer, tuner, ranker)
- Dashboard (needs metric fixes, not rebuild)

---

## Proposed Architecture: Strategy Lifecycle Pipeline

### Three Tiers (Rob's Framework)

```
Tier 0: LIBRARY (Research)          → Generate & catalog strategy ideas
Tier 1: BACKTEST (Validation)       → ML training + pattern backtesting on historical data
Tier 2: FORWARD TEST (Live Proof)   → Best backtests run on live data, paper trading
         ↓ (future)
Tier 3: LIVE (Real Money)           → Proven forward-test winners with real capital
```

### Master Strategy Registry

New table: `strategy_registry` — single source of truth for ALL strategies.

```sql
CREATE TABLE strategy_registry (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT UNIQUE NOT NULL,
    tier INTEGER DEFAULT 0,           -- 0=library, 1=backtest, 2=forward, 3=live
    strategy_type TEXT,                -- 'pattern', 'ml_entry', 'ml_exit', 'ensemble', 'hybrid'
    description TEXT,
    source TEXT,                       -- 'designed', 'tuned', 'librarian', 'manual'
    file_path TEXT,                    -- path to .py file (NULL = ghost)
    ml_entry_model TEXT,              -- model name for entry signals (NULL = rule-based)
    ml_exit_model TEXT,               -- model name for exit signals (NULL = rule-based)
    
    -- Backtest metrics (Tier 1)
    bt_win_rate REAL,
    bt_sharpe REAL,
    bt_pnl_pct REAL,
    bt_max_dd REAL,
    bt_trades INTEGER,
    bt_eep_score REAL,
    bt_last_run TEXT,                  -- ISO timestamp
    
    -- Forward test metrics (Tier 2)  
    ft_win_rate REAL,
    ft_sharpe REAL,
    ft_pnl_pct REAL,
    ft_max_dd REAL,
    ft_trades INTEGER,
    ft_eep_score REAL,
    ft_started TEXT,
    ft_last_update TEXT,
    
    -- Lifecycle
    promoted_at TEXT,                  -- when moved to current tier
    demoted_at TEXT,
    archived INTEGER DEFAULT 0,
    archive_reason TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### ML Redesign

#### Problem: Current ML
- 5 models, all trained on same features, all predicting future price movement
- direction_predictor has data leakage (returns ≈ label)
- Models aren't used in the signal/trade pipeline
- No entry vs exit distinction

#### Fix: Purpose-Built ML Models

**Entry Models** (predict: "should I enter this trade?")
- `ml_entry_classifier`: Given features at signal time, predict if trade will be profitable (binary)
- `ml_entry_confidence`: Predict expected return magnitude (regression)
- Training data: historical signals → matched paper trade outcomes (was it profitable? by how much?)
- **No leakage possible**: Features at time T, label from time T+N (actual trade result)

**Exit Models** (predict: "should I exit now?")
- `ml_exit_timing`: Given open position + current features, predict optimal exit (hold/close)
- `ml_trailing_stop`: Predict dynamic stop-loss level based on volatility regime
- Training data: open trades → subsequent price action → optimal exit point (hindsight labeling with proper temporal split)

**Regime Models** (predict market state)
- `ml_regime_detector`: Classify current market regime (trending/ranging/volatile/quiet)
- Used to weight strategy selection — trend strategies in trending markets, mean-reversion in ranging

#### Training Data Pipeline
```
ticks → features → signals (with known outcomes from paper_trades)
                         ↓
              ML training pairs:
              X = features at signal time
              y = actual trade outcome (profit/loss, hold time, max adverse excursion)
```

**Critical**: 24-hour temporal embargo between train/test split. No shuffling. Walk-forward validation.

### Strategy Types

1. **Pattern strategies** (current): Rule-based entry/exit from technical indicators
   - Already working: rsi_divergence, vwap_reversion, momentum, etc.
   - Backtest with historical replay, score with EEP

2. **ML-enhanced strategies**: Pattern entry + ML exit (or ML entry + pattern exit)
   - Use entry model confidence to filter weak signals
   - Use exit model to dynamically adjust stop/target

3. **Pure ML strategies**: Both entry and exit driven by trained models
   - Based on Librarian concepts (mtf_trend_align, ml_gbt_5m, mtf_momentum_confirm)
   - Need actual implementation as .py strategy files

4. **Ensemble strategies**: Combine multiple strategy signals
   - Vote-based: enter when 2+ strategies agree
   - Weighted: weight by recent EEP score

### Librarian → Real Strategy Conversion

The 3 Librarian candidates have good concepts but no code. Convert them:

| Ghost Strategy | Concept | Implementation |
|---|---|---|
| mtf_trend_align | Multi-timeframe trend alignment (5m + 1h) | Build real strategy using 5m RSI + 1h EMA trend filter |
| ml_gbt_5m | Gradient boosted tree on 5m features | Pure ML entry strategy using retrained GBT model |
| mtf_momentum_confirm | Multi-timeframe momentum confirmation (15m + 4h) | Pattern strategy: 15m momentum signal confirmed by 4h trend |

### Pipeline Schedule

| Component | Frequency | What It Does |
|---|---|---|
| **Ingestor** | Continuous | Websocket → ticks (already running) |
| **Signal Generator** | Continuous | Strategies scan ticks → signals (already running) |
| **Paper Trader** | Continuous | Confirmed signals → paper trades (already running) |
| **Backtester** | Every 2h | Run all Tier 0/1 strategies against last 7 days of ticks |
| **ML Trainer** | Every 4h | Retrain entry/exit/regime models with latest trade outcomes |
| **Strategy Tuner** | Every 6h | LLM-guided parameter tuning for underperforming strategies |
| **Ranker/Promoter** | Daily | Score all strategies, promote/demote between tiers |
| **Designer** | Daily | Generate 3-5 new strategy concepts (Tier 0) |
| **Pruner** | Daily | Archive strategies that fail gates after 3+ attempts |

### Promotion Gates

**Tier 0 → Tier 1** (Library → Backtest):
- Strategy file exists and imports cleanly
- Can generate at least 1 signal on historical data

**Tier 1 → Tier 2** (Backtest → Forward Test):
- ≥ 50 backtest trades
- Win rate ≥ 40%
- Sharpe ≥ 0.5
- Max DD ≤ 30%
- EEP score ≥ 50
- Profit factor ≥ 1.1

**Tier 2 → Tier 3** (Forward Test → Live):
- ≥ 100 forward-test trades
- Win rate ≥ 45%
- Sharpe ≥ 1.0
- Backtest-to-live convergence < 20% drift
- EEP score ≥ 65
- Profit factor ≥ 1.3
- 14+ days of forward testing

### Dashboard Fixes

1. **Top metrics**: Show Tier 2 (forward-test) strategy averages, not all strategies
2. **Strategy table**: Add tier column, color-coded (green=T2, blue=T1, gray=T0)
3. **ML section**: Show actual model performance with proper out-of-sample metrics
4. **Remove ghost data**: Purge Librarian DB entries, replace with real implementations

---

## Execution Plan

### Phase 1: Foundation (Day 1) — 3 Subagents
**Subagent A (Sonnet)**: Database & Registry
- Create `strategy_registry` table
- Migration script to populate from existing `strategy_scores` + strategy files
- Mark ghosts as archived, real strategies as Tier 2 (forward-testing)
- Add tier tracking to all queries

**Subagent B (Sonnet)**: ML Fix
- Fix data leakage in direction_predictor (remove `returns` from features, or fix label generation)
- Build `ml_entry_classifier` — train on signals + trade outcomes
- Build `ml_exit_timing` — train on open trades + subsequent price action
- Proper temporal split with 24h embargo
- Add ML predictions to signal confirmation pipeline

**Subagent C (Sonnet)**: Librarian → Real Strategies
- Implement `mtf_trend_align.py` — multi-timeframe trend alignment
- Implement `ml_gbt_5m.py` — gradient boosted tree entry
- Implement `mtf_momentum_confirm.py` — multi-timeframe momentum
- Register in strategy_registry as Tier 0
- Run initial backtests

### Phase 2: Integration (Day 2) — 2 Subagents
**Subagent D (Sonnet)**: Pipeline Scheduling
- Break daily_runner into component services with separate timers
- Backtester: every 2h
- ML trainer: every 4h
- Tuner: every 6h
- Ranker/promoter/pruner: daily
- Add promotion/demotion logic

**Subagent E (Sonnet)**: Dashboard & Monitoring
- Fix dashboard to show tier-based metrics
- Add strategy lifecycle visualization
- Fix heartbeat to check ML model freshness, strategy counts per tier, signal generation health
- Add real ML out-of-sample metrics display

### Phase 3: Acceleration (Day 3+)
- Crank up strategy generation (5+ per hour)
- Add ensemble strategies combining top performers
- A/B test ML-enhanced vs pure-pattern strategies
- Set up hourly backtesting for rapid iteration

---

## Cost & Resource Assessment

- **Claude Max 5x ($100/mo)**: Sufficient for all LLM calls (strategy design, tuning)
- **CPU**: ML training is lightweight (XGBoost on <100K samples = seconds, not hours)
- **Disk**: Current 34M ticks ≈ 2GB. Sustainable for months.
- **No paid APIs needed**: All data from existing websocket feeds
- **No Claude Teams needed**: This is a 3-day fix with 5 Sonnet subagents, well within current plan

## Open Questions
1. Should we keep all 36 symbols or focus on top 10 by volume for faster iteration?
2. Target holding period: keep current ~100 min avg, or experiment with shorter (5-15 min scalping)?
3. Risk budget per strategy for eventual Tier 3 (live): fixed $ amount or Kelly Criterion?
