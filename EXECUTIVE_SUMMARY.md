# Executive Summary — Blofin AI Trading Pipeline v3 (REVISED)

## What We're Building

A fully automated, AI-driven trading research platform that:

1. **Continuously designs new trading strategies** (Opus) every day
2. **Backtests everything first** on last 7 days of data (multiple timeframes)
3. **Keeps 20 strategies active** at all times (top performers + new candidates)
4. **Builds 5 ML models every day** (direction, risk, price, momentum, volatility)
5. **Validates models via backtesting** (not just accuracy, but real-world Sharpe ratio)
6. **Creates ensemble combinations** (multiple models voting for better predictions)
7. **Generates daily reports** (human-readable + AI-readable JSON)
8. **Requires zero manual intervention** (fully automated loops)
9. **Runs forever in research mode** until top 3 strategies ready for live trading

**Status:** Build all code in next 24 hours. Deploy tomorrow. Run daily forever.

---

## The Daily Workflow (Every 24 Hours at Midnight UTC)

```
00:00 UTC — Pipeline Starts
│
├─ Score all 20 strategies + 5 models (2 min)
│  └─ Pull last 7 days backtest results
│
├─ [PARALLEL] Design new strategies + backtest
│  ├─ Opus: Design 2-3 new candidates (15 min)
│  ├─ Sonnet: Backtest on 1m/5m/60m (45 min)
│  └─ Haiku: Validate vs live data (5 min)
│
├─ [PARALLEL] Tune underperformers
│  └─ Sonnet: Analyze failures, suggest param changes (20 min)
│
├─ [PARALLEL] Build ML models
│  ├─ Sonnet: Train 5 models in parallel (50 min)
│  ├─ Sonnet: Backtest each model (20 min)
│  └─ Sonnet: Test ensemble combinations (15 min)
│
├─ Rank & Update
│  ├─ Keep top 20 strategies
│  ├─ Keep top 5 models
│  ├─ Keep top 3 ensemble configs
│  └─ Archive bottom performers
│
├─ Generate Daily Report
│  ├─ Human-readable summary (what changed, why)
│  ├─ AI-readable JSON (metrics, changes, performance)
│  └─ Opus review: "What should we focus on tomorrow?"
│
└─ 02:30 UTC — Pipeline Complete
   └─ Results saved, cron waits 21.5 hours for next run
```

**Total execution time:** ~2.5 hours (mostly parallel)

---

## Cost Analysis (Claude Max 5x Plan)

### Monthly Cost: **$100** (Fixed)
- Covers all development + operations
- Generous token pool that refreshes every few hours
- Can run pipeline daily forever

### Marginal Cost per Pipeline Run: **$0**
- After monthly subscription paid, each run is free
- 365 runs/year = included
- Scale infinitely, cost stays $100/month

### Economics
- Payoff time: 2-3 months if one strategy goes live
- Cost to find 3 good strategies: $100-300
- Cost of failed strategies: $0 (backtest-only)
- **ROI:** One 2% monthly strategy pays for a year of research in 50 days

---

## Timeline

**Immediate (Next 24 Hours):**
- Phase 1: Feature library (Sonnet Agent #1)
- Phase 2: Backtester (Sonnet Agent #2)
- Phase 3: ML pipeline (Sonnet Agent #3)
- Phase 4: Strategy designer (Opus Agent + Sonnet)
- Phase 5: Orchestration (Sonnet Agent #4)
- **Result:** Complete system, tested, deployed

**Day 1 (Tomorrow +26h):**
- Deploy cron job
- Run first full cycle
- Verify all components working
- Check for bugs/crashes

**Days 2-60 (Research Phase):**
- Pipeline runs daily at midnight
- Strategies evolve, ML improves
- Daily reports show progress
- Keep going until top 3 look good

**Day 61+ (Live):**
- Deploy top 3 strategies live with real money
- Keep research pipeline running (evolve new ones)
- Continuous improvement

---

## What Makes This Work

### 1. Backtest-First Validation
- New strategy: backtest on 7 days (20 min) vs live (7 days real time)
- **7x faster feedback** = **7x more iterations**
- Failed backtest costs $0, failed trade costs money
- Only strategies passing backtest get consideration for live

### 2. Continuous Design Pipeline
- Opus designs 2-3 new candidates daily
- Sonnet backtests them automatically
- Bottom performers automatically replaced
- **No human bottleneck** in design

### 3. ML Model Evolution
- Train 5 new models daily (fresh data)
- Each can specialize (direction, risk, price, etc.)
- Ensemble combinations tested automatically
- Retraining detects and fixes model drift

### 4. Dynamic Ranking (Not Hard Thresholds)
- Always keep top 20 strategies
- Always keep top 5 models
- Market conditions change → thresholds should change too
- Forces continuous improvement

### 5. Daily Execution
- Same cycle, every single day
- Predictable, automatable, reliable
- Cheap (already paid for with $100/month)
- No manual steps

---

## Success Criteria (After 2 Months)

When we're ready to go live, we should have:

✅ **Top 3 Strategies:** All with >50 backtest score, <15% live drift, >40% win rate
✅ **2+ Ensemble Models:** Accuracy >55%, F1 >0.55 on live holdout data
✅ **Zero Manual Design:** All strategies AI-created, zero human-coded
✅ **Robust Backtester:** Handles edge cases, reliable metrics
✅ **Full Automation:** Daily runs, zero crashes, zero manual intervention
✅ **Cost Validated:** ~$100/month operational cost confirmed
✅ **Ready for $1000 live test:** Confident in at least 3 candidates

---

## Why This Beats Manual Approaches

| Aspect | Manual | Automated |
|--------|--------|-----------|
| Strategies designed/week | 1-2 (slow) | 10-15 (AI) |
| Testing time | 7 days live | 20 min backtest |
| Failure cost | Real money | $0 |
| Iteration speed | Weeks | Days |
| ML model evolution | Never | Daily |
| Oversight required | 5-10 hours/week | 5 min/day (reading report) |
| Scalability | Manual limit (time) | Parallel, infinite |

---

## Files Being Built (Next 24 Hours)

**Core Components:**
- [ ] `features/feature_manager.py` (central API for 50+ features)
- [ ] `features/price_features.py` (OHLC, returns, momentum)
- [ ] `features/volume_features.py` (volume indicators)
- [ ] `features/technical_indicators.py` (RSI, MACD, Bollinger, ATR, etc.)
- [ ] `features/volatility_features.py` (volatility measures)
- [ ] `features/market_regime.py` (trending, ranging, volatile)
- [ ] `backtester/backtest_engine.py` (data replay + metrics)
- [ ] `backtester/aggregator.py` (1min → 5min → 60min)
- [ ] `backtester/metrics.py` (score, Sharpe, drawdown calculation)
- [ ] `ml_pipeline/train.py` (5 model types in parallel)
- [ ] `ml_pipeline/validate.py` (backtest models, calc accuracy/F1)
- [ ] `ml_pipeline/tune.py` (detect drift, retrain)
- [ ] `orchestration/daily_runner.py` (main 24h cycle)
- [ ] `orchestration/strategy_designer.py` (Opus designs new strategies)
- [ ] `orchestration/strategy_tuner.py` (Sonnet tunes parameters)
- [ ] `orchestration/ranker.py` (keep top 20/5/3)
- [ ] `orchestration/reporter.py` (daily human + AI report)
- [ ] Database schema updates (new tables for results)
- [ ] Systemd cron job setup
- [ ] `blofin-stack-daily.service` + `.timer`

**Total:** ~20 code files, all built in parallel by multiple Sonnet agents.

---

## The Build Plan (24 Hours)

**Now to +4 hours (while sleeping):**
- Sonnet Agent #1: Feature library (all 6 modules)
- Sonnet Agent #2: Backtester (3 modules + tests)
- Sonnet Agent #3: ML pipeline (3 modules + model implementations)

**+4 to +8 hours:**
- Sonnet Agent #4: Orchestration (4 modules)
- Opus Agent: Strategy designer + tuner prompts
- Sonnet Agent #5: Integration + testing

**+8 to +12 hours:**
- Testing + bug fixes
- Database schema integration
- Cron setup

**+12 to +24 hours:**
- Final validation
- Deploy
- First pipeline run
- Monitor + report

---

## Key Decisions (Final Check)

1. ✅ **Daily execution** (every 24 hours at midnight UTC)
2. ✅ **7-day backtest window** (last week of data)
3. ✅ **Multi-timeframe testing** (1m, 5m, 60m)
4. ✅ **20 active strategies** (always full, always rotating)
5. ✅ **5 active ML models** (retrain daily)
6. ✅ **3 ensemble configurations** (weighted, voting, stacking)
7. ✅ **Dynamic ranking** (keep top X, no hard thresholds)
8. ✅ **50+ feature library** (single source of truth)
9. ✅ **Research mode forever** (until top 3 ready for live)
10. ✅ **$100/month fixed cost** (covered by Claude Max 5x plan)

All good?

---

**Let's build it. Spawning agents now.**
