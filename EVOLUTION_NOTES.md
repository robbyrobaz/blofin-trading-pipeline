# Blofin Stack Evolution Notes
## Pre-Evolution Snapshot — 2026-02-12

### System Status
- **1.94M ticks** across **36 symbols**
- **5,682 signals** generated, **4,135 paper trades** executed
- **32 symbols** at 100% 7-day historical coverage
- Live websocket ingestor + 30-min gap-fill timer working
- Dashboard at :8780

### Strategy Performance (Closed Trades as of 2026-02-12)
| Strategy | Trades | Win Rate | Avg PnL% | Total PnL% |
|---|---|---|---|---|
| rsi_divergence | 20 | 45.0% | +0.12% | +2.35% |
| vwap_reversion | 38 | 31.6% | -0.23% | -8.75% |
| bb_squeeze | 49 | 20.4% | -0.55% | -26.74% |
| breakout | 272 | 31.2% | -0.25% | -67.45% |
| momentum | 701 | 35.7% | -0.14% | -97.24% |
| reversal | 2329 | 39.2% | -0.04% | -97.85% |

### Key Observations
1. **Only RSI divergence is profitable** — but tiny sample size (20 trades)
2. **All other strategies are net negative** — signals fire too aggressively
3. **Reversal dominates signal count** (70%) but has mediocre win rate
4. **BB squeeze has worst win rate** (20.4%) — needs rethinking
5. **Momentum and breakout** have decent trade counts but lose money consistently

### What This Tells Us
- The system generates signals aggressively but lacks filtering
- No position sizing or risk management
- No market regime awareness (trending vs ranging)
- Strategies don't adapt to current conditions
- Need: confirmation layers, dynamic thresholds, regime detection

### Architecture Before Evolution
```
ingestor.py — monolithic, all 6 strategies hardcoded
  ↓ signals table
paper_engine.py — confirms + trades all signals equally
  ↓ paper_trades table  
api_server.py — dashboard display
```

### What We're Building
```
strategies/           — plugin folder, one file per strategy
strategy_manager.py   — loads, scores, enables/disables strategies
knowledge_base.py     — tracks performance, stores lessons learned
ai_review.py          — periodic AI analysis of KB → strategy adjustments
ingestor.py           — calls strategy_manager instead of hardcoded logic
paper_engine.py       — confidence-weighted position sizing
```

---

## Dashboard CPU Optimization — 2026-02-15

### Problem
`api_server.py` was burning ~37% CPU average due to an unbounded 3-table JOIN
fetching all 12,415 paper_trades every 3 seconds plus Python-side advanced stats.

### Changes
- Replaced per-trade JOIN with `GROUP BY` aggregate (~8 rows vs 12K+)
- Dropped Sortino/profit-factor/max-drawdown from dashboard (simplified scoring)
- Top trades: SQL `ORDER BY pnl_pct DESC LIMIT 10` (was 2,000 rows + Python sort)
- Refresh interval: 60 seconds (was 3s backend / 2s frontend)

### Scoring Formula (simplified)
```
pnl_component = clamp(0, 100, ((avg_pnl + 2.0) / 4.0) * 100)
score = (win_rate * 0.6) + (pnl_component * 0.4)
Grade: A (85+), B (70-84), C (55-69), D (40-54), F (<40)
```

### Result
CPU: ~37% → <1% average. Dashboard columns: Strategy, Signals, Trades, Win%, PnL%, Grade.
