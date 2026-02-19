# Blofin ML Trading Pipeline

Autonomous AI-driven crypto strategy research and paper trading engine.

## Architecture

```
Data (35M+ ticks) → Strategy Library (52 strategies) → Backtest → Forward Test → [Live]
     ↓                        ↓                            ↓            ↓
  Ingestor              LLM Designer               BacktestEngine   Paper Engine
  (WebSocket)           (Opus/Sonnet)               (7-day replay)  (Real-time)
     ↓                        ↓                            ↓            ↓
  SQLite DB            strategy_registry              EEP Scoring    ML Models
  (ticks, signals)     (tier tracking)              (PF-weighted)   (entry/exit)
```

## Strategy Lifecycle (3-Tier Pipeline)

| Tier | Name | Count | Description |
|------|------|-------|-------------|
| 0 | Library | 43 | Strategy file exists, not yet backtested |
| 1 | Backtest | 0 | Backtested, accumulating metrics |
| 2 | Forward Test | 9 | Paper trading with real signals |
| 3 | Live | 0 | Real money (future) |

**Promotion Gates:**
- T0→T1: File exists + generates ≥1 signal
- T1→T2: ≥50 trades, WR≥40%, Sharpe≥0.5, EEP≥50
- T2→T3: ≥100 trades, WR≥45%, Sharpe≥1.0, EEP≥65

## Key Metrics

**EEP Score** (Entry + Exit Package) — NOT win rate:
- Entry (60%): Profit Factor (30%) + Sharpe (25%) + Max DD (20%) + Sortino (15%) + Expectancy (10%)
- Exit (40%): Max profit capture (25%) + R:R realization (20%) + Stop-hit freq (15%) + Breakeven (10%)

## Services

| Service | Port | Description |
|---------|------|-------------|
| `blofin-stack-ingestor` | — | WebSocket tick ingestion (36 symbols) |
| `blofin-stack-paper` | — | Paper trading engine |
| `blofin-dashboard` | 8888 | Dashboard + API |

## Dashboard

**http://127.0.0.1:8888** — Pipeline visualization, strategy table (EEP-ranked), ML model status, live data feed.

## Database

SQLite at `data/blofin_monitor.db`:
- `ticks` — 35M+ market ticks
- `signals` — 45K+ trading signals
- `paper_trades` — 35K+ paper trades
- `strategy_registry` — 52 strategies with tier tracking
- `strategy_scores` — Historical backtest scores
- `ml_model_results` — ML model training results

## Pipeline Automation

```bash
# Run full pipeline (hourly via cron)
python3 orchestration/run_pipeline.py

# Individual runners
python3 orchestration/run_backtester.py    # Backtest all strategies
python3 orchestration/run_ml_trainer.py    # Train ML models
python3 orchestration/run_strategy_tuner.py # Tune underperformers
python3 orchestration/run_ranker.py        # Rank + promote/demote
```

## Quick Start

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
. .venv/bin/activate
python3 orchestration/run_pipeline.py --dry-run  # Preview what would happen
python3 orchestration/run_pipeline.py             # Run the pipeline
```
