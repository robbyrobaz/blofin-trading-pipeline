# Getting Started with Blofin AI Stack

## Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment (`.venv` directory with dependencies installed)
- SQLite database with historical tick data

### Running the Pipeline Manually

```bash
cd /home/rob/.openclaw/workspace/blofin-stack

# Activate virtual environment
source .venv/bin/activate

# Run the daily pipeline
python orchestration/daily_runner.py
```

The pipeline will:
1. Load historical tick data from the database
2. Run strategy backtests
3. Compute features for ML models
4. Train ML models (when ready)
5. Generate a daily report
6. Save all results to the database
7. Export report to `data/reports/YYYY-MM-DD.json`

### Expected Output

Check these locations after a run:

- **Log file:** `data/pipeline.log`
- **Daily report:** `data/reports/YYYY-MM-DD.json`
- **Database:** `data/blofin_monitor.db` (updated with backtest/model results)

### Testing

Run the test suite to verify everything works:

```bash
# All tests
python -m pytest tests/ -v

# Integration tests only
python -m pytest tests/test_integration.py -v

# Backtester tests
python -m pytest backtester/tests/ -v
```

### Minimum Requirements for First Run

The pipeline needs:
- At least 100 ticks in the database for the target symbol
- 7 days of historical data (default, configurable)
- All Python dependencies installed

If data is insufficient, the pipeline will log a warning and exit gracefully.

## Module Overview

### Features (`features/`)
Computes 50+ technical features:
- Price features (momentum, returns, gaps)
- Volume features (VWAP, OBV, volume surges)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volatility measures (ATR, historical volatility)
- Market regime detection

### Backtester (`backtester/`)
Backtests strategies and ML models:
- Loads historical OHLCV data
- Runs strategy signal functions
- Calculates performance metrics (win rate, Sharpe, drawdown)
- Supports multiple timeframes (1m, 5m, 15m, 1h, etc.)

### ML Pipeline (`ml_pipeline/`)
Trains predictive models:
- Direction Predictor (UP/DOWN classification)
- Price Predictor (future price regression)
- Momentum Classifier (momentum state)
- Volatility Regressor (volatility prediction)
- Risk Scorer (risk assessment)

### Orchestration (`orchestration/`)
Coordinates the entire pipeline:
- `daily_runner.py` - Main entry point for daily execution

## Troubleshooting

### "Insufficient data" Error

If you see:
```
Insufficient data: only X ticks. Need at least 100.
```

Solution:
- Run the ingestor to collect more data: `python ingestor.py`
- Or reduce `DAYS_BACK` in `daily_runner.py` (not recommended)

### Import Errors

Ensure virtual environment is activated:
```bash
source .venv/bin/activate
which python  # Should point to .venv/bin/python
```

### Database Locked

If you see "database is locked":
- Stop other processes using the database (ingestor, API server)
- Check for stale `.db-wal` or `.db-shm` files
- Wait a few seconds and retry

## Next Steps

- Review the daily report in `data/reports/`
- Check the database for top strategies and models
- Tune strategy parameters based on backtest results
- Enable the systemd timer for daily execution (see `DEPLOYMENT.md`)
