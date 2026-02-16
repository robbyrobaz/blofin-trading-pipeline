# Orchestration Module

Automated daily pipeline that coordinates all blofin-stack components.

## Components

### 1. `daily_runner.py` - Main Orchestrator
Entry point for the daily pipeline. Coordinates all steps:

- **Score strategies** (2 min, Haiku) - Evaluate all active strategies
- **Design strategies** (45 min, Opus) - Create new strategies based on portfolio gaps
- **Tune strategies** (20 min, Sonnet) - Improve underperformers
- **Train ML models** (50 min, Sonnet) - Train and evaluate models
- **Rank & update** (2 min) - Dynamic ranking, keep top performers
- **Generate report** (5 min, Haiku) - Daily summary
- **AI review** (10 min, Opus) - Strategic analysis and recommendations

**Total runtime:** ~2.5 hours

### 2. `strategy_designer.py` - Opus-Powered Strategy Creation
Analyzes current portfolio and designs new strategies:

- Identifies gaps in strategy coverage
- Learns from top/bottom performers
- Adapts to current market regime
- Generates Python code for new strategies
- Registers strategies in database

### 3. `strategy_tuner.py` - Sonnet-Powered Optimization
Improves underperforming strategies:

- Analyzes failure patterns
- Suggests parameter adjustments
- Generates tuned strategy versions
- Logs all changes to knowledge base

### 4. `ranker.py` - Dynamic Ranking System
No hard pass/fail - everything is ranked:

- `keep_top_strategies(count=20)` - Top 20 by score
- `keep_top_models(count=5)` - Top 5 by F1
- `keep_top_ensembles(count=3)` - Top 3 by accuracy
- `archive_bottom(names, reason)` - Archive with reason
- All ranking decisions logged for auditability

### 5. `reporter.py` - Daily Report Generator
Comprehensive daily summaries:

- Activity metrics (designed/tuned/archived)
- Top performers (strategies, models, ensembles)
- Performance trends
- Portfolio health assessment
- Saves to JSON and database
- Integrates AI review results

## Installation

### 1. Update Database Schema
The orchestration module adds new tables. Initialize:

```bash
cd ~/.openclaw/workspace/blofin-stack
python3 -c "import db; con = db.connect('data/blofin_monitor.db'); db.init_db(con); con.close()"
```

### 2. Install Systemd Timer
Run daily at 00:00 UTC:

```bash
cd ~/.openclaw/workspace/blofin-stack/orchestration
./install_systemd.sh
```

### 3. Verify Installation
```bash
# Check timer status
systemctl --user status blofin-stack-daily.timer

# View next scheduled run
systemctl --user list-timers blofin-stack-daily.timer

# View logs
journalctl --user -u blofin-stack-daily.service -f
```

## Manual Execution

### Run Full Pipeline
```bash
cd ~/.openclaw/workspace/blofin-stack
python3 orchestration/daily_runner.py
```

### Run Individual Components
```bash
# Test ranker
python3 orchestration/ranker.py

# Test reporter
python3 orchestration/reporter.py

# Test strategy designer
python3 orchestration/strategy_designer.py

# Test strategy tuner
python3 orchestration/strategy_tuner.py
```

## Database Schema

### New Tables

#### `strategy_backtest_results`
Stores backtest results for all strategies.

#### `ml_model_results`
Stores ML model training results and metrics.

#### `ml_ensembles`
Stores ensemble configurations and performance.

#### `daily_reports`
Stores generated reports with AI review.

#### `ranking_history`
Audit log of all ranking decisions.

## Configuration

### Environment Variables
- `BLOFIN_WORKSPACE` - Workspace directory (default: `~/.openclaw/workspace/blofin-stack`)
- `PYTHONPATH` - Should include workspace directory

### Parallel Execution
The pipeline uses `ThreadPoolExecutor` to run independent tasks in parallel:

- Design, tune, and ML training run concurrently
- Scoring, ranking, and reporting run sequentially (dependencies)
- Max 3 workers to avoid rate limits

### Error Handling
- Each step logs success/failure independently
- Pipeline continues even if one component fails
- All errors logged to `data/pipeline.log`
- Failed steps return error dict but don't crash pipeline

## Extensibility

### Adding New Steps
Edit `daily_runner.py` and add a new step method:

```python
def step_your_new_task(self) -> Dict[str, Any]:
    self._log_step('your_task', 'started')
    start_time = datetime.utcnow()
    
    try:
        # Your logic here
        result = {'success': True}
        self._log_step('your_task', 'success', result)
        return result
    except Exception as e:
        self._log_step('your_task', 'failure', {'error': str(e)})
        return {'error': str(e)}
```

Then add to `run_daily_pipeline()`:
```python
# Sequential
self.results['your_task'] = self.step_your_new_task()

# Or parallel
parallel_tasks.append(('your_task', self.step_your_new_task, ()))
```

### Customizing Ranking
Edit `ranker.py` to change:
- Top N counts (currently 20 strategies, 5 models, 3 ensembles)
- Ranking metrics (score, sharpe, F1, etc.)
- Archive thresholds

### Customizing Reports
Edit `reporter.py` to add:
- New metrics
- Different time windows
- Custom health checks
- Additional visualizations

## Monitoring

### Check Pipeline Status
```bash
# View recent logs
tail -f ~/.openclaw/workspace/blofin-stack/data/pipeline.log

# Check if pipeline is running
systemctl --user is-active blofin-stack-daily.service

# View last run results
cat ~/.openclaw/workspace/blofin-stack/data/reports/$(date +%Y-%m-%d).json | jq .
```

### Debugging
```bash
# Run with verbose logging
python3 -u orchestration/daily_runner.py 2>&1 | tee debug.log

# Test individual components
python3 orchestration/ranker.py
python3 orchestration/reporter.py
```

## Integration with Agents #1-3

The orchestration layer currently stubs out these components:

1. **Strategy Manager** (Agent #1) - Called in `step_score_strategies()`
2. **ML Pipeline** (Agent #2) - Called in `step_train_ml_models()`
3. **Backtesting** (Agent #3) - Called in `step_backtest_new_strategies()`

As these components are built, replace the stub calls with actual integrations.

## Logs and Outputs

- **Pipeline log:** `data/pipeline.log`
- **Daily reports:** `data/reports/YYYY-MM-DD.json`
- **Systemd logs:** `journalctl --user -u blofin-stack-daily.service`

## Troubleshooting

### Timer not running
```bash
# Check if timer is enabled
systemctl --user is-enabled blofin-stack-daily.timer

# Enable if needed
systemctl --user enable blofin-stack-daily.timer
systemctl --user start blofin-stack-daily.timer
```

### Pipeline fails immediately
```bash
# Check Python environment
which python3
python3 --version

# Verify virtual environment
source ~/.openclaw/workspace/blofin-stack/.venv/bin/activate
python3 -c "import sqlite3; print('OK')"
```

### Opus/Sonnet not responding
- Check OpenClaw CLI is installed: `openclaw --version`
- Verify API credentials: `openclaw config list`
- Check rate limits in logs

## Performance

Expected durations (may vary):
- Scoring: 2 min
- Design: 45 min (2-3 strategies × 15 min each)
- Tuning: 20 min (3 strategies × 7 min each)
- ML training: 50 min
- Ranking: 2 min
- Reporting: 5 min
- AI review: 10 min

**Total:** ~2.5 hours

Peak occurs during parallel phase (design + tune + ML = max 50 min).
