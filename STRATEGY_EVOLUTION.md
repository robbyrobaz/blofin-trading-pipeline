# Strategy Evolution System

## Overview

The Blofin trading stack now features a self-improving strategy system with:

- **11 modular strategies** (6 migrated + 5 new)
- **Plugin architecture** for easy strategy development
- **Knowledge base** tracking performance, lessons, and configuration history
- **AI review system** for periodic analysis and auto-optimization
- **Backward compatibility** with fallback to legacy hardcoded strategies

## Architecture

```
strategies/               # Plugin directory
├── base_strategy.py      # Abstract base class
├── momentum.py           # Existing strategies (migrated)
├── breakout.py
├── reversal.py
├── vwap_reversion.py
├── rsi_divergence.py
├── bb_squeeze.py
├── ema_crossover.py      # NEW: EMA crossover signals
├── volume_spike.py       # NEW: Volume surge detection
├── support_resistance.py # NEW: S/R levels
├── macd_divergence.py    # NEW: MACD divergence
├── candle_patterns.py    # NEW: Candlestick patterns
└── __init__.py           # Auto-discovery

strategy_manager.py       # Load, enable/disable, execute strategies
knowledge_base.py         # Performance scoring, lessons, config history
ai_review.py              # Periodic AI analysis script
ingestor.py               # Modified to use StrategyManager (w/ fallback)
db.py                     # Extended with knowledge base tables
```

## Strategies

### Migrated Strategies (6)
1. **momentum** - Price momentum in time window
2. **breakout** - Price breaking above/below recent highs/lows
3. **reversal** - Bounce from local extremes
4. **vwap_reversion** - Mean reversion from VWAP
5. **rsi_divergence** - RSI overbought/oversold
6. **bb_squeeze** - Bollinger Band squeeze breakouts

### New Strategies (5)
7. **ema_crossover** - 9/21 EMA crossover (configurable periods)
8. **volume_spike** - Volume surge (2-3x avg) + price confirmation
9. **support_resistance** - Price level clustering + rejection detection
10. **macd_divergence** - MACD histogram divergence from price
11. **candle_patterns** - Engulfing, hammer, shooting star, doji detection

## Knowledge Base

### Database Tables

**strategy_scores** - Performance metrics per strategy/symbol/window:
- Tracks: trades, wins, losses, win_rate, avg_pnl_pct, total_pnl_pct, sharpe_ratio, max_drawdown_pct, score (0-100)
- Windows: 24h, 7d, all
- Composite score formula: `win_rate*40 + avg_pnl*30 + sharpe*20 - drawdown*10`

**knowledge_entries** - Lessons learned, recommendations, observations:
- Categories: performance, lesson, recommendation, change
- Sources: ai_review, auto_score, manual

**strategy_configs** - Configuration change history:
- Tracks who changed what and why
- Enables rollback and audit

### Functions

- `compute_strategy_scores(con, strategy, window, symbol)` - Calculate metrics
- `update_all_scores(con)` - Refresh all scores
- `add_knowledge_entry(con, category, content, ...)` - Log insights
- `get_knowledge_summary(con)` - Format for AI review
- `auto_manage_strategies(con, manager)` - Enable/disable based on scores
- `save_strategy_config(con, strategy, config, ...)` - Track config changes

## AI Review System

**ai_review.py** runs periodically (twice daily by default) to:

1. Pull current performance data
2. Detect market regime (ranging/trending/volatile)
3. Generate AI prompt with context
4. Call AI API (placeholder for now)
5. Parse recommendations (JSON)
6. Apply safe changes automatically:
   - Enable/disable strategies
   - Tune parameters within bounds
7. Log everything to knowledge base
8. Write JSON report to `data/ai_reviews/`

### Recommendation Format

```json
{
  "analysis": "text summary",
  "recommendations": [
    {"action": "disable", "strategy": "bb_squeeze", "reason": "poor performance"},
    {"action": "tune", "strategy": "momentum", "params": {"up_pct": 0.8}, "reason": "reduce sensitivity"},
    {"action": "enable", "strategy": "ema_crossover", "reason": "good in trending market"}
  ],
  "market_regime": "ranging",
  "confidence": 0.75
}
```

### Systemd Timer

```bash
# Install timer
sudo cp blofin-stack-ai-review.service /etc/systemd/system/
sudo cp blofin-stack-ai-review.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable blofin-stack-ai-review.timer
sudo systemctl start blofin-stack-ai-review.timer

# Check status
systemctl status blofin-stack-ai-review.timer
systemctl list-timers | grep blofin

# Manual run
sudo systemctl start blofin-stack-ai-review.service

# View logs
journalctl -u blofin-stack-ai-review.service -f
```

## Ingestor Integration

The ingestor now:
1. Tries to load StrategyManager on startup
2. Falls back to legacy strategies if plugin system unavailable
3. Detects signals using `manager.detect_all()` when available
4. Updates performance scores every 5 minutes
5. Auto-manages strategies every hour
6. Maintains full backward compatibility

## Testing

```bash
# Run test suite
python test_strategies.py

# Manual verification
python -c "from strategies import get_all_strategies; print([s.name for s in get_all_strategies()])"

# Test AI review
python ai_review.py
```

## Development

### Adding a New Strategy

1. Create `strategies/your_strategy.py`:

```python
from .base_strategy import BaseStrategy, Signal

class YourStrategy(BaseStrategy):
    name = "your_strategy"
    version = "1.0"
    description = "What it does"
    
    def __init__(self):
        # Load config from env or defaults
        self.param = float(os.getenv("YOUR_PARAM", "1.0"))
    
    def detect(self, symbol, price, volume, ts_ms, prices, volumes):
        # Your logic here
        if condition:
            return Signal(
                symbol=symbol,
                signal="BUY",
                strategy=self.name,
                confidence=0.75,
                details={"info": "value"}
            )
        return None
    
    def get_config(self):
        return {"param": self.param}
    
    def update_config(self, params):
        if "param" in params:
            self.param = float(params["param"])
```

2. Add to `strategies/__init__.py`:

```python
from .your_strategy import YourStrategy

def get_all_strategies():
    strategies = [
        # ... existing ...
        YourStrategy(),
    ]
    return strategies
```

3. Test: `python test_strategies.py`

### Tuning Parameters

Via environment variables:
```bash
export MOMENTUM_UP_PCT=0.8
export EMA_FAST_PERIOD=12
```

Via AI review (automatic):
```python
{"action": "tune", "strategy": "momentum", "params": {"up_pct": 0.8}}
```

Via code:
```python
manager.update_strategy_config("momentum", {"up_pct": 0.8})
```

## Performance Monitoring

Dashboard (TODO):
- Strategy performance table
- Win rate trends
- Enable/disable toggles
- Config editor
- Knowledge entries log
- AI review history

Query performance manually:
```sql
SELECT strategy, window, trades, win_rate, avg_pnl_pct, score
FROM strategy_scores
WHERE symbol IS NULL
ORDER BY strategy, 
  CASE window WHEN '24h' THEN 1 WHEN '7d' THEN 2 ELSE 3 END;
```

## Safety Features

1. **Fallback mode** - If plugin system fails, uses legacy strategies
2. **Score-based auto-disable** - Strategies with score <30 (10+ trades) disabled automatically
3. **Cooldown protection** - Same signal type has 4-minute cooldown
4. **Audit trail** - All changes logged to knowledge_entries + strategy_configs
5. **Bounded tuning** - AI can only tune within safe parameter ranges

## Next Steps

1. **Implement real AI API** - Replace mock in `ai_review.py` with OpenClaw/OpenAI/Claude
2. **Dashboard integration** - Add strategy management UI
3. **Backtesting** - Test strategies on historical data before enabling
4. **Regime-aware execution** - Activate different strategies based on market regime
5. **Multi-timeframe analysis** - Combine signals across timeframes
6. **Risk management** - Position sizing based on confidence + volatility
7. **Strategy breeding** - Genetic algorithms to evolve new strategy combinations

## Files Modified

- `db.py` - Added knowledge base tables
- `ingestor.py` - Integrated StrategyManager with fallback

## Files Created

- `strategies/` (13 files) - Plugin system
- `strategy_manager.py` - Strategy orchestration
- `knowledge_base.py` - Performance tracking
- `ai_review.py` - AI analysis script
- `test_strategies.py` - Test suite
- `blofin-stack-ai-review.service` - Systemd service
- `blofin-stack-ai-review.timer` - Systemd timer
- `STRATEGY_EVOLUTION.md` - This document
