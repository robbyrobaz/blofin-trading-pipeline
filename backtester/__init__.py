#!/usr/bin/env python3
"""
Backtesting module for the Blofin trading stack.

Provides:
- BacktestEngine: Core backtesting engine for strategies and models
- OHLCV aggregation functions (1m -> 5m -> 60m)
- Performance metrics (win rate, Sharpe, drawdown, score)

Usage:
    from backtester import BacktestEngine, calculate_all_metrics
    
    engine = BacktestEngine(symbol='BTC-USDT', days_back=7)
    results = engine.run_strategy(my_strategy, timeframe='5m')
    print(results['metrics'])
"""

from .backtest_engine import BacktestEngine
from .aggregator import (
    aggregate_ohlcv,
    aggregate_ticks_to_1m_ohlcv,
    fast_aggregate_numpy,
    timeframe_to_minutes
)
from .metrics import (
    calculate_win_rate,
    calculate_avg_pnl_pct,
    calculate_total_pnl_pct,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_score,
    calculate_all_metrics,
    format_metrics,
    is_profitable
)

__all__ = [
    # Core engine
    'BacktestEngine',
    
    # Aggregation
    'aggregate_ohlcv',
    'aggregate_ticks_to_1m_ohlcv',
    'fast_aggregate_numpy',
    'timeframe_to_minutes',
    
    # Metrics
    'calculate_win_rate',
    'calculate_avg_pnl_pct',
    'calculate_total_pnl_pct',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_score',
    'calculate_all_metrics',
    'format_metrics',
    'is_profitable',
]
