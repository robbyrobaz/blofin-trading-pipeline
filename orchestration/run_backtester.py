#!/usr/bin/env python3
"""
Backtester runner - runs all active strategies against last 7 days of tick data.
Scheduled every 2h via blofin-stack-backtester.timer.
"""
import sys
import os
import json
import logging
import importlib.util
from datetime import datetime
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('run_backtester')


def load_all_strategies(strategies_dir: Path) -> list:
    """Load all strategy objects from the strategies directory."""
    strategies = []
    for py_file in sorted(strategies_dir.glob('*.py')):
        if py_file.name.startswith('__'):
            continue
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if hasattr(attr, 'detect') and hasattr(attr, 'name') and callable(attr):
                    try:
                        inst = attr()
                        if hasattr(inst, 'name') and inst.name:
                            strategies.append(inst)
                    except Exception:
                        pass
        except Exception:
            pass
    # Deduplicate by name
    seen = set()
    unique = []
    for s in strategies:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)
    return unique


def run_backtester(workspace_dir: Path) -> dict:
    from backtester.backtest_engine import BacktestEngine

    db_path = workspace_dir / 'data' / 'blofin_monitor.db'
    strategies_dir = workspace_dir / 'strategies'
    logger = logging.getLogger('run_backtester')

    logger.info("Loading strategies from %s", strategies_dir)
    strategy_objects = load_all_strategies(strategies_dir)
    logger.info("Found %d unique strategies", len(strategy_objects))

    symbols = ['BTC-USDT', 'ETH-USDT']
    results = []

    import sqlite3
    for strategy_obj in strategy_objects:
        for symbol in symbols:
            try:
                engine = BacktestEngine(
                    symbol=symbol,
                    days_back=7,
                    db_path=str(db_path),
                    initial_capital=10000.0
                )
                if not engine.ticks:
                    logger.info("No tick data for %s, skipping", symbol)
                    continue

                bt_result = engine.run_strategy(
                    strategy_obj,
                    timeframe='5m',
                    stop_loss_pct=3.0,
                    take_profit_pct=5.0
                )

                metrics = bt_result.get('metrics', {})
                logger.info(
                    "Backtest %s/%s: %d trades, final=$%.2f",
                    strategy_obj.name, symbol,
                    len(bt_result['trades']), bt_result['final_capital']
                )

                # Save to database
                try:
                    ts_ms = int(datetime.utcnow().timestamp() * 1000)
                    ts_iso = datetime.utcnow().isoformat() + 'Z'
                    con = sqlite3.connect(str(db_path), timeout=30)
                    con.execute('''
                        INSERT OR REPLACE INTO strategy_backtest_results
                        (ts_ms, ts_iso, strategy, symbol, timeframe, days_back,
                         total_trades, win_rate, sharpe_ratio, max_drawdown_pct,
                         total_pnl_pct, final_capital, results_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ts_ms, ts_iso, strategy_obj.name, symbol, '5m', 7,
                        len(bt_result['trades']),
                        metrics.get('win_rate', 0),
                        metrics.get('sharpe_ratio', 0),
                        metrics.get('max_drawdown_pct', 0),
                        metrics.get('total_pnl_pct', 0),
                        bt_result['final_capital'],
                        json.dumps({'trades': bt_result['trades'][:50], 'metrics': metrics})
                    ))
                    con.commit()
                    con.close()
                except Exception as db_err:
                    logger.warning("Failed to save backtest result: %s", db_err)

                results.append({
                    'strategy': strategy_obj.name,
                    'symbol': symbol,
                    'trades': len(bt_result['trades']),
                    'final_capital': bt_result['final_capital'],
                    'metrics': metrics
                })

            except Exception as e:
                logger.warning("Backtest failed for %s/%s: %s", strategy_obj.name, symbol, e)

    return {
        'backtested_count': len(results),
        'strategies_run': len(strategy_objects),
        'results': results
    }


def main():
    workspace_dir = Path(os.environ.get(
        'BLOFIN_WORKSPACE',
        os.path.expanduser('~/.openclaw/workspace/blofin-stack')
    ))
    log_path = workspace_dir / 'data' / 'backtester.log'
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("BACKTESTER START  %s", datetime.utcnow().isoformat() + 'Z')
    logger.info("=" * 60)

    start = datetime.utcnow()
    try:
        result = run_backtester(workspace_dir)
        duration = (datetime.utcnow() - start).total_seconds()
        logger.info("BACKTESTER DONE in %.1fs: %s", duration, json.dumps(result, default=str))
        sys.exit(0)
    except Exception as e:
        duration = (datetime.utcnow() - start).total_seconds()
        logger.error("BACKTESTER FAILED after %.1fs: %s", duration, e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
