#!/usr/bin/env python3
"""
Strategy tuner runner - LLM-guided parameter tuning for underperforming strategies.
Scheduled every 6h via blofin-stack-strategy-tuner.timer.
"""
import sys
import os
import json
import logging
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
    return logging.getLogger('run_strategy_tuner')


def run_strategy_tuner(workspace_dir: Path) -> dict:
    from orchestration.strategy_tuner import StrategyTuner

    db_path = workspace_dir / 'data' / 'blofin_monitor.db'
    strategies_dir = workspace_dir / 'strategies'
    logger = logging.getLogger('run_strategy_tuner')

    tuner = StrategyTuner(str(db_path), str(strategies_dir))

    logger.info("Identifying underperforming strategies...")
    tuning_results = tuner.tune_underperformers(max_strategies=3)

    strategy_names = [r['strategy_name'] for r in tuning_results]
    logger.info("Tuned %d strategies: %s", len(tuning_results), strategy_names)

    return {
        'strategies_tuned': len(tuning_results),
        'strategy_names': strategy_names,
        'results': [
            {
                'strategy_name': r['strategy_name'],
                'tuning_attempt': r.get('tuning_attempt'),
                'expected_improvement': r.get('suggestions', {}).get('expected_improvement', '')
            }
            for r in tuning_results
        ]
    }


def main():
    workspace_dir = Path(os.environ.get(
        'BLOFIN_WORKSPACE',
        os.path.expanduser('~/.openclaw/workspace/blofin-stack')
    ))
    log_path = workspace_dir / 'data' / 'strategy_tuner.log'
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("STRATEGY TUNER START  %s", datetime.utcnow().isoformat() + 'Z')
    logger.info("=" * 60)

    start = datetime.utcnow()
    try:
        result = run_strategy_tuner(workspace_dir)
        duration = (datetime.utcnow() - start).total_seconds()
        logger.info("STRATEGY TUNER DONE in %.1fs: %s", duration, json.dumps(result, default=str))
        sys.exit(0)
    except Exception as e:
        duration = (datetime.utcnow() - start).total_seconds()
        logger.error("STRATEGY TUNER FAILED after %.1fs: %s", duration, e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
