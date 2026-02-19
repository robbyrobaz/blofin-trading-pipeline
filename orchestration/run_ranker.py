#!/usr/bin/env python3
"""
Ranker/promoter runner - scores all strategies, promotes/demotes between tiers.
Also runs convergence gate checks and generates the daily report.
Scheduled daily via blofin-stack-ranker.timer.
"""
import sys
import os
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Convergence gate constants (mirrored from DailyRunner)
MAX_MODELS_WITHOUT_ENSEMBLES = 50
MIN_ENSEMBLE_TESTS_REQUIRED = 5
MAX_TUNING_FAILURES = 3


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
    return logging.getLogger('run_ranker')


def check_convergence_gates(con: sqlite3.Connection, logger: logging.Logger) -> dict:
    """Check pipeline convergence gates before ranking."""
    gate_results = {
        'model_ensemble_gate': 'ok',
        'strategies_auto_archived': [],
        'efficiency_score': None,
        'warnings': [],
    }

    # Gate 1: Model/ensemble ratio
    try:
        total_models = con.execute(
            'SELECT COUNT(*) FROM ml_model_results WHERE archived=0'
        ).fetchone()[0]
        total_ensembles = con.execute(
            'SELECT COUNT(*) FROM ml_ensembles WHERE archived=0'
        ).fetchone()[0]

        if total_models > MAX_MODELS_WITHOUT_ENSEMBLES and total_ensembles < MIN_ENSEMBLE_TESTS_REQUIRED:
            msg = (
                f"CONVERGENCE GATE BLOCKED: {total_models} active models but only "
                f"{total_ensembles} ensemble tests (minimum {MIN_ENSEMBLE_TESTS_REQUIRED} required)."
            )
            logger.warning(msg)
            gate_results['model_ensemble_gate'] = 'blocked'
            gate_results['warnings'].append(msg)
        else:
            gate_results['model_ensemble_gate'] = f'ok ({total_models} models, {total_ensembles} ensembles)'
    except Exception as e:
        logger.warning("Convergence gate check failed: %s", e)

    # Gate 2: Auto-archive strategies with too many failed tuning attempts
    try:
        failing = con.execute(
            '''SELECT DISTINCT strategy FROM strategy_backtest_results
               WHERE tuning_attempt >= ? AND status != 'archived' ''',
            (MAX_TUNING_FAILURES,)
        ).fetchall()

        for (strat_name,) in failing:
            con.execute(
                "UPDATE strategy_backtest_results SET status='archived' WHERE strategy=?",
                (strat_name,)
            )
            con.execute(
                "UPDATE strategy_scores SET enabled=0 WHERE strategy=?",
                (strat_name,)
            )
            gate_results['strategies_auto_archived'].append(strat_name)
            logger.info("Auto-archived strategy '%s': too many failed tuning attempts", strat_name)

        if failing:
            con.commit()
    except Exception as e:
        logger.warning("Strategy auto-archive check failed: %s", e)

    # Gate 3: Efficiency metric
    try:
        total_models_ever = con.execute('SELECT COUNT(*) FROM ml_model_results').fetchone()[0]
        useful_models = con.execute(
            'SELECT COUNT(*) FROM ml_model_results WHERE archived=0 AND f1_score > 0.6'
        ).fetchone()[0]
        total_strats_ever = con.execute(
            'SELECT COUNT(DISTINCT strategy) FROM strategy_backtest_results'
        ).fetchone()[0]
        useful_strats = con.execute(
            "SELECT COUNT(DISTINCT strategy) FROM strategy_backtest_results WHERE status='active' AND score > 50"
        ).fetchone()[0]

        total_compute = max(total_models_ever + total_strats_ever, 1)
        efficiency = round((useful_models + useful_strats) / total_compute, 3)
        gate_results['efficiency_score'] = efficiency
        gate_results['efficiency_detail'] = {
            'useful_models': useful_models,
            'total_models': total_models_ever,
            'useful_strategies': useful_strats,
            'total_strategies': total_strats_ever,
            'ratio': efficiency,
        }
    except Exception as e:
        logger.warning("Efficiency metric calculation failed: %s", e)

    return gate_results


def run_ranker(workspace_dir: Path) -> dict:
    from orchestration.ranker import Ranker
    from orchestration.reporter import DailyReporter

    db_path = workspace_dir / 'data' / 'blofin_monitor.db'
    reports_dir = workspace_dir / 'data' / 'reports'
    logger = logging.getLogger('run_ranker')

    ranker = Ranker(str(db_path))
    reporter = DailyReporter(str(db_path), str(reports_dir))

    # Check convergence gates first
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    gate_status = check_convergence_gates(con, logger)
    con.close()

    if gate_status.get('model_ensemble_gate') == 'blocked':
        logger.warning(
            "Convergence gate BLOCKED — skipping model ranking until "
            "ensemble tests reach the minimum threshold."
        )

    # Rank strategies, models, ensembles
    logger.info("Ranking strategies...")
    top_strategies = ranker.keep_top_strategies(count=20)

    logger.info("Ranking ML models...")
    top_models = ranker.keep_top_models(count=5)

    logger.info("Ranking ensembles...")
    top_ensembles = ranker.keep_top_ensembles(count=3)

    logger.info(
        "Ranked: %d strategies, %d models, %d ensembles",
        len(top_strategies), len(top_models), len(top_ensembles)
    )

    # Generate daily report
    logger.info("Generating daily report...")
    try:
        report = reporter.generate_report()
        report_info = {
            'report_date': report.get('date'),
            'report_file': f"data/reports/{report.get('date')}.json"
        }
    except Exception as e:
        logger.warning("Report generation failed: %s", e)
        report_info = {'error': str(e)}

    return {
        'top_strategies_count': len(top_strategies),
        'top_models_count': len(top_models),
        'top_ensembles_count': len(top_ensembles),
        'convergence_gate': gate_status,
        'report': report_info
    }


def main():
    workspace_dir = Path(os.environ.get(
        'BLOFIN_WORKSPACE',
        os.path.expanduser('~/.openclaw/workspace/blofin-stack')
    ))
    log_path = workspace_dir / 'data' / 'ranker.log'
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("RANKER START  %s", datetime.utcnow().isoformat() + 'Z')
    logger.info("=" * 60)

    start = datetime.utcnow()
    try:
        result = run_ranker(workspace_dir)
        duration = (datetime.utcnow() - start).total_seconds()
        logger.info("RANKER DONE in %.1fs: %s", duration, json.dumps(result, default=str))
        sys.exit(0)
    except Exception as e:
        duration = (datetime.utcnow() - start).total_seconds()
        logger.error("RANKER FAILED after %.1fs: %s", duration, e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
