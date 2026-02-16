#!/usr/bin/env python3
"""
Daily pipeline orchestrator for blofin-stack.
Coordinates all components: scoring, designing, tuning, ML training, ranking, reporting.
"""
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import json
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.ranker import Ranker
from orchestration.reporter import DailyReporter
from orchestration.strategy_designer import StrategyDesigner
from orchestration.strategy_tuner import StrategyTuner
from ml_pipeline.train import TrainingPipeline
from ml_pipeline.db_connector import MLDatabaseConnector
from features.feature_manager import FeatureManager


class DailyRunner:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.db_path = self.workspace_dir / 'data' / 'blofin_monitor.db'
        self.log_path = self.workspace_dir / 'data' / 'pipeline.log'
        
        # Initialize components
        self.ranker = Ranker(str(self.db_path))
        self.reporter = DailyReporter(
            str(self.db_path), 
            str(self.workspace_dir / 'data' / 'reports')
        )
        self.designer = StrategyDesigner(
            str(self.db_path),
            str(self.workspace_dir / 'strategies')
        )
        self.tuner = StrategyTuner(
            str(self.db_path),
            str(self.workspace_dir / 'strategies')
        )
        
        # Setup logging
        self._setup_logging()
        
        # Pipeline state
        self.results = {}
    
    def _setup_logging(self):
        """Configure logging to file and console."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DailyRunner')
    
    def _log_step(self, step_name: str, status: str, details: Optional[Dict] = None):
        """Log pipeline step with structured data."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'step': step_name,
            'status': status,
            'details': details or {}
        }
        
        if status == 'success':
            self.logger.info(f"✓ {step_name}: {json.dumps(details)}")
        elif status == 'failure':
            self.logger.error(f"✗ {step_name}: {json.dumps(details)}")
        else:
            self.logger.info(f"⋯ {step_name}: {status}")
    
    def step_score_strategies(self) -> Dict[str, Any]:
        """Step 1: Score all strategies (2 min, Haiku)."""
        self._log_step('score_strategies', 'started')
        start_time = datetime.utcnow()
        
        try:
            # Call strategy scoring module (stub for now)
            # In production, this would call strategy_manager.py or similar
            self.logger.info("Scoring strategies (STUB - will integrate with strategy_manager)")
            
            result = {
                'scored_count': 0,
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('score_strategies', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('score_strategies', 'failure', {'error': str(e)})
            return {'error': str(e)}
    
    def step_design_strategies(self) -> Dict[str, Any]:
        """Step 2: Design new strategies with Opus (45 min)."""
        self._log_step('design_strategies', 'started')
        start_time = datetime.utcnow()
        
        try:
            # Design 2-3 new strategies
            designs = []
            for i in range(2):
                self.logger.info(f"Designing strategy {i+1}/2...")
                result = self.designer.design_new_strategy()
                if result:
                    designs.append(result)
            
            result = {
                'strategies_designed': len(designs),
                'strategy_names': [d['strategy_name'] for d in designs],
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('design_strategies', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('design_strategies', 'failure', {'error': str(e)})
            return {'error': str(e), 'strategies_designed': 0}
    
    def step_backtest_new_strategies(self, designed_strategies: List[str]) -> Dict[str, Any]:
        """Step 2b: Backtest newly designed strategies using real tick data."""
        self._log_step('backtest_strategies', 'started')
        start_time = datetime.utcnow()
        
        try:
            from backtester.backtest_engine import BacktestEngine
            import importlib
            
            backtest_results = []
            symbols = ['BTC-USDT', 'ETH-USDT']
            
            for strat_name in designed_strategies:
                # Try to load strategy module
                strategy_obj = None
                try:
                    # Check numbered strategies and named strategies
                    strategies_dir = self.workspace_dir / 'strategies'
                    for py_file in strategies_dir.glob('*.py'):
                        if py_file.name.startswith('__'):
                            continue
                        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                        mod = importlib.util.module_from_spec(spec)
                        try:
                            spec.loader.exec_module(mod)
                        except Exception:
                            continue
                        # Look for strategy class with matching name
                        for attr_name in dir(mod):
                            attr = getattr(mod, attr_name)
                            if hasattr(attr, 'detect') and hasattr(attr, 'name'):
                                try:
                                    inst = attr() if callable(attr) else attr
                                    if getattr(inst, 'name', '') == strat_name:
                                        strategy_obj = inst
                                        break
                                except Exception:
                                    continue
                        if strategy_obj:
                            break
                except Exception as e:
                    self.logger.warning(f"Could not load strategy {strat_name}: {e}")
                    continue
                
                if not strategy_obj:
                    self.logger.warning(f"Strategy {strat_name} not found, skipping backtest")
                    continue
                
                # Run backtest on each symbol with real tick data
                for symbol in symbols:
                    try:
                        engine = BacktestEngine(
                            symbol=symbol,
                            days_back=7,
                            db_path=str(self.db_path),
                            initial_capital=10000.0
                        )
                        
                        if not engine.ticks:
                            self.logger.info(f"No tick data for {symbol}, skipping")
                            continue
                        
                        bt_result = engine.run_strategy(
                            strategy_obj,
                            timeframe='5m',
                            stop_loss_pct=3.0,
                            take_profit_pct=5.0
                        )
                        
                        self.logger.info(
                            f"Backtest {strat_name}/{symbol}: "
                            f"{len(bt_result['trades'])} trades, "
                            f"final=${bt_result['final_capital']:.2f}, "
                            f"{bt_result['num_candles']} candles"
                        )
                        
                        backtest_results.append({
                            'strategy': strat_name,
                            'symbol': symbol,
                            'trades': len(bt_result['trades']),
                            'final_capital': bt_result['final_capital'],
                            'metrics': bt_result.get('metrics', {})
                        })
                        
                        # Save to strategy_backtest_results table
                        self._save_backtest_result(strat_name, symbol, bt_result)
                        
                    except Exception as e:
                        self.logger.warning(f"Backtest failed for {strat_name}/{symbol}: {e}")
            
            result = {
                'backtested_count': len(backtest_results),
                'results': backtest_results,
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('backtest_strategies', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('backtest_strategies', 'failure', {'error': str(e)})
            return {'error': str(e)}
    
    def _save_backtest_result(self, strategy_name: str, symbol: str, bt_result: Dict):
        """Save backtest result to database."""
        import sqlite3
        try:
            con = sqlite3.connect(str(self.db_path), timeout=30)
            ts_ms = int(datetime.utcnow().timestamp() * 1000)
            ts_iso = datetime.utcnow().isoformat() + 'Z'
            metrics = bt_result.get('metrics', {})
            
            con.execute('''
                INSERT OR REPLACE INTO strategy_backtest_results 
                (ts_ms, ts_iso, strategy, symbol, timeframe, days_back,
                 total_trades, win_rate, sharpe_ratio, max_drawdown_pct, 
                 total_pnl_pct, final_capital, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ts_ms, ts_iso, strategy_name, symbol, '5m', 7,
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
        except Exception as e:
            self.logger.warning(f"Failed to save backtest result: {e}")
    
    def step_tune_underperformers(self) -> Dict[str, Any]:
        """Step 3: Tune underperforming strategies with Sonnet (20 min)."""
        self._log_step('tune_strategies', 'started')
        start_time = datetime.utcnow()
        
        try:
            tuning_results = self.tuner.tune_underperformers(max_strategies=3)
            
            result = {
                'strategies_tuned': len(tuning_results),
                'strategy_names': [r['strategy_name'] for r in tuning_results],
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('tune_strategies', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('tune_strategies', 'failure', {'error': str(e)})
            return {'error': str(e), 'strategies_tuned': 0}
    
    def step_train_ml_models(self) -> Dict[str, Any]:
        """Step 4: Train ML models with Sonnet (50 min)."""
        self._log_step('train_ml_models', 'started')
        start_time = datetime.utcnow()
        
        try:
            self.logger.info("Training 5 ML models in parallel...")
            
            # Initialize ML components
            pipeline = TrainingPipeline(
                base_model_dir=str(self.workspace_dir / 'models')
            )
            db_connector = MLDatabaseConnector(str(self.db_path))
            feature_manager = FeatureManager(str(self.db_path))
            
            # Get training data
            self.logger.info("Fetching feature data for ML training...")
            try:
                # Try to get real feature data from database
                features_df = feature_manager.get_features(
                    symbol='BTC-USDT',
                    timeframe='1m',
                    lookback_bars=1000  # Use smaller dataset for speed
                )
                
                # Generate targets (simplified for initial testing)
                import pandas as pd
                import numpy as np
                
                features_df['target_direction'] = (features_df['close'].shift(-5) > features_df['close']).astype(int)
                features_df['target_risk'] = features_df['close'].rolling(20).std().fillna(0) * 100
                features_df['target_price'] = features_df['close'].shift(-5).fillna(features_df['close'])
                features_df['target_momentum'] = pd.cut(
                    features_df['close'].pct_change(),
                    bins=[-np.inf, -0.01, 0.01, np.inf],
                    labels=[0, 1, 2]
                ).astype(int)
                features_df['target_volatility'] = features_df['close'].rolling(10).std().fillna(0) / 1000
                
                # Drop rows with NaN targets
                features_df = features_df.dropna()
                
                self.logger.info(f"✓ Loaded {len(features_df)} samples from feature_manager")
                
            except Exception as e:
                # Fallback to synthetic data if feature_manager fails
                self.logger.warning(f"Feature manager failed ({e}), using synthetic data")
                features_df = pipeline.generate_synthetic_data(n_samples=1000)
            
            # Train all models in parallel
            self.logger.info(f"Training {len(pipeline.models)} ML models...")
            training_results = pipeline.train_all_models(
                features_df,
                max_workers=5
            )
            
            # Save results to database
            self.logger.info("Saving training results to database...")
            row_ids = db_connector.save_all_results(training_results)
            
            result = {
                'models_trained': training_results.get('successful', 0),
                'models_failed': training_results.get('failed', 0),
                'db_rows_saved': len(row_ids),
                'training_time': training_results.get('total_time', 0),
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('train_ml_models', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('train_ml_models', 'failure', {'error': str(e)})
            return {'error': str(e)}
    
    def step_rank_and_update(self) -> Dict[str, Any]:
        """Step 5: Rank strategies/models and update active pools (2 min)."""
        self._log_step('rank_and_update', 'started')
        start_time = datetime.utcnow()
        
        try:
            top_strategies = self.ranker.keep_top_strategies(count=20)
            top_models = self.ranker.keep_top_models(count=5)
            top_ensembles = self.ranker.keep_top_ensembles(count=3)
            
            result = {
                'top_strategies_count': len(top_strategies),
                'top_models_count': len(top_models),
                'top_ensembles_count': len(top_ensembles),
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('rank_and_update', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('rank_and_update', 'failure', {'error': str(e)})
            return {'error': str(e)}
    
    def step_generate_report(self) -> Dict[str, Any]:
        """Step 6: Generate daily report with Haiku (5 min)."""
        self._log_step('generate_report', 'started')
        start_time = datetime.utcnow()
        
        try:
            report = self.reporter.generate_report()
            
            result = {
                'report_date': report['date'],
                'report_file': f"data/reports/{report['date']}.json",
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('generate_report', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('generate_report', 'failure', {'error': str(e)})
            return {'error': str(e)}
    
    def step_ai_review(self, report_date: str) -> Dict[str, Any]:
        """Step 7: AI review with Opus (10 min)."""
        self._log_step('ai_review', 'started')
        start_time = datetime.utcnow()
        
        try:
            # Load report
            report_file = self.workspace_dir / 'data' / 'reports' / f'{report_date}.json'
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Build review prompt
            prompt = f"""You are an expert trading system analyst. Review this daily report and provide:
1. Key insights and observations
2. Risk assessment
3. Recommendations for improvement
4. Priority actions for tomorrow

DAILY REPORT:
{report['summary']}

Provide analysis in JSON format:
{{
  "insights": ["insight 1", "insight 2", ...],
  "risk_level": "low|medium|high",
  "risk_factors": ["factor 1", "factor 2", ...],
  "recommendations": ["rec 1", "rec 2", ...],
  "priority_actions": ["action 1", "action 2", ...]
}}
"""
            
            # Call Opus
            result = subprocess.run(
                ['openclaw', 'chat', '--model', 'opus', '--prompt', prompt],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Parse review
            import re
            json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
            if json_match:
                ai_review = json.loads(json_match.group(0))
            else:
                ai_review = {'raw_output': result.stdout}
            
            # Update report with AI review
            self.reporter.update_ai_review(report_date, ai_review)
            
            result = {
                'review_completed': True,
                'risk_level': ai_review.get('risk_level', 'unknown'),
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds()
            }
            
            self._log_step('ai_review', 'success', result)
            return result
            
        except Exception as e:
            self._log_step('ai_review', 'failure', {'error': str(e)})
            return {'error': str(e)}
    
    def run_parallel(self, tasks: List[tuple]) -> Dict[str, Any]:
        """Run independent tasks in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(task_func, *task_args): task_name
                for task_name, task_func, task_args in tasks
            }
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    self.logger.error(f"Task {task_name} failed: {e}")
                    results[task_name] = {'error': str(e)}
        
        return results
    
    def run_daily_pipeline(self):
        """Execute the complete daily pipeline."""
        self.logger.info("="*80)
        self.logger.info("STARTING DAILY PIPELINE")
        self.logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        self.logger.info("="*80)
        
        pipeline_start = datetime.utcnow()
        
        try:
            # Step 1: Score strategies (sequential, needed for ranking)
            self.results['scoring'] = self.step_score_strategies()
            
            # Step 2-4: Design, tune, and train can run in parallel
            parallel_tasks = [
                ('design', self.step_design_strategies, ()),
                ('tune', self.step_tune_underperformers, ()),
                ('ml_train', self.step_train_ml_models, ())
            ]
            
            parallel_results = self.run_parallel(parallel_tasks)
            self.results.update(parallel_results)
            
            # Step 2b: Backtest new strategies (sequential after design)
            if 'design' in self.results and 'strategy_names' in self.results['design']:
                self.results['backtest'] = self.step_backtest_new_strategies(
                    self.results['design']['strategy_names']
                )
            
            # Step 5: Rank and update (sequential, depends on scoring/training)
            self.results['ranking'] = self.step_rank_and_update()
            
            # Step 6: Generate report (sequential)
            self.results['report'] = self.step_generate_report()
            
            # Step 7: AI review (sequential, depends on report)
            if 'report' in self.results and 'report_date' in self.results['report']:
                self.results['ai_review'] = self.step_ai_review(
                    self.results['report']['report_date']
                )
            
            # Calculate total duration
            pipeline_duration = (datetime.utcnow() - pipeline_start).total_seconds()
            
            self.logger.info("="*80)
            self.logger.info("PIPELINE COMPLETED")
            self.logger.info(f"Total duration: {pipeline_duration:.1f} seconds ({pipeline_duration/60:.1f} minutes)")
            self.logger.info(f"Results: {json.dumps(self.results, indent=2)}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
            raise


def main():
    """Entry point for daily cron job."""
    workspace_dir = os.environ.get(
        'BLOFIN_WORKSPACE',
        os.path.expanduser('~/.openclaw/workspace/blofin-stack')
    )
    
    runner = DailyRunner(workspace_dir)
    runner.run_daily_pipeline()


if __name__ == '__main__':
    main()
