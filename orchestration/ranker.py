#!/usr/bin/env python3
"""
Dynamic ranking system for strategies, models, and ensembles.
No hard pass/fail - everything is ranked and top N are kept active.
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class Ranker:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con
    
    def _log_ranking(self, con: sqlite3.Connection, entity_type: str, entity_name: str, 
                     rank: Optional[int], score: Optional[float], metric_name: str, 
                     metric_value: Optional[float], action: str, reason: str):
        """Log all ranking decisions for auditability."""
        ts_ms = int(datetime.utcnow().timestamp() * 1000)
        ts_iso = datetime.utcnow().isoformat() + 'Z'
        
        con.execute('''
            INSERT INTO ranking_history 
            (ts_ms, ts_iso, entity_type, entity_name, rank, score, metric_name, metric_value, action, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ts_ms, ts_iso, entity_type, entity_name, rank, score, metric_name, metric_value, action, reason))
    
    def keep_top_strategies(self, count: int = 20, metric: str = 'score') -> List[Dict[str, Any]]:
        """
        Rank all strategies by specified metric and return top N.
        
        Args:
            count: Number of top strategies to keep active
            metric: Metric to rank by (score, sharpe_ratio, win_rate, total_pnl_pct)
        
        Returns:
            List of top strategy dicts with name, rank, and metric value
        """
        con = self._connect()
        try:
            # Get latest scores for each strategy
            query = f'''
                SELECT 
                    strategy,
                    MAX(ts_ms) as latest_ts,
                    {metric} as metric_value,
                    score,
                    win_rate,
                    sharpe_ratio,
                    total_pnl_pct,
                    trades,
                    window
                FROM strategy_scores
                WHERE {metric} IS NOT NULL
                GROUP BY strategy
                ORDER BY metric_value DESC
            '''
            
            cursor = con.execute(query)
            all_strategies = [dict(row) for row in cursor.fetchall()]
            
            # Rank them
            top_strategies = []
            for rank, strat in enumerate(all_strategies[:count], 1):
                strat['rank'] = rank
                strat['metric_name'] = metric
                top_strategies.append(strat)
                
                self._log_ranking(
                    con, 'strategy', strat['strategy'], rank, strat['score'],
                    metric, strat['metric_value'], 'keep_top', f'Ranked #{rank} by {metric}'
                )
            
            # Archive the rest
            for rank, strat in enumerate(all_strategies[count:], count + 1):
                self._log_ranking(
                    con, 'strategy', strat['strategy'], rank, strat['score'],
                    metric, strat['metric_value'], 'archive', 
                    f'Ranked #{rank} by {metric}, below top {count} threshold'
                )
            
            con.commit()
            return top_strategies
            
        finally:
            con.close()
    
    def keep_top_models(self, count: int = 5, metric: str = 'f1_score') -> List[Dict[str, Any]]:
        """
        Rank all ML models by specified metric and return top N.
        
        Args:
            count: Number of top models to keep active
            metric: Metric to rank by (f1_score, test_accuracy, roc_auc)
        
        Returns:
            List of top model dicts with name, rank, and metric value
        """
        con = self._connect()
        try:
            # Get latest results for each model (non-archived only)
            query = f'''
                SELECT 
                    model_name,
                    model_type,
                    MAX(ts_ms) as latest_ts,
                    {metric} as metric_value,
                    f1_score,
                    test_accuracy,
                    roc_auc,
                    precision_score,
                    recall_score
                FROM ml_model_results
                WHERE archived = 0 AND {metric} IS NOT NULL
                GROUP BY model_name
                ORDER BY metric_value DESC
            '''
            
            cursor = con.execute(query)
            all_models = [dict(row) for row in cursor.fetchall()]
            
            # Rank and keep top N
            top_models = []
            for rank, model in enumerate(all_models[:count], 1):
                model['rank'] = rank
                model['metric_name'] = metric
                top_models.append(model)
                
                self._log_ranking(
                    con, 'ml_model', model['model_name'], rank, model['metric_value'],
                    metric, model['metric_value'], 'keep_top', f'Ranked #{rank} by {metric}'
                )
            
            # Archive the rest
            for model in all_models[count:]:
                con.execute('''
                    UPDATE ml_model_results 
                    SET archived = 1, archive_reason = ?
                    WHERE model_name = ? AND archived = 0
                ''', (f'Below top {count} threshold (metric: {metric})', model['model_name']))
                
                self._log_ranking(
                    con, 'ml_model', model['model_name'], None, model['metric_value'],
                    metric, model['metric_value'], 'archive', 
                    f'Below top {count} threshold'
                )
            
            con.commit()
            return top_models
            
        finally:
            con.close()
    
    def keep_top_ensembles(self, count: int = 3, metric: str = 'test_accuracy') -> List[Dict[str, Any]]:
        """
        Rank all ensembles by specified metric and return top N.
        
        Args:
            count: Number of top ensembles to keep active
            metric: Metric to rank by (test_accuracy, f1_score)
        
        Returns:
            List of top ensemble dicts with name, rank, and metric value
        """
        con = self._connect()
        try:
            # Get latest results for each ensemble (non-archived only)
            query = f'''
                SELECT 
                    ensemble_name,
                    MAX(ts_ms) as latest_ts,
                    {metric} as metric_value,
                    test_accuracy,
                    f1_score,
                    voting_method
                FROM ml_ensembles
                WHERE archived = 0 AND {metric} IS NOT NULL
                GROUP BY ensemble_name
                ORDER BY metric_value DESC
            '''
            
            cursor = con.execute(query)
            all_ensembles = [dict(row) for row in cursor.fetchall()]
            
            # Rank and keep top N
            top_ensembles = []
            for rank, ens in enumerate(all_ensembles[:count], 1):
                ens['rank'] = rank
                ens['metric_name'] = metric
                top_ensembles.append(ens)
                
                self._log_ranking(
                    con, 'ml_ensemble', ens['ensemble_name'], rank, ens['metric_value'],
                    metric, ens['metric_value'], 'keep_top', f'Ranked #{rank} by {metric}'
                )
            
            # Archive the rest
            for ens in all_ensembles[count:]:
                con.execute('''
                    UPDATE ml_ensembles 
                    SET archived = 1, archive_reason = ?
                    WHERE ensemble_name = ? AND archived = 0
                ''', (f'Below top {count} threshold (metric: {metric})', ens['ensemble_name']))
                
                self._log_ranking(
                    con, 'ml_ensemble', ens['ensemble_name'], None, ens['metric_value'],
                    metric, ens['metric_value'], 'archive', 
                    f'Below top {count} threshold'
                )
            
            con.commit()
            return top_ensembles
            
        finally:
            con.close()
    
    def archive_bottom(self, strategy_names: List[str], reason: str) -> int:
        """
        Archive specific strategies with a reason.
        
        Args:
            strategy_names: List of strategy names to archive
            reason: Human-readable reason for archiving
        
        Returns:
            Number of strategies archived
        """
        if not strategy_names:
            return 0
        
        con = self._connect()
        try:
            archived_count = 0
            for strategy_name in strategy_names:
                # Mark as disabled in strategy_scores
                result = con.execute('''
                    UPDATE strategy_scores 
                    SET enabled = 0
                    WHERE strategy = ?
                ''', (strategy_name,))
                
                if result.rowcount > 0:
                    archived_count += 1
                    self._log_ranking(
                        con, 'strategy', strategy_name, None, None,
                        'enabled', 0.0, 'archive', reason
                    )
            
            con.commit()
            return archived_count
            
        finally:
            con.close()
    
    def get_ranking_history(self, entity_type: Optional[str] = None, 
                           entity_name: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve ranking history for analysis.
        
        Args:
            entity_type: Filter by type (strategy, ml_model, ml_ensemble)
            entity_name: Filter by specific entity name
            limit: Max records to return
        
        Returns:
            List of ranking history records
        """
        con = self._connect()
        try:
            query = 'SELECT * FROM ranking_history WHERE 1=1'
            params = []
            
            if entity_type:
                query += ' AND entity_type = ?'
                params.append(entity_type)
            
            if entity_name:
                query += ' AND entity_name = ?'
                params.append(entity_name)
            
            query += ' ORDER BY ts_ms DESC LIMIT ?'
            params.append(limit)
            
            cursor = con.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        finally:
            con.close()


if __name__ == '__main__':
    # Quick test
    import os
    db_path = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db')
    ranker = Ranker(db_path)
    
    print("Testing ranker...")
    top_strategies = ranker.keep_top_strategies(count=20)
    print(f"Top strategies: {len(top_strategies)}")
    
    top_models = ranker.keep_top_models(count=5)
    print(f"Top models: {len(top_models)}")
    
    print("Ranker test complete!")
