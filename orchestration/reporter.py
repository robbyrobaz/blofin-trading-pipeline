#!/usr/bin/env python3
"""
Daily report generator for the blofin-stack pipeline.
Aggregates results from all components and generates human-readable summaries.
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


class DailyReporter:
    def __init__(self, db_path: str, reports_dir: str):
        self.db_path = db_path
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con
    
    def _get_today_activity(self, con: sqlite3.Connection) -> Dict[str, int]:
        """Count activities from the last 24 hours."""
        cutoff_ms = int((datetime.utcnow() - timedelta(hours=24)).timestamp() * 1000)
        
        # Strategies designed (new entries in strategy_configs)
        strategies_designed = con.execute('''
            SELECT COUNT(*) as cnt FROM strategy_configs WHERE ts_ms > ?
        ''', (cutoff_ms,)).fetchone()['cnt']
        
        # Strategies tuned (updates with source='tuner')
        strategies_tuned = con.execute('''
            SELECT COUNT(*) as cnt FROM strategy_configs 
            WHERE ts_ms > ? AND source = 'tuner'
        ''', (cutoff_ms,)).fetchone()['cnt']
        
        # Strategies archived (ranking history with action='archive')
        strategies_archived = con.execute('''
            SELECT COUNT(DISTINCT entity_name) as cnt FROM ranking_history 
            WHERE ts_ms > ? AND entity_type = 'strategy' AND action = 'archive'
        ''', (cutoff_ms,)).fetchone()['cnt']
        
        # Models trained
        models_trained = con.execute('''
            SELECT COUNT(*) as cnt FROM ml_model_results WHERE ts_ms > ?
        ''', (cutoff_ms,)).fetchone()['cnt']
        
        # Models archived
        models_archived = con.execute('''
            SELECT COUNT(*) as cnt FROM ml_model_results 
            WHERE ts_ms > ? AND archived = 1
        ''', (cutoff_ms,)).fetchone()['cnt']
        
        # Ensembles tested
        ensembles_tested = con.execute('''
            SELECT COUNT(*) as cnt FROM ml_ensembles WHERE ts_ms > ?
        ''', (cutoff_ms,)).fetchone()['cnt']
        
        return {
            'strategies_designed': strategies_designed,
            'strategies_tuned': strategies_tuned,
            'strategies_archived': strategies_archived,
            'models_trained': models_trained,
            'models_archived': models_archived,
            'ensembles_tested': ensembles_tested
        }
    
    def _get_top_strategies(self, con: sqlite3.Connection, limit: int = 5) -> List[Dict[str, Any]]:
        """Get current top N strategies by score."""
        cursor = con.execute('''
            SELECT 
                strategy,
                score,
                win_rate,
                sharpe_ratio,
                total_pnl_pct,
                trades,
                window,
                ts_iso
            FROM strategy_scores
            WHERE score IS NOT NULL AND enabled = 1
            ORDER BY score DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_top_models(self, con: sqlite3.Connection, limit: int = 5) -> List[Dict[str, Any]]:
        """Get current top N models by F1 score."""
        cursor = con.execute('''
            SELECT 
                model_name,
                model_type,
                f1_score,
                test_accuracy,
                roc_auc,
                ts_iso
            FROM ml_model_results
            WHERE archived = 0 AND f1_score IS NOT NULL
            ORDER BY f1_score DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_performance_trends(self, con: sqlite3.Connection) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        # Get average scores from last 7 days
        cutoff_ms = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
        
        trend_data = con.execute('''
            SELECT 
                DATE(ts_iso) as date,
                AVG(score) as avg_score,
                AVG(win_rate) as avg_win_rate,
                COUNT(DISTINCT strategy) as active_strategies
            FROM strategy_scores
            WHERE ts_ms > ? AND enabled = 1
            GROUP BY DATE(ts_iso)
            ORDER BY date DESC
        ''', (cutoff_ms,)).fetchall()
        
        trends = [dict(row) for row in trend_data]
        
        # Calculate trend direction
        if len(trends) >= 2:
            recent_avg = trends[0]['avg_score'] or 0
            older_avg = trends[-1]['avg_score'] or 0
            trend_direction = 'improving' if recent_avg > older_avg else 'declining' if recent_avg < older_avg else 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        return {
            'daily_averages': trends,
            'trend_direction': trend_direction,
            'days_analyzed': len(trends)
        }
    
    def _get_portfolio_health(self, con: sqlite3.Connection) -> Dict[str, Any]:
        """Assess overall portfolio health."""
        # Count active strategies
        active_strategies = con.execute('''
            SELECT COUNT(DISTINCT strategy) as cnt FROM strategy_scores WHERE enabled = 1
        ''').fetchone()['cnt']
        
        # Count active models
        active_models = con.execute('''
            SELECT COUNT(DISTINCT model_name) as cnt FROM ml_model_results WHERE archived = 0
        ''').fetchone()['cnt']
        
        # Average portfolio score
        avg_score = con.execute('''
            SELECT AVG(score) as avg FROM strategy_scores WHERE enabled = 1 AND score IS NOT NULL
        ''').fetchone()['avg']
        
        # Recent trade performance
        recent_trades = con.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as winning_trades,
                AVG(pnl_pct) as avg_pnl
            FROM paper_trades
            WHERE status = 'CLOSED' AND closed_ts_ms > ?
        ''', (int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000),)).fetchone()
        
        win_rate = 0
        if recent_trades['total_trades'] > 0:
            win_rate = (recent_trades['winning_trades'] / recent_trades['total_trades']) * 100
        
        # Health assessment
        health_score = 0
        if active_strategies >= 10:
            health_score += 25
        if active_models >= 3:
            health_score += 25
        if avg_score and avg_score > 0.6:
            health_score += 25
        if win_rate > 55:
            health_score += 25
        
        health_status = 'excellent' if health_score >= 75 else 'good' if health_score >= 50 else 'fair' if health_score >= 25 else 'poor'
        
        return {
            'active_strategies': active_strategies,
            'active_models': active_models,
            'avg_portfolio_score': round(avg_score, 3) if avg_score else None,
            'recent_win_rate': round(win_rate, 2),
            'recent_trades': recent_trades['total_trades'],
            'health_score': health_score,
            'health_status': health_status
        }
    
    def generate_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive daily report.
        
        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.
        
        Returns:
            Dict containing full report structure
        """
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        
        con = self._connect()
        try:
            # Gather all data
            activity = self._get_today_activity(con)
            top_strategies = self._get_top_strategies(con, limit=5)
            top_models = self._get_top_models(con, limit=5)
            trends = self._get_performance_trends(con)
            health = self._get_portfolio_health(con)
            
            # Build summary text
            summary_lines = [
                f"Daily Report for {date}",
                f"",
                f"Activity Summary:",
                f"  - Strategies designed: {activity['strategies_designed']}",
                f"  - Strategies tuned: {activity['strategies_tuned']}",
                f"  - Strategies archived: {activity['strategies_archived']}",
                f"  - ML models trained: {activity['models_trained']}",
                f"  - ML models archived: {activity['models_archived']}",
                f"  - Ensembles tested: {activity['ensembles_tested']}",
                f"",
                f"Portfolio Health: {health['health_status'].upper()} ({health['health_score']}/100)",
                f"  - Active strategies: {health['active_strategies']}",
                f"  - Active models: {health['active_models']}",
                f"  - Avg portfolio score: {health['avg_portfolio_score']}",
                f"  - Recent win rate: {health['recent_win_rate']}%",
                f"",
                f"Top 5 Strategies:",
            ]
            
            for i, strat in enumerate(top_strategies, 1):
                summary_lines.append(
                    f"  {i}. {strat['strategy']}: score={strat['score']:.3f}, "
                    f"win_rate={strat['win_rate']:.1f}%, sharpe={strat['sharpe_ratio']:.2f}"
                )
            
            summary_lines.append("")
            summary_lines.append("Top 5 Models:")
            for i, model in enumerate(top_models, 1):
                summary_lines.append(
                    f"  {i}. {model['model_name']} ({model['model_type']}): "
                    f"F1={model['f1_score']:.3f}, acc={model['test_accuracy']:.3f}"
                )
            
            summary_lines.append("")
            summary_lines.append(f"Performance Trend: {trends['trend_direction']}")
            
            summary = '\n'.join(summary_lines)
            
            # Build full report structure
            report = {
                'date': date,
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'summary': summary,
                'activity': activity,
                'top_strategies': top_strategies,
                'top_models': top_models,
                'performance_trends': trends,
                'portfolio_health': health,
                'ai_review': None  # Will be populated by AI reviewer
            }
            
            # Save to database
            ts_ms = int(datetime.utcnow().timestamp() * 1000)
            ts_iso = datetime.utcnow().isoformat() + 'Z'
            
            con.execute('''
                INSERT INTO daily_reports 
                (ts_ms, ts_iso, date, report_type, summary, 
                 strategies_designed, strategies_tuned, strategies_archived,
                 models_trained, models_archived, ensembles_tested,
                 top_strategies_json, top_models_json, performance_trends_json, full_report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, report_type) DO UPDATE SET
                    ts_ms=excluded.ts_ms,
                    ts_iso=excluded.ts_iso,
                    summary=excluded.summary,
                    strategies_designed=excluded.strategies_designed,
                    strategies_tuned=excluded.strategies_tuned,
                    strategies_archived=excluded.strategies_archived,
                    models_trained=excluded.models_trained,
                    models_archived=excluded.models_archived,
                    ensembles_tested=excluded.ensembles_tested,
                    top_strategies_json=excluded.top_strategies_json,
                    top_models_json=excluded.top_models_json,
                    performance_trends_json=excluded.performance_trends_json,
                    full_report_json=excluded.full_report_json
            ''', (
                ts_ms, ts_iso, date, 'daily', summary,
                activity['strategies_designed'], activity['strategies_tuned'], activity['strategies_archived'],
                activity['models_trained'], activity['models_archived'], activity['ensembles_tested'],
                json.dumps(top_strategies), json.dumps(top_models), json.dumps(trends), json.dumps(report)
            ))
            
            con.commit()
            
            # Save JSON report to file
            report_file = self.reports_dir / f"{date}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Report saved to {report_file}")
            return report
            
        finally:
            con.close()
    
    def update_ai_review(self, date: str, ai_review: Dict[str, Any]):
        """
        Update report with AI review results.
        
        Args:
            date: Date string (YYYY-MM-DD)
            ai_review: Dict containing AI's analysis and recommendations
        """
        con = self._connect()
        try:
            con.execute('''
                UPDATE daily_reports 
                SET ai_review_json = ?
                WHERE date = ? AND report_type = 'daily'
            ''', (json.dumps(ai_review), date))
            
            con.commit()
            
            # Also update the JSON file
            report_file = self.reports_dir / f"{date}.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                report['ai_review'] = ai_review
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
            
        finally:
            con.close()


if __name__ == '__main__':
    # Quick test
    import os
    db_path = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/blofin_monitor.db')
    reports_dir = os.path.expanduser('~/.openclaw/workspace/blofin-stack/data/reports')
    
    reporter = DailyReporter(db_path, reports_dir)
    print("Generating test report...")
    report = reporter.generate_report()
    print("\n" + report['summary'])
    print("\nReporter test complete!")
