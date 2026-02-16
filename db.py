#!/usr/bin/env python3
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL;')
    con.execute('PRAGMA synchronous=NORMAL;')
    con.execute('PRAGMA temp_store=MEMORY;')
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(
        '''
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            source TEXT DEFAULT 'blofin_ws',
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            strategy TEXT NOT NULL,
            confidence REAL,
            price REAL NOT NULL,
            details_json TEXT
        );

        CREATE TABLE IF NOT EXISTS confirmed_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            score REAL NOT NULL,
            rationale TEXT,
            UNIQUE(signal_id)
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            confirmed_signal_id INTEGER NOT NULL,
            opened_ts_ms INTEGER NOT NULL,
            opened_ts_iso TEXT NOT NULL,
            closed_ts_ms INTEGER,
            closed_ts_iso TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            qty REAL NOT NULL DEFAULT 1.0,
            status TEXT NOT NULL DEFAULT 'OPEN',
            pnl_pct REAL,
            reason TEXT,
            UNIQUE(confirmed_signal_id)
        );

        CREATE TABLE IF NOT EXISTS service_heartbeats (
            service TEXT PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            details_json TEXT
        );

        CREATE TABLE IF NOT EXISTS gap_fill_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            symbol TEXT NOT NULL,
            gaps_found INTEGER NOT NULL,
            rows_inserted INTEGER NOT NULL,
            first_gap_ts_ms INTEGER,
            last_gap_ts_ms INTEGER,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS dashboard_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            ok INTEGER NOT NULL,
            status_code INTEGER,
            latency_ms INTEGER,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_signals_signal_ts ON signals(signal, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_confirmed_symbol_ts ON confirmed_signals(symbol, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_paper_symbol_status ON paper_trades(symbol, status);
        
        CREATE TABLE IF NOT EXISTS strategy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            strategy TEXT NOT NULL,
            symbol TEXT,
            window TEXT NOT NULL,
            trades INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            avg_pnl_pct REAL,
            total_pnl_pct REAL,
            sharpe_ratio REAL,
            max_drawdown_pct REAL,
            score REAL,
            enabled INTEGER DEFAULT 1
        );
        
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            category TEXT NOT NULL,
            strategy TEXT,
            symbol TEXT,
            content TEXT NOT NULL,
            source TEXT,
            metadata TEXT
        );
        
        CREATE TABLE IF NOT EXISTS strategy_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            strategy TEXT NOT NULL,
            config_json TEXT NOT NULL,
            source TEXT,
            note TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_strategy_scores_strategy ON strategy_scores(strategy, window);
        CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category, strategy);
        CREATE INDEX IF NOT EXISTS idx_strategy_configs_strategy ON strategy_configs(strategy, ts_ms);
        
        CREATE TABLE IF NOT EXISTS strategy_backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            backtest_window_days INTEGER NOT NULL,
            total_trades INTEGER,
            win_rate REAL,
            sharpe_ratio REAL,
            max_drawdown_pct REAL,
            total_pnl_pct REAL,
            avg_pnl_pct REAL,
            score REAL,
            config_json TEXT,
            metrics_json TEXT,
            tuning_attempt INTEGER DEFAULT 0,
            design_prompt TEXT,
            status TEXT DEFAULT 'active'
        );
        
        CREATE TABLE IF NOT EXISTS ml_model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            symbol TEXT,
            features_json TEXT,
            train_accuracy REAL,
            test_accuracy REAL,
            f1_score REAL,
            precision_score REAL,
            recall_score REAL,
            roc_auc REAL,
            config_json TEXT,
            metrics_json TEXT,
            archived INTEGER DEFAULT 0,
            archive_reason TEXT
        );
        
        CREATE TABLE IF NOT EXISTS ml_ensembles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            ensemble_name TEXT NOT NULL,
            model_ids_json TEXT NOT NULL,
            symbol TEXT,
            test_accuracy REAL,
            f1_score REAL,
            voting_method TEXT,
            config_json TEXT,
            metrics_json TEXT,
            archived INTEGER DEFAULT 0,
            archive_reason TEXT,
            weights_json TEXT
        );
        
        CREATE TABLE IF NOT EXISTS daily_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            date TEXT NOT NULL,
            report_type TEXT DEFAULT 'daily',
            summary TEXT,
            strategies_designed INTEGER,
            strategies_tuned INTEGER,
            strategies_archived INTEGER,
            models_trained INTEGER,
            models_archived INTEGER,
            ensembles_tested INTEGER,
            top_strategies_json TEXT,
            top_models_json TEXT,
            performance_trends_json TEXT,
            ai_review_json TEXT,
            full_report_json TEXT,
            strategy_changes_json TEXT,
            model_changes_json TEXT,
            UNIQUE(date, report_type)
        );
        
        CREATE TABLE IF NOT EXISTS ranking_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            ts_iso TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_name TEXT NOT NULL,
            rank INTEGER,
            score REAL,
            metric_name TEXT,
            metric_value REAL,
            action TEXT,
            reason TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_backtest_strategy_ts ON strategy_backtest_results(strategy, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_ml_model_name ON ml_model_results(model_name, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_ml_ensemble_name ON ml_ensembles(ensemble_name, ts_ms);
        CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON daily_reports(date);
        CREATE INDEX IF NOT EXISTS idx_ranking_entity ON ranking_history(entity_type, entity_name, ts_ms);
        '''
    )
    con.commit()


def insert_tick(con: sqlite3.Connection, row: Dict[str, Any]) -> None:
    con.execute(
        'INSERT INTO ticks (ts_ms, ts_iso, symbol, price, source, raw_json) VALUES (?, ?, ?, ?, ?, ?)',
        (row['ts_ms'], row['ts_iso'], row['symbol'], row['price'], row.get('source', 'blofin_ws'), row.get('raw_json')),
    )


def insert_signal(con: sqlite3.Connection, row: Dict[str, Any]) -> None:
    con.execute(
        'INSERT INTO signals (ts_ms, ts_iso, symbol, signal, strategy, confidence, price, details_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (row['ts_ms'], row['ts_iso'], row['symbol'], row['signal'], row['strategy'], row.get('confidence'), row['price'], row.get('details_json')),
    )


def upsert_heartbeat(con: sqlite3.Connection, service: str, ts_ms: int, ts_iso: str, details_json: str = '{}') -> None:
    con.execute(
        '''
        INSERT INTO service_heartbeats (service, ts_ms, ts_iso, details_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(service) DO UPDATE SET
            ts_ms=excluded.ts_ms,
            ts_iso=excluded.ts_iso,
            details_json=excluded.details_json
        ''',
        (service, ts_ms, ts_iso, details_json),
    )


def latest_ticks(con: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    cur = con.execute('SELECT ts_iso, symbol, price, source FROM ticks ORDER BY ts_ms DESC LIMIT ?', (limit,))
    return [dict(r) for r in cur.fetchall()]


def latest_signals(con: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    cur = con.execute('SELECT ts_iso, symbol, signal, strategy, confidence, price, details_json FROM signals ORDER BY ts_ms DESC LIMIT ?', (limit,))
    return [dict(r) for r in cur.fetchall()]


# ==================== NEW HELPER FUNCTIONS ====================

def insert_backtest_result(con: sqlite3.Connection, result: Dict[str, Any]) -> int:
    """Insert strategy backtest result and return the ID."""
    now_ms = int(datetime.now().timestamp() * 1000)
    now_iso = datetime.now().isoformat()
    
    cur = con.execute(
        '''INSERT INTO strategy_backtest_results 
        (ts_ms, ts_iso, strategy, symbol, backtest_window_days, total_trades, win_rate, 
         sharpe_ratio, max_drawdown_pct, total_pnl_pct, avg_pnl_pct, score, config_json, 
         metrics_json, tuning_attempt, design_prompt, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            now_ms, now_iso,
            result['strategy'],
            result['symbol'],
            result.get('backtest_window_days', 30),
            result.get('total_trades', 0),
            result.get('win_rate', 0.0),
            result.get('sharpe_ratio', 0.0),
            result.get('max_drawdown_pct', 0.0),
            result.get('total_pnl_pct', 0.0),
            result.get('avg_pnl_pct', 0.0),
            result.get('score', 0.0),
            json.dumps(result.get('config', {})),
            json.dumps(result.get('metrics', {})),
            result.get('tuning_attempt', 0),
            result.get('design_prompt', ''),
            result.get('status', 'active')
        )
    )
    con.commit()
    return cur.lastrowid


def insert_ml_model_result(con: sqlite3.Connection, result: Dict[str, Any]) -> int:
    """Insert ML model training result and return the ID."""
    now_ms = int(datetime.now().timestamp() * 1000)
    now_iso = datetime.now().isoformat()
    
    cur = con.execute(
        '''INSERT INTO ml_model_results
        (ts_ms, ts_iso, model_name, model_type, symbol, features_json, train_accuracy,
         test_accuracy, f1_score, precision_score, recall_score, roc_auc, config_json,
         metrics_json, archived, archive_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            now_ms, now_iso,
            result['model_name'],
            result['model_type'],
            result.get('symbol', ''),
            json.dumps(result.get('features', [])),
            result.get('train_accuracy', 0.0),
            result.get('test_accuracy', 0.0),
            result.get('f1_score', 0.0),
            result.get('precision_score', 0.0),
            result.get('recall_score', 0.0),
            result.get('roc_auc', 0.0),
            json.dumps(result.get('config', {})),
            json.dumps(result.get('metrics', {})),
            0,
            ''
        )
    )
    con.commit()
    return cur.lastrowid


def insert_ensemble_result(con: sqlite3.Connection, result: Dict[str, Any]) -> int:
    """Insert ensemble result and return the ID."""
    now_ms = int(datetime.now().timestamp() * 1000)
    now_iso = datetime.now().isoformat()
    
    cur = con.execute(
        '''INSERT INTO ml_ensembles
        (ts_ms, ts_iso, ensemble_name, model_ids_json, symbol, test_accuracy, f1_score,
         voting_method, config_json, metrics_json, weights_json, archived, archive_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            now_ms, now_iso,
            result['ensemble_name'],
            json.dumps(result.get('model_ids', [])),
            result.get('symbol', ''),
            result.get('test_accuracy', 0.0),
            result.get('f1_score', 0.0),
            result.get('voting_method', 'soft'),
            json.dumps(result.get('config', {})),
            json.dumps(result.get('metrics', {})),
            json.dumps(result.get('weights', [])),
            0,
            ''
        )
    )
    con.commit()
    return cur.lastrowid


def insert_daily_report(con: sqlite3.Connection, report: Dict[str, Any]) -> int:
    """Insert or update daily report."""
    now_ms = int(datetime.now().timestamp() * 1000)
    now_iso = datetime.now().isoformat()
    
    cur = con.execute(
        '''INSERT INTO daily_reports
        (ts_ms, ts_iso, date, report_type, summary, strategies_designed, strategies_tuned,
         strategies_archived, models_trained, models_archived, ensembles_tested,
         top_strategies_json, top_models_json, performance_trends_json, ai_review_json,
         full_report_json, strategy_changes_json, model_changes_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ai_review_json=excluded.ai_review_json,
            full_report_json=excluded.full_report_json,
            strategy_changes_json=excluded.strategy_changes_json,
            model_changes_json=excluded.model_changes_json
        ''',
        (
            now_ms, now_iso,
            report['date'],
            report.get('report_type', 'daily'),
            report.get('summary', ''),
            report.get('strategies_designed', 0),
            report.get('strategies_tuned', 0),
            report.get('strategies_archived', 0),
            report.get('models_trained', 0),
            report.get('models_archived', 0),
            report.get('ensembles_tested', 0),
            json.dumps(report.get('top_strategies', [])),
            json.dumps(report.get('top_models', [])),
            json.dumps(report.get('performance_trends', {})),
            json.dumps(report.get('ai_review', {})),
            json.dumps(report.get('full_report', {})),
            json.dumps(report.get('strategy_changes', [])),
            json.dumps(report.get('model_changes', []))
        )
    )
    con.commit()
    return cur.lastrowid


def query_top_strategies(con: sqlite3.Connection, limit: int = 10, status: str = 'active') -> List[Dict[str, Any]]:
    """Query top-performing strategies by score."""
    cur = con.execute(
        '''SELECT strategy, symbol, score, win_rate, sharpe_ratio, total_pnl_pct, total_trades
        FROM strategy_backtest_results
        WHERE status = ?
        ORDER BY score DESC, ts_ms DESC
        LIMIT ?''',
        (status, limit)
    )
    return [dict(r) for r in cur.fetchall()]


def query_top_models(con: sqlite3.Connection, limit: int = 10, archived: int = 0) -> List[Dict[str, Any]]:
    """Query top-performing ML models by test accuracy."""
    cur = con.execute(
        '''SELECT model_name, model_type, test_accuracy, f1_score, precision_score, recall_score
        FROM ml_model_results
        WHERE archived = ?
        ORDER BY test_accuracy DESC, f1_score DESC, ts_ms DESC
        LIMIT ?''',
        (archived, limit)
    )
    return [dict(r) for r in cur.fetchall()]


def archive_strategy(con: sqlite3.Connection, strategy_name: str, reason: str = '') -> None:
    """Mark a strategy as archived."""
    con.execute(
        '''UPDATE strategy_backtest_results
        SET status = 'archived'
        WHERE strategy = ?''',
        (strategy_name,)
    )
    
    # Record in ranking history
    now_ms = int(datetime.now().timestamp() * 1000)
    now_iso = datetime.now().isoformat()
    con.execute(
        '''INSERT INTO ranking_history
        (ts_ms, ts_iso, entity_type, entity_name, action, reason)
        VALUES (?, ?, 'strategy', ?, 'archive', ?)''',
        (now_ms, now_iso, strategy_name, reason)
    )
    con.commit()


def archive_model(con: sqlite3.Connection, model_name: str, reason: str = '') -> None:
    """Mark an ML model as archived."""
    con.execute(
        '''UPDATE ml_model_results
        SET archived = 1, archive_reason = ?
        WHERE model_name = ?''',
        (reason, model_name)
    )
    
    # Record in ranking history
    now_ms = int(datetime.now().timestamp() * 1000)
    now_iso = datetime.now().isoformat()
    con.execute(
        '''INSERT INTO ranking_history
        (ts_ms, ts_iso, entity_type, entity_name, action, reason)
        VALUES (?, ?, 'model', ?, 'archive', ?)''',
        (now_ms, now_iso, model_name, reason)
    )
    con.commit()
