#!/usr/bin/env python3
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


def now_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def iso_utc(ms: int) -> str:
    """Convert timestamp to ISO format."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def compute_strategy_scores(
    con: sqlite3.Connection,
    strategy: str,
    window: str,
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute performance metrics for a strategy.
    
    Args:
        con: Database connection
        strategy: Strategy name
        window: Time window ('24h', '7d', 'all')
        symbol: Optional symbol filter (None = all symbols)
    
    Returns:
        Dictionary with performance metrics
    """
    # Calculate time cutoff
    cutoff_ms = 0
    if window == '24h':
        cutoff_ms = now_ms() - (24 * 60 * 60 * 1000)
    elif window == '7d':
        cutoff_ms = now_ms() - (7 * 24 * 60 * 60 * 1000)
    # 'all' uses cutoff_ms = 0
    
    # Build query
    query = '''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
            AVG(pnl_pct) as avg_pnl_pct,
            SUM(pnl_pct) as total_pnl_pct
        FROM paper_trades pt
        JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
        JOIN signals s ON cs.signal_id = s.id
        WHERE s.strategy = ?
            AND pt.status = 'CLOSED'
            AND pt.closed_ts_ms >= ?
    '''
    
    params = [strategy, cutoff_ms]
    
    if symbol:
        query += ' AND pt.symbol = ?'
        params.append(symbol)
    
    row = con.execute(query, params).fetchone()
    
    if not row or row['total_trades'] == 0:
        return {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'total_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'score': 0.0,
        }
    
    trades = row['total_trades']
    wins = row['wins'] or 0
    losses = row['losses'] or 0
    avg_pnl_pct = row['avg_pnl_pct'] or 0.0
    total_pnl_pct = row['total_pnl_pct'] or 0.0
    win_rate = (wins / trades) if trades > 0 else 0.0
    
    # Calculate Sharpe ratio (simplified)
    pnl_query = '''
        SELECT pnl_pct
        FROM paper_trades pt
        JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
        JOIN signals s ON cs.signal_id = s.id
        WHERE s.strategy = ?
            AND pt.status = 'CLOSED'
            AND pt.closed_ts_ms >= ?
    '''
    pnl_params = [strategy, cutoff_ms]
    if symbol:
        pnl_query += ' AND pt.symbol = ?'
        pnl_params.append(symbol)
    
    pnl_values = [r['pnl_pct'] for r in con.execute(pnl_query, pnl_params).fetchall()]
    
    sharpe_ratio = 0.0
    if len(pnl_values) > 1:
        mean_pnl = sum(pnl_values) / len(pnl_values)
        variance = sum((x - mean_pnl) ** 2 for x in pnl_values) / len(pnl_values)
        std_dev = variance ** 0.5
        if std_dev > 0:
            sharpe_ratio = mean_pnl / std_dev
    
    # Calculate max drawdown
    max_drawdown_pct = 0.0
    cumulative = 0.0
    peak = 0.0
    for pnl in pnl_values:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown
    
    # Composite score: weighted combination of metrics
    # Score components: win_rate (40%), avg_pnl (30%), sharpe (20%), drawdown penalty (10%)
    score = (
        (win_rate * 40) +
        (min(5, max(-5, avg_pnl_pct)) * 6) +  # Cap at ±5% avg PnL
        (min(3, max(-3, sharpe_ratio)) * 6.67) +  # Cap Sharpe at ±3
        (max(0, 10 - max_drawdown_pct))  # Penalty for drawdown
    )
    score = max(0, min(100, score))  # Clamp to 0-100
    
    return {
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 4),
        'avg_pnl_pct': round(avg_pnl_pct, 4),
        'total_pnl_pct': round(total_pnl_pct, 4),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown_pct': round(max_drawdown_pct, 4),
        'score': round(score, 2),
    }


def update_all_scores(con: sqlite3.Connection) -> int:
    """
    Refresh performance scores for all strategies.
    
    Args:
        con: Database connection
    
    Returns:
        Number of score records inserted
    """
    # Get list of strategies
    strategies = con.execute(
        'SELECT DISTINCT strategy FROM signals ORDER BY strategy'
    ).fetchall()
    
    ts = now_ms()
    ts_iso = iso_utc(ts)
    inserted = 0
    
    for strategy_row in strategies:
        strategy = strategy_row['strategy']
        
        for window in ['24h', '7d', 'all']:
            # Overall score
            metrics = compute_strategy_scores(con, strategy, window, symbol=None)
            
            con.execute('''
                INSERT INTO strategy_scores (
                    ts_ms, ts_iso, strategy, symbol, window,
                    trades, wins, losses, win_rate, avg_pnl_pct,
                    total_pnl_pct, sharpe_ratio, max_drawdown_pct, score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ts, ts_iso, strategy, None, window,
                metrics['trades'], metrics['wins'], metrics['losses'],
                metrics['win_rate'], metrics['avg_pnl_pct'], metrics['total_pnl_pct'],
                metrics['sharpe_ratio'], metrics['max_drawdown_pct'], metrics['score']
            ))
            inserted += 1
            
            # Per-symbol scores (only for 'all' window to reduce data)
            if window == 'all':
                symbols = con.execute(
                    '''SELECT DISTINCT pt.symbol 
                       FROM paper_trades pt
                       JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
                       JOIN signals s ON cs.signal_id = s.id
                       WHERE s.strategy = ? AND pt.status = 'CLOSED'
                    ''',
                    (strategy,)
                ).fetchall()
                
                for sym_row in symbols:
                    symbol = sym_row['symbol']
                    sym_metrics = compute_strategy_scores(con, strategy, 'all', symbol=symbol)
                    
                    if sym_metrics['trades'] > 0:
                        con.execute('''
                            INSERT INTO strategy_scores (
                                ts_ms, ts_iso, strategy, symbol, window,
                                trades, wins, losses, win_rate, avg_pnl_pct,
                                total_pnl_pct, sharpe_ratio, max_drawdown_pct, score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            ts, ts_iso, strategy, symbol, 'all',
                            sym_metrics['trades'], sym_metrics['wins'], sym_metrics['losses'],
                            sym_metrics['win_rate'], sym_metrics['avg_pnl_pct'],
                            sym_metrics['total_pnl_pct'], sym_metrics['sharpe_ratio'],
                            sym_metrics['max_drawdown_pct'], sym_metrics['score']
                        ))
                        inserted += 1
    
    con.commit()
    return inserted


def add_knowledge_entry(
    con: sqlite3.Connection,
    category: str,
    content: str,
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    source: str = 'auto',
    metadata: Optional[Dict] = None
) -> int:
    """
    Add a knowledge entry (lesson, recommendation, observation).
    
    Args:
        con: Database connection
        category: Entry category ('performance', 'lesson', 'recommendation', 'change')
        content: Entry content
        strategy: Optional strategy name
        symbol: Optional symbol
        source: Source of entry ('ai_review', 'auto_score', 'manual')
        metadata: Optional metadata dictionary
    
    Returns:
        ID of inserted entry
    """
    ts = now_ms()
    ts_iso = iso_utc(ts)
    metadata_json = json.dumps(metadata) if metadata else None
    
    cursor = con.execute('''
        INSERT INTO knowledge_entries (
            ts_ms, ts_iso, category, strategy, symbol,
            content, source, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ts, ts_iso, category, strategy, symbol, content, source, metadata_json))
    
    con.commit()
    return cursor.lastrowid


def get_knowledge_summary(con: sqlite3.Connection, limit: int = 50) -> str:
    """
    Get formatted knowledge summary for AI review.
    
    Args:
        con: Database connection
        limit: Maximum number of recent entries to include
    
    Returns:
        Formatted text summary
    """
    lines = []
    lines.append("=== STRATEGY PERFORMANCE SUMMARY ===\n")
    
    # Get latest scores for each strategy
    latest_scores = con.execute('''
        SELECT strategy, window, trades, win_rate, avg_pnl_pct, total_pnl_pct, score, enabled
        FROM strategy_scores
        WHERE symbol IS NULL
        ORDER BY strategy, 
                 CASE window 
                     WHEN '24h' THEN 1 
                     WHEN '7d' THEN 2 
                     WHEN 'all' THEN 3 
                 END
    ''').fetchall()
    
    if latest_scores:
        lines.append("Strategy Performance:")
        lines.append("-" * 80)
        current_strategy = None
        
        for row in latest_scores:
            if current_strategy != row['strategy']:
                if current_strategy is not None:
                    lines.append("")
                current_strategy = row['strategy']
                enabled_status = "ENABLED" if row['enabled'] else "DISABLED"
                lines.append(f"\n{row['strategy']} ({enabled_status}):")
            
            lines.append(
                f"  {row['window']:>4}: {row['trades']:>4} trades | "
                f"WR: {row['win_rate']*100:>5.1f}% | "
                f"Avg: {row['avg_pnl_pct']:>6.2f}% | "
                f"Total: {row['total_pnl_pct']:>7.2f}% | "
                f"Score: {row['score']:>5.1f}"
            )
    else:
        lines.append("No performance data available yet.\n")
    
    # Recent knowledge entries
    lines.append("\n\n=== RECENT KNOWLEDGE ENTRIES ===\n")
    
    entries = con.execute('''
        SELECT ts_iso, category, strategy, content, source
        FROM knowledge_entries
        ORDER BY ts_ms DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    
    if entries:
        for entry in entries:
            strategy_tag = f"[{entry['strategy']}]" if entry['strategy'] else "[SYSTEM]"
            lines.append(
                f"{entry['ts_iso'][:19]} {strategy_tag:20} [{entry['category']:15}] "
                f"{entry['content'][:80]}"
            )
    else:
        lines.append("No knowledge entries yet.\n")
    
    return "\n".join(lines)


def auto_manage_strategies(
    con: sqlite3.Connection,
    manager,
    disable_threshold: float = 30.0,
    enable_threshold: float = 50.0
) -> Dict[str, List[str]]:
    """
    Automatically enable/disable strategies based on performance scores.
    
    Args:
        con: Database connection
        manager: StrategyManager instance
        disable_threshold: Score below which to disable strategies
        enable_threshold: Score above which to re-enable strategies
    
    Returns:
        Dictionary with 'disabled' and 'enabled' lists
    """
    # Get latest 24h scores
    scores = con.execute('''
        SELECT strategy, score, trades
        FROM strategy_scores
        WHERE window = '24h' AND symbol IS NULL
        ORDER BY ts_ms DESC
    ''').fetchall()
    
    # Group by strategy (get most recent)
    strategy_scores = {}
    for row in scores:
        if row['strategy'] not in strategy_scores:
            strategy_scores[row['strategy']] = {
                'score': row['score'],
                'trades': row['trades']
            }
    
    disabled = []
    enabled = []
    
    for strategy, data in strategy_scores.items():
        # Only manage if we have enough data
        if data['trades'] < 10:
            continue
        
        # Disable poor performers
        if data['score'] < disable_threshold and manager.enabled.get(strategy, True):
            manager.disable(strategy)
            disabled.append(strategy)
            add_knowledge_entry(
                con,
                category='change',
                content=f"Auto-disabled {strategy} (score: {data['score']:.1f}, trades: {data['trades']})",
                strategy=strategy,
                source='auto_score'
            )
        
        # Re-enable good performers
        elif data['score'] >= enable_threshold and not manager.enabled.get(strategy, False):
            manager.enable(strategy)
            enabled.append(strategy)
            add_knowledge_entry(
                con,
                category='change',
                content=f"Auto-enabled {strategy} (score: {data['score']:.1f}, trades: {data['trades']})",
                strategy=strategy,
                source='auto_score'
            )
    
    con.commit()
    return {'disabled': disabled, 'enabled': enabled}


def save_strategy_config(
    con: sqlite3.Connection,
    strategy: str,
    config: Dict,
    source: str = 'manual',
    note: Optional[str] = None
) -> int:
    """
    Save strategy configuration to history.
    
    Args:
        con: Database connection
        strategy: Strategy name
        config: Configuration dictionary
        source: Source of change ('default', 'ai_review', 'manual')
        note: Optional note
    
    Returns:
        ID of inserted config record
    """
    ts = now_ms()
    ts_iso = iso_utc(ts)
    config_json = json.dumps(config)
    
    cursor = con.execute('''
        INSERT INTO strategy_configs (
            ts_ms, ts_iso, strategy, config_json, source, note
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (ts, ts_iso, strategy, config_json, source, note))
    
    con.commit()
    return cursor.lastrowid
