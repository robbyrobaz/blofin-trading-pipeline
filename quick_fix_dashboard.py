#!/usr/bin/env python3
"""Quick fix: Convert strategy_scores into strategy_backtest_results to populate dashboard."""
import sqlite3
import time
import json

DB_PATH = "data/blofin_monitor.db"

def main():
    print("="*70)
    print("QUICK FIX: Populate strategy_backtest_results from strategy_scores")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH, timeout=30)
    
    # Check current state
    cur = conn.execute("SELECT COUNT(*) FROM strategy_backtest_results")
    before_count = cur.fetchone()[0]
    print(f"\nBefore: strategy_backtest_results has {before_count} rows")
    
    cur = conn.execute("SELECT COUNT(*) FROM strategy_scores")
    scores_count = cur.fetchone()[0]
    print(f"Strategy scores available: {scores_count:,} rows")
    
    # Get latest scores for each strategy (grouped by strategy only, not symbol)
    print("\nAggregating strategy scores...")
    cur = conn.execute('''
        SELECT 
            strategy,
            'BTC-USDT' as symbol,
            COUNT(*) as total_trades,
            SUM(wins) as wins,
            SUM(losses) as losses,
            AVG(win_rate) as avg_win_rate,
            AVG(sharpe_ratio) as avg_sharpe,
            AVG(max_drawdown_pct) as avg_max_dd,
            AVG(total_pnl_pct) as avg_pnl,
            AVG(score) as avg_score,
            MAX(ts_ms) as latest_ts_ms,
            MAX(ts_iso) as latest_ts_iso
        FROM strategy_scores
        WHERE enabled = 1 AND window = 'all'
        GROUP BY strategy
        ORDER BY avg_score DESC
        LIMIT 25
    ''')
    
    strategies = cur.fetchall()
    print(f"Found {len(strategies)} active strategies to convert")
    
    # Insert as backtest results
    ts_ms = int(time.time() * 1000)
    ts_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    
    inserted = 0
    for row in strategies:
        strategy, symbol, total_trades, wins, losses, avg_win_rate, avg_sharpe, avg_max_dd, avg_pnl, avg_score, latest_ts_ms, latest_ts_iso = row
        
        # Calculate actual win_rate from wins/losses
        if total_trades > 0:
            actual_win_rate = (wins / total_trades) * 100 if wins else 0
        else:
            actual_win_rate = avg_win_rate or 0
        
        metrics = {
            'win_rate': actual_win_rate,
            'sharpe_ratio': avg_sharpe or 0,
            'max_drawdown_pct': avg_max_dd or 0,
            'total_pnl_pct': avg_pnl or 0,
            'total_trades': total_trades,
            'wins': wins or 0,
            'losses': losses or 0,
        }
        
        # Check if already exists
        existing = conn.execute('''
            SELECT COUNT(*) FROM strategy_backtest_results 
            WHERE strategy = ? AND symbol = ?
        ''', (strategy, symbol)).fetchone()[0]
        
        if existing > 0:
            # Update existing
            conn.execute('''
                UPDATE strategy_backtest_results
                SET ts_ms = ?, ts_iso = ?, total_trades = ?, win_rate = ?, 
                    sharpe_ratio = ?, max_drawdown_pct = ?, total_pnl_pct = ?, 
                    score = ?, metrics_json = ?
                WHERE strategy = ? AND symbol = ?
            ''', (
                ts_ms, ts_iso, total_trades, actual_win_rate,
                avg_sharpe or 0, avg_max_dd or 0, avg_pnl or 0,
                avg_score or 0, json.dumps(metrics),
                strategy, symbol
            ))
            print(f"  Updated: {strategy:20} - trades={total_trades:6,}, wr={actual_win_rate:5.1f}%, score={avg_score:5.2f}")
        else:
            # Insert new
            conn.execute('''
                INSERT INTO strategy_backtest_results 
                (ts_ms, ts_iso, strategy, symbol, backtest_window_days, total_trades, 
                 win_rate, sharpe_ratio, max_drawdown_pct, total_pnl_pct, score, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ts_ms, ts_iso, strategy, symbol, 7,  # assume 7 day window
                total_trades, actual_win_rate,
                avg_sharpe or 0, avg_max_dd or 0, avg_pnl or 0,
                avg_score or 0, json.dumps(metrics)
            ))
            print(f"  Inserted: {strategy:20} - trades={total_trades:6,}, wr={actual_win_rate:5.1f}%, score={avg_score:5.2f}")
        
        inserted += 1
    
    conn.commit()
    
    # Check after
    cur = conn.execute("SELECT COUNT(*) FROM strategy_backtest_results")
    after_count = cur.fetchone()[0]
    print(f"\nAfter: strategy_backtest_results has {after_count} rows")
    print(f"Processed {inserted} strategies")
    
    # Show top 5
    print("\nTop 5 Strategies in backtest results:")
    cur = conn.execute('''
        SELECT strategy, total_trades, win_rate, sharpe_ratio, total_pnl_pct, score
        FROM strategy_backtest_results
        ORDER BY score DESC
        LIMIT 5
    ''')
    for row in cur.fetchall():
        print(f"  {row[0]:20} - trades={row[1]:6,}, wr={row[2]:5.1f}%, sharpe={row[3]:6.2f}, pnl={row[4]:6.2f}%, score={row[5]:5.2f}")
    
    conn.close()
    print("\n" + "="*70)
    print("✓ DASHBOARD FIX COMPLETE")
    print("="*70)
    print("\nRefresh dashboard at: http://localhost:8888/blofin-dashboard.html")
    print("(Use Ctrl+Shift+R to hard refresh)")

if __name__ == '__main__':
    main()
