#!/usr/bin/env python3
"""
populate_registry_eep.py
Calculates EEP scores for all strategies in strategy_registry and populates
the bt_* (backtest) and ft_* (forward-test) columns.
"""

import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/rob/.openclaw/workspace/blofin-stack")
from eep_scorer import compute_eep_from_metrics, compute_eep_from_trades

DB_PATH = "data/blofin_monitor.db"


def get_backtest_metrics(conn):
    """Aggregate backtest metrics per strategy from strategy_scores (per-symbol rows)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT strategy,
               SUM(trades) as total_trades,
               SUM(wins) as total_wins,
               SUM(losses) as total_losses,
               SUM(win_rate * trades) / NULLIF(SUM(trades), 0) as avg_wr,
               SUM(avg_pnl_pct * trades) / NULLIF(SUM(trades), 0) as avg_pnl,
               SUM(total_pnl_pct) as sum_pnl,
               SUM(sharpe_ratio * trades) / NULLIF(SUM(trades), 0) as avg_sharpe,
               MAX(max_drawdown_pct) as worst_dd
        FROM strategy_scores
        WHERE trades > 0 AND symbol IS NOT NULL
        GROUP BY strategy
    """)
    results = {}
    for row in cur.fetchall():
        strategy, total_trades, total_wins, total_losses, avg_wr, avg_pnl, sum_pnl, avg_sharpe, worst_dd = row
        results[strategy] = {
            "strategy": strategy,
            "win_rate": avg_wr or 0.0,
            "sharpe_ratio": avg_sharpe or 0.0,
            "total_pnl_pct": sum_pnl or 0.0,
            "avg_pnl_pct": avg_pnl or 0.0,
            "max_drawdown_pct": worst_dd or 0.0,
            "num_trades": total_trades or 0,
        }
    return results


def get_forward_test_trades(conn):
    """Get closed paper trades grouped by strategy."""
    cur = conn.cursor()
    cur.execute("""
        SELECT s.strategy, pt.pnl_pct, pt.reason
        FROM paper_trades pt
        JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
        JOIN signals s ON cs.signal_id = s.id
        WHERE pt.status = 'CLOSED'
        ORDER BY s.strategy, pt.opened_ts_ms
    """)
    grouped = {}
    for strategy, pnl_pct, reason in cur.fetchall():
        if strategy not in grouped:
            grouped[strategy] = []
        # Extract exit reason from compound reason strings like
        # "ENTRY: 3 strategies agreed in 30m | EXIT: SL"
        exit_reason = reason or ""
        if "EXIT:" in exit_reason:
            # Take the part after the last "EXIT:"
            exit_reason = exit_reason.split("EXIT:")[-1].strip()
        grouped[strategy].append({"pnl_pct": pnl_pct, "reason": exit_reason})
    return grouped


def compute_ft_metrics_from_trades(trades):
    """Compute win_rate, sharpe, pnl_pct, max_dd from trade list for ft_* columns."""
    if not trades:
        return {}
    pnls = [t["pnl_pct"] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n if n > 0 else 0.0
    total_pnl = sum(pnls)

    # Sharpe
    if n >= 2:
        mean = total_pnl / n
        variance = sum((p - mean) ** 2 for p in pnls) / n
        import math
        std = math.sqrt(variance)
        sharpe = mean / std if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "win_rate": win_rate,
        "sharpe": sharpe,
        "pnl_pct": total_pnl,
        "max_dd": max_dd,
        "trades": n,
    }


def write_updates_with_retry(all_strategies, bt_results, ft_results, now_iso, max_retries=10):
    """Write updates to strategy_registry with retry logic for DB lock contention."""
    import time
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            cur = conn.cursor()
            updated = 0
            for strategy_name in all_strategies:
                bt = bt_results.get(strategy_name)
                ft = ft_results.get(strategy_name)

                if bt:
                    m = bt["metrics"]
                    e = bt["eep"]
                    cur.execute("""
                        UPDATE strategy_registry SET
                            bt_win_rate = ?, bt_sharpe = ?, bt_pnl_pct = ?,
                            bt_max_dd = ?, bt_trades = ?, bt_eep_score = ?,
                            bt_last_run = ?, updated_at = ?
                        WHERE strategy_name = ?
                    """, (
                        round(m["win_rate"], 4),
                        round(m["sharpe_ratio"], 4),
                        round(m["total_pnl_pct"], 4),
                        round(m["max_drawdown_pct"], 4),
                        m["num_trades"],
                        e["eep_score"],
                        now_iso,
                        now_iso,
                        strategy_name,
                    ))
                    updated += cur.rowcount

                if ft:
                    fm = ft["metrics"]
                    fe = ft["eep"]
                    cur.execute("""
                        UPDATE strategy_registry SET
                            ft_win_rate = ?, ft_sharpe = ?, ft_pnl_pct = ?,
                            ft_max_dd = ?, ft_trades = ?, ft_eep_score = ?,
                            ft_last_update = ?, updated_at = ?
                        WHERE strategy_name = ?
                    """, (
                        round(fm["win_rate"], 4),
                        round(fm["sharpe"], 4),
                        round(fm["pnl_pct"], 4),
                        round(fm["max_dd"], 4),
                        fm["trades"],
                        fe["eep_score"],
                        now_iso,
                        now_iso,
                        strategy_name,
                    ))
                    updated += cur.rowcount

            conn.commit()
            conn.close()
            return updated
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"  DB locked, retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(wait)
            else:
                raise


def main():
    now_iso = datetime.now(timezone.utc).isoformat()

    # --- Read phase: use read-only connection ---
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)

    # --- Backtest EEP ---
    print("=" * 70)
    print("BACKTEST EEP SCORES (from strategy_scores)")
    print("=" * 70)

    bt_metrics = get_backtest_metrics(conn)
    bt_results = {}
    for strategy, metrics in sorted(bt_metrics.items()):
        result = compute_eep_from_metrics(metrics)
        bt_results[strategy] = {
            "eep": result,
            "metrics": metrics,
        }

    # --- Forward-test EEP ---
    print()
    print("=" * 70)
    print("FORWARD-TEST EEP SCORES (from paper_trades)")
    print("=" * 70)

    ft_trades = get_forward_test_trades(conn)
    ft_results = {}
    for strategy, trades in sorted(ft_trades.items()):
        result = compute_eep_from_trades(trades, label=strategy)
        ft_metrics = compute_ft_metrics_from_trades(trades)
        ft_results[strategy] = {
            "eep": result,
            "metrics": ft_metrics,
        }

    # --- Get all strategy names ---
    cur = conn.cursor()
    cur.execute("SELECT strategy_name FROM strategy_registry")
    all_strategies = [row[0] for row in cur.fetchall()]
    conn.close()

    # --- Write phase: separate connection with retry ---
    print("\nUpdating strategy_registry...")
    updated = write_updates_with_retry(all_strategies, bt_results, ft_results, now_iso)
    print(f"Updated {updated} rows.")

    # --- Print summary ---
    print()
    print("=" * 70)
    print("SUMMARY: ALL STRATEGIES RANKED BY EEP SCORE")
    print("=" * 70)

    # Combine BT and FT into one ranked view
    all_scored = []
    for name in all_strategies:
        bt = bt_results.get(name)
        ft = ft_results.get(name)
        bt_score = bt["eep"]["eep_score"] if bt else None
        ft_score = ft["eep"]["eep_score"] if ft else None
        bt_gate = bt["eep"]["gate_pass"] if bt else None
        ft_gate = ft["eep"]["gate_pass"] if ft else None
        bt_trades_n = bt["metrics"]["num_trades"] if bt else 0
        ft_trades_n = ft["metrics"]["trades"] if ft else 0
        # Use best available score for ranking
        best_score = max(filter(None, [bt_score, ft_score]), default=0)
        all_scored.append({
            "name": name,
            "bt_score": bt_score,
            "ft_score": ft_score,
            "bt_gate": bt_gate,
            "ft_gate": ft_gate,
            "bt_trades": bt_trades_n,
            "ft_trades": ft_trades_n,
            "best": best_score,
        })

    all_scored.sort(key=lambda x: -x["best"])

    print(f"\n{'Rank':<5} {'Strategy':<30} {'BT EEP':>8} {'BT Gate':>8} {'BT Trades':>10} {'FT EEP':>8} {'FT Gate':>8} {'FT Trades':>10}")
    print("-" * 97)
    for i, s in enumerate(all_scored, 1):
        bt_str = f"{s['bt_score']:>8.2f}" if s["bt_score"] is not None else "     N/A"
        ft_str = f"{s['ft_score']:>8.2f}" if s["ft_score"] is not None else "     N/A"
        bt_g = "PASS" if s["bt_gate"] else ("FAIL" if s["bt_gate"] is not None else "N/A")
        ft_g = "PASS" if s["ft_gate"] else ("FAIL" if s["ft_gate"] is not None else "N/A")
        bt_t = f"{s['bt_trades']:>10}" if s["bt_trades"] else "       N/A"
        ft_t = f"{s['ft_trades']:>10}" if s["ft_trades"] else "       N/A"
        print(f"{i:<5} {s['name']:<30} {bt_str} {bt_g:>8} {bt_t} {ft_str} {ft_g:>8} {ft_t}")

    # Print detailed BT results
    print()
    print("=" * 70)
    print("BACKTEST DETAIL")
    print("=" * 70)
    for strategy in sorted(bt_results, key=lambda s: -bt_results[s]["eep"]["eep_score"]):
        r = bt_results[strategy]
        e = r["eep"]
        m = r["metrics"]
        print(f"\n  {strategy}")
        print(f"    EEP: {e['eep_score']:>6.2f}  Entry: {e['entry_score']:>6.2f}  Exit: {e['exit_score']:>6.2f}")
        print(f"    WR: {m['win_rate']:.2%}  Sharpe: {m['sharpe_ratio']:.3f}  PnL: {m['total_pnl_pct']:.2f}%  DD: {m['max_drawdown_pct']:.2f}%  Trades: {m['num_trades']}")
        print(f"    Gate: {'PASS' if e['gate_pass'] else 'FAIL'}", end="")
        if not e["gate_pass"]:
            print(f"  [{', '.join(e['gate_fails'])}]", end="")
        print()

    # Print detailed FT results
    print()
    print("=" * 70)
    print("FORWARD-TEST DETAIL")
    print("=" * 70)
    for strategy in sorted(ft_results, key=lambda s: -ft_results[s]["eep"]["eep_score"]):
        r = ft_results[strategy]
        e = r["eep"]
        fm = r["metrics"]
        print(f"\n  {strategy}")
        print(f"    EEP: {e['eep_score']:>6.2f}  Entry: {e['entry_score']:>6.2f}  Exit: {e['exit_score']:>6.2f}")
        print(f"    WR: {fm['win_rate']:.2%}  Sharpe: {fm['sharpe']:.3f}  PnL: {fm['pnl_pct']:.2f}%  DD: {fm['max_dd']:.2f}%  Trades: {fm['trades']}")
        print(f"    Gate: {'PASS' if e['gate_pass'] else 'FAIL'}", end="")
        if not e["gate_pass"]:
            print(f"  [{', '.join(e['gate_fails'])}]", end="")
        print()

    # Summary stats
    bt_with_data = len(bt_results)
    ft_with_data = len(ft_results)
    bt_passing = sum(1 for r in bt_results.values() if r["eep"]["gate_pass"])
    ft_passing = sum(1 for r in ft_results.values() if r["eep"]["gate_pass"])
    total_reg = len(all_strategies)
    no_data = total_reg - len(set(list(bt_results.keys()) + list(ft_results.keys())))

    print()
    print("=" * 70)
    print("STATS")
    print("=" * 70)
    print(f"  Total in registry:          {total_reg}")
    print(f"  With backtest data:         {bt_with_data} ({bt_passing} pass gates)")
    print(f"  With forward-test data:     {ft_with_data} ({ft_passing} pass gates)")
    print(f"  No data (columns stay NULL): {no_data}")
    print(f"  DB rows updated:            {updated}")
    print(f"  Timestamp:                  {now_iso}")
    print()

    conn.close()


if __name__ == "__main__":
    main()
