#!/usr/bin/env python3
"""
sync_optimizer_to_db.py
-----------------------
Reads the latest optimizer_results_*.json and writes top strategies
into the database so the dashboard shows real results instead of stale data.

Usage:
    python3 sync_optimizer_to_db.py                  # latest file
    python3 sync_optimizer_to_db.py --file PATH      # specific file
    python3 sync_optimizer_to_db.py --all            # all optimizer result files

Tables updated:
    - strategy_scores         (read by blofin-dashboard /api/strategies)
    - strategy_backtest_results (read by blofin-stack /api/summary)
    - optimizer_runs          (created if not exists — audit trail)
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
REPORTS_DIR = ROOT / "data" / "reports"

# Import canonical EEP scorer (Phase 2 implementation)
sys.path.insert(0, str(ROOT))
from eep_scorer import score_strategy_eep, passes_hard_gates, compute_eep_from_metrics


# ─── DB helpers ──────────────────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_optimizer_runs_table(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS optimizer_runs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms            INTEGER NOT NULL,
            ts_iso           TEXT    NOT NULL,
            source_file      TEXT    NOT NULL UNIQUE,
            run_timestamp    TEXT,
            total_time_min   REAL,
            strategies_count INTEGER,
            top_strategy     TEXT,
            top_pnl_pct      REAL,
            top_win_rate     REAL,
            top_sharpe       REAL,
            raw_json         TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_optimizer_runs_ts
            ON optimizer_runs(ts_ms);
    """)
    con.commit()


def score_strategy(s: dict) -> float:
    """Return EEP composite score scaled 0-100 (replaces win-rate-only ranking)."""
    result = score_strategy_eep(s)
    return result["eep_score"]


# ─── Core sync ───────────────────────────────────────────────────────────────

def sync_file(con: sqlite3.Connection, json_path: Path, verbose: bool = True) -> dict:
    """
    Sync one optimizer results JSON into DB.
    Returns a summary dict.
    """
    if verbose:
        print(f"\n📂  Loading: {json_path.name}")

    with open(json_path) as f:
        data = json.load(f)

    run_ts  = data.get("timestamp", datetime.now(timezone.utc).isoformat())
    run_min = data.get("total_time_min")
    top_strats = data.get("top_strategies", [])

    if not top_strats:
        print("  ⚠  No top_strategies found — skipping")
        return {"skipped": True, "reason": "empty top_strategies"}

    now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    now_iso = datetime.now(timezone.utc).isoformat()

    # ── 1. Log the optimizer run ─────────────────────────────────────────────
    best = top_strats[0]
    try:
        con.execute(
            """INSERT INTO optimizer_runs
               (ts_ms, ts_iso, source_file, run_timestamp, total_time_min,
                strategies_count, top_strategy, top_pnl_pct, top_win_rate,
                top_sharpe, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_file) DO UPDATE SET
                   ts_ms            = excluded.ts_ms,
                   ts_iso           = excluded.ts_iso,
                   run_timestamp    = excluded.run_timestamp,
                   total_time_min   = excluded.total_time_min,
                   strategies_count = excluded.strategies_count,
                   top_strategy     = excluded.top_strategy,
                   top_pnl_pct      = excluded.top_pnl_pct,
                   top_win_rate     = excluded.top_win_rate,
                   top_sharpe       = excluded.top_sharpe,
                   raw_json         = excluded.raw_json""",
            (
                now_ms, now_iso, str(json_path), run_ts, run_min,
                len(top_strats),
                best.get("strategy"), best.get("total_pnl_pct"),
                best.get("win_rate"), best.get("sharpe_ratio"),
                json.dumps(data),
            )
        )
    except Exception as e:
        print(f"  ⚠  optimizer_runs insert error: {e}")

    inserted_scores  = 0
    inserted_backtest = 0

    for s in top_strats:
        strategy   = s.get("strategy", "unknown")
        symbol     = s.get("symbol", "UNKNOWN")
        timeframe  = s.get("timeframe", "5m")
        win_rate   = float(s.get("win_rate", 0))
        sharpe     = float(s.get("sharpe_ratio", 0))
        pnl_total  = float(s.get("total_pnl_pct", 0))
        pnl_avg    = float(s.get("avg_pnl_pct", 0))
        max_dd     = float(s.get("max_drawdown_pct", 0))
        num_trades = int(s.get("num_trades", 0))
        params     = s.get("params", {})
        risk_params = s.get("risk_params", {})
        composite  = score_strategy(s)

        wins   = round(num_trades * win_rate)
        losses = num_trades - wins

        # EEP analysis
        eep_result = score_strategy_eep(s)
        gate_pass  = eep_result["gate_pass"]
        gate_fails = eep_result["gate_fails"]

        if verbose:
            gate_str = "✅ PASS" if gate_pass else f"⛔ FAIL ({', '.join(gate_fails)})"

        # ── 2. strategy_scores (used by /api/strategies) ─────────────────────
        try:
            con.execute(
                """INSERT INTO strategy_scores
                   (ts_ms, ts_iso, strategy, symbol, window, trades, wins, losses,
                    win_rate, avg_pnl_pct, total_pnl_pct, sharpe_ratio,
                    max_drawdown_pct, score, enabled)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    now_ms, now_iso, strategy, symbol, timeframe,
                    num_trades, wins, losses,
                    win_rate, pnl_avg, pnl_total, sharpe,
                    max_dd, composite,
                )
            )
            inserted_scores += 1
        except Exception as e:
            print(f"  ⚠  strategy_scores insert error ({strategy}): {e}")

        # ── 3. strategy_backtest_results (used by blofin-stack /api/summary) ─
        config_blob = json.dumps({**params, **risk_params})
        metrics_blob = json.dumps({
            "timeframe": timeframe, "rank": s.get("rank"),
            "source": "optimizer", "source_file": json_path.name,
        })
        # Only use columns that actually exist in the live table
        # (the table was created before newer columns were added to db.py)
        try:
            con.execute(
                """INSERT INTO strategy_backtest_results
                   (ts_ms, ts_iso, strategy, symbol, backtest_window_days,
                    total_trades, win_rate, sharpe_ratio, max_drawdown_pct,
                    total_pnl_pct, avg_pnl_pct, score, config_json,
                    metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now_ms, now_iso, strategy, symbol, 30,
                    num_trades, win_rate, sharpe, max_dd,
                    pnl_total, pnl_avg, composite,
                    config_blob, metrics_blob,
                )
            )
            inserted_backtest += 1
        except Exception as e:
            print(f"  ⚠  strategy_backtest_results insert error ({strategy}): {e}")

        if verbose:
            gate_str = "✅" if gate_pass else f"⛔({','.join(gate_fails)})"
            print(
                f"  ✓ rank={s.get('rank','-')} {strategy:<28} {symbol:<12}"
                f"  wr={win_rate:.1%}  pnl={pnl_total:+.2f}%"
                f"  sharpe={sharpe:.2f}  EEP={composite}  gates={gate_str}"
            )

    con.commit()

    summary = {
        "source_file": json_path.name,
        "run_timestamp": run_ts,
        "strategies_synced": len(top_strats),
        "strategy_scores_inserted": inserted_scores,
        "backtest_results_inserted": inserted_backtest,
    }

    if verbose:
        print(f"\n  ✅  Synced {inserted_scores} rows → strategy_scores")
        print(f"  ✅  Synced {inserted_backtest} rows → strategy_backtest_results")

    return summary


def find_latest_file(reports_dir: Path) -> Path | None:
    files = sorted(reports_dir.glob("optimizer_results_*.json"), reverse=True)
    return files[0] if files else None


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync optimizer results to DB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", help="Specific JSON file path")
    group.add_argument("--all",  action="store_true", help="Sync ALL optimizer result files")
    parser.add_argument("--db",  default=DB_PATH, help="Database path")
    args = parser.parse_args()

    con = connect(args.db)
    ensure_optimizer_runs_table(con)

    if args.all:
        files = sorted(REPORTS_DIR.glob("optimizer_results_*.json"))
        if not files:
            print("No optimizer result files found.")
            return
        print(f"Syncing {len(files)} file(s)...")
        for f in files:
            sync_file(con, f)
    elif args.file:
        sync_file(con, Path(args.file))
    else:
        latest = find_latest_file(REPORTS_DIR)
        if not latest:
            print(f"No optimizer_results_*.json found in {REPORTS_DIR}")
            return
        sync_file(con, latest)

    con.close()
    print("\nDone. Restart or wait for the dashboard to refresh.")


if __name__ == "__main__":
    main()
