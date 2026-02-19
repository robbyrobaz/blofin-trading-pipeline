#!/usr/bin/env python3
"""
ab_test_models.py
-----------------
A/B Testing Framework for ML Model Swap Decision.

  ARM A (v1_backtest) — models trained on historical backtest data
  ARM B (v2_paper)    — models retrained on paper-trading data

Decision rules:
  1. ≥ 100 closed trades per arm required before evaluation
  2. Arm B Sharpe ≥ Arm A Sharpe
  3. Arm B positive expectancy (mean pnl > 0)
  4. All decisions logged to DB for full audit trail

Usage:
    python3 ab_test_models.py               # evaluate + decide
    python3 ab_test_models.py --dry-run     # evaluate only, no swap
    python3 ab_test_models.py --status      # show current state
    python3 ab_test_models.py --force-swap  # force swap regardless
    python3 ab_test_models.py --db PATH     # specify DB path
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT     = Path(__file__).resolve().parent
DB_PATH  = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
LOG_DIR  = ROOT / "logs"
LOG_FILE = LOG_DIR / "phase2_retrain.log"

MIN_TRADES_PER_ARM = 100

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ab_test")


# ─── DB helpers ───────────────────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def ensure_ab_tables(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS ab_test_runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms             INTEGER NOT NULL,
            ts_iso            TEXT    NOT NULL,
            arm_a_label       TEXT    NOT NULL DEFAULT 'v1_backtest',
            arm_b_label       TEXT    NOT NULL DEFAULT 'v2_paper',
            arm_a_trades      INTEGER,
            arm_b_trades      INTEGER,
            arm_a_sharpe      REAL,
            arm_b_sharpe      REAL,
            arm_a_expectancy  REAL,
            arm_b_expectancy  REAL,
            arm_a_win_rate    REAL,
            arm_b_win_rate    REAL,
            arm_a_pf          REAL,
            arm_b_pf          REAL,
            swap_recommended  INTEGER DEFAULT 0,
            swap_executed     INTEGER DEFAULT 0,
            decision_reason   TEXT,
            dry_run           INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS ab_active_model (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            active_arm  TEXT    NOT NULL DEFAULT 'v1_backtest',
            swapped_at  TEXT,
            swap_run_id INTEGER
        );

        INSERT OR IGNORE INTO ab_active_model (id, active_arm)
        VALUES (1, 'v1_backtest');
    """)
    con.commit()


# ─── Trade loading ─────────────────────────────────────────────────────────────

def get_retrain_ts(con: sqlite3.Connection) -> Optional[int]:
    """Timestamp of the most recent successful Phase 2 retrain (or None)."""
    try:
        row = con.execute(
            """SELECT ts_ms FROM phase2_retrain_runs
               WHERE success = 1
               ORDER BY ts_ms DESC LIMIT 1"""
        ).fetchone()
        return int(row["ts_ms"]) if row else None
    except Exception:
        return None


def load_all_closed_trades(con: sqlite3.Connection, since_ms: int) -> List[Dict]:
    """Load all closed paper trades at or after `since_ms`."""
    rows = con.execute(
        """SELECT id, symbol, side, entry_price, exit_price,
                  opened_ts_ms, closed_ts_ms, pnl_pct, reason
           FROM paper_trades
           WHERE status = 'CLOSED'
             AND opened_ts_ms >= ?
             AND exit_price IS NOT NULL
             AND pnl_pct IS NOT NULL
           ORDER BY opened_ts_ms""",
        (since_ms,),
    ).fetchall()
    return [dict(r) for r in rows]


def split_trades_by_arm(
    all_trades: List[Dict],
    retrain_ts_ms: Optional[int],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Arm A = trades BEFORE the Phase 2 retrain (backtest-model era).
    Arm B = trades AFTER the retrain  (paper-model era).

    If no retrain timestamp, falls back to a 50/50 temporal split.
    """
    if not all_trades:
        return [], []

    if retrain_ts_ms is not None:
        arm_a = [t for t in all_trades if t["opened_ts_ms"] <  retrain_ts_ms]
        arm_b = [t for t in all_trades if t["opened_ts_ms"] >= retrain_ts_ms]
    else:
        mid   = len(all_trades) // 2
        arm_a = all_trades[:mid]
        arm_b = all_trades[mid:]

    return arm_a, arm_b


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _pnls(trades: List[Dict]) -> List[float]:
    return [float(t.get("pnl_pct") or 0) for t in trades]


def _sharpe(pnls: List[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    var  = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std  = math.sqrt(var)
    return (mean / std) if std > 0 else 0.0


def _expectancy(pnls: List[float]) -> float:
    return sum(pnls) / len(pnls) if pnls else 0.0


def _profit_factor(pnls: List[float]) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def _win_rate(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def compute_arm_metrics(trades: List[Dict]) -> Dict[str, Any]:
    pnls = _pnls(trades)
    return {
        "num_trades":    len(pnls),
        "sharpe":        round(_sharpe(pnls), 4),
        "expectancy":    round(_expectancy(pnls), 4),
        "win_rate":      round(_win_rate(pnls), 4),
        "profit_factor": round(_profit_factor(pnls), 3),
        "total_pnl":     round(sum(pnls), 4),
    }


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_ab(
    arm_a_trades: List[Dict],
    arm_b_trades: List[Dict],
) -> Dict[str, Any]:
    """
    Compare arm metrics and return a swap recommendation.
    """
    metrics_a = compute_arm_metrics(arm_a_trades)
    metrics_b = compute_arm_metrics(arm_b_trades)

    reasons = []

    # Minimum sample gates
    if metrics_a["num_trades"] < MIN_TRADES_PER_ARM:
        return {
            "ready":            False,
            "reason":           f"Arm A only {metrics_a['num_trades']} trades (need {MIN_TRADES_PER_ARM})",
            "arm_a":            metrics_a,
            "arm_b":            metrics_b,
            "swap_recommended": False,
        }
    if metrics_b["num_trades"] < MIN_TRADES_PER_ARM:
        return {
            "ready":            False,
            "reason":           f"Arm B only {metrics_b['num_trades']} trades (need {MIN_TRADES_PER_ARM})",
            "arm_a":            metrics_a,
            "arm_b":            metrics_b,
            "swap_recommended": False,
        }

    sharpe_pass     = metrics_b["sharpe"]     >= metrics_a["sharpe"]
    expectancy_pass = metrics_b["expectancy"] >  0.0
    swap_recommended = sharpe_pass and expectancy_pass

    if sharpe_pass:
        reasons.append(
            f"B Sharpe {metrics_b['sharpe']:.3f} ≥ A Sharpe {metrics_a['sharpe']:.3f} ✅"
        )
    else:
        reasons.append(
            f"B Sharpe {metrics_b['sharpe']:.3f} < A Sharpe {metrics_a['sharpe']:.3f} ❌"
        )

    if expectancy_pass:
        reasons.append(f"B expectancy {metrics_b['expectancy']:+.4f} > 0 ✅")
    else:
        reasons.append(f"B expectancy {metrics_b['expectancy']:+.4f} ≤ 0 ❌")

    return {
        "ready":             True,
        "reason":            "; ".join(reasons),
        "arm_a":             metrics_a,
        "arm_b":             metrics_b,
        "swap_recommended":  swap_recommended,
        "sharpe_delta":      round(metrics_b["sharpe"]     - metrics_a["sharpe"],     4),
        "expectancy_delta":  round(metrics_b["expectancy"] - metrics_a["expectancy"], 4),
    }


# ─── Swap helpers ─────────────────────────────────────────────────────────────

def get_active_model(con: sqlite3.Connection) -> str:
    try:
        row = con.execute(
            "SELECT active_arm FROM ab_active_model WHERE id=1"
        ).fetchone()
        return row["active_arm"] if row else "v1_backtest"
    except Exception:
        return "v1_backtest"


def execute_swap(con: sqlite3.Connection, run_id: int) -> None:
    ts_iso = datetime.now(timezone.utc).isoformat()
    con.execute(
        """UPDATE ab_active_model
           SET active_arm='v2_paper', swapped_at=?, swap_run_id=?
           WHERE id=1""",
        (ts_iso, run_id),
    )
    con.commit()
    log.info(f"✅ Active model → v2_paper at {ts_iso}")


def log_run_to_db(
    con: sqlite3.Connection,
    eval_result: Dict,
    dry_run: bool,
    swap_executed: bool,
) -> int:
    ts_ms  = int(time.time() * 1000)
    ts_iso = datetime.now(timezone.utc).isoformat()
    arm_a  = eval_result.get("arm_a", {})
    arm_b  = eval_result.get("arm_b", {})

    try:
        cur = con.execute(
            """INSERT INTO ab_test_runs
               (ts_ms, ts_iso, arm_a_label, arm_b_label,
                arm_a_trades, arm_b_trades,
                arm_a_sharpe, arm_b_sharpe,
                arm_a_expectancy, arm_b_expectancy,
                arm_a_win_rate, arm_b_win_rate,
                arm_a_pf, arm_b_pf,
                swap_recommended, swap_executed,
                decision_reason, dry_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts_ms, ts_iso, "v1_backtest", "v2_paper",
                arm_a.get("num_trades"),    arm_b.get("num_trades"),
                arm_a.get("sharpe"),        arm_b.get("sharpe"),
                arm_a.get("expectancy"),    arm_b.get("expectancy"),
                arm_a.get("win_rate"),      arm_b.get("win_rate"),
                arm_a.get("profit_factor"), arm_b.get("profit_factor"),
                1 if eval_result.get("swap_recommended") else 0,
                1 if swap_executed else 0,
                eval_result.get("reason", ""),
                1 if dry_run else 0,
            ),
        )
        con.commit()
        return cur.lastrowid
    except Exception as e:
        log.warning(f"Failed to log A/B run: {e}")
        return -1


# ─── Status display ───────────────────────────────────────────────────────────

def show_status(con: sqlite3.Connection) -> None:
    active = get_active_model(con)
    print(f"\n📊 A/B Test Status")
    print(f"   Active model: {active}")

    try:
        swap_row = con.execute(
            "SELECT swapped_at FROM ab_active_model WHERE id=1"
        ).fetchone()
        if swap_row and swap_row["swapped_at"]:
            print(f"   Last swap:    {swap_row['swapped_at']}")
    except Exception:
        pass

    try:
        row = con.execute(
            """SELECT ts_iso, arm_a_trades, arm_b_trades,
                      arm_a_sharpe, arm_b_sharpe,
                      arm_a_expectancy, arm_b_expectancy,
                      swap_recommended, swap_executed, decision_reason
               FROM ab_test_runs ORDER BY ts_ms DESC LIMIT 1"""
        ).fetchone()
        if row:
            print(f"\n   Last evaluation: {row['ts_iso']}")
            print(f"   Arm A (v1_backtest):"
                  f"  trades={row['arm_a_trades']}"
                  f"  sharpe={row['arm_a_sharpe']}"
                  f"  exp={row['arm_a_expectancy']}")
            print(f"   Arm B (v2_paper):   "
                  f"  trades={row['arm_b_trades']}"
                  f"  sharpe={row['arm_b_sharpe']}"
                  f"  exp={row['arm_b_expectancy']}")
            print(f"   Swap recommended: {'✅' if row['swap_recommended'] else '❌'}")
            print(f"   Swap executed:    {'✅' if row['swap_executed'] else '❌'}")
            if row["decision_reason"]:
                print(f"   Reason: {row['decision_reason']}")
        else:
            print("   No evaluations run yet")
    except Exception as e:
        print(f"   Error fetching status: {e}")


# ─── Main evaluation pipeline ─────────────────────────────────────────────────

def run_ab_evaluation(
    db_path: str,
    dry_run: bool = False,
    force_swap: bool = False,
) -> Dict[str, Any]:
    log.info("=" * 60)
    log.info("A/B Model Evaluation — Starting")
    log.info(f"  DB:         {db_path}")
    log.info(f"  Dry-run:    {dry_run}")
    log.info(f"  Force swap: {force_swap}")
    log.info("=" * 60)

    con = connect(db_path)
    ensure_ab_tables(con)

    active_model = get_active_model(con)
    log.info(f"Current active model: {active_model}")

    # Load trades from last 60 days (or from first retrain start)
    since_ms   = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp() * 1000)
    retrain_ts = get_retrain_ts(con)
    all_trades = load_all_closed_trades(con, since_ms)

    if retrain_ts:
        log.info(
            f"Retrain timestamp: "
            f"{datetime.fromtimestamp(retrain_ts/1000, tz=timezone.utc).isoformat()}"
        )
    else:
        log.info("No successful retrain on record — using 50/50 temporal split")

    arm_a, arm_b = split_trades_by_arm(all_trades, retrain_ts)
    log.info(
        f"Trades: total={len(all_trades)}, "
        f"arm_a={len(arm_a)}, arm_b={len(arm_b)}"
    )

    # Evaluate
    eval_result = evaluate_ab(arm_a, arm_b)

    log.info("\n📋 Evaluation Results:")
    log.info(f"  Ready:  {eval_result['ready']}")
    log.info(f"  Reason: {eval_result.get('reason', '')}")

    if eval_result.get("ready"):
        a = eval_result["arm_a"]
        b = eval_result["arm_b"]
        log.info(
            f"\n  Arm A: trades={a['num_trades']} sharpe={a['sharpe']:+.3f} "
            f"exp={a['expectancy']:+.4f} wr={a['win_rate']:.1%} pf={a['profit_factor']:.2f}"
        )
        log.info(
            f"  Arm B: trades={b['num_trades']} sharpe={b['sharpe']:+.3f} "
            f"exp={b['expectancy']:+.4f} wr={b['win_rate']:.1%} pf={b['profit_factor']:.2f}"
        )
        log.info(f"\n  Δ Sharpe:     {eval_result['sharpe_delta']:+.4f}")
        log.info(f"  Δ Expectancy: {eval_result['expectancy_delta']:+.4f}")
        log.info(f"  Swap recommended: {'✅ YES' if eval_result['swap_recommended'] else '❌ NO'}")

    # Log to DB before potential swap
    run_id        = log_run_to_db(con, eval_result, dry_run, False)
    swap_executed = False

    should_swap = (
        force_swap
        or (eval_result.get("ready") and eval_result.get("swap_recommended"))
    )

    if should_swap:
        if dry_run:
            log.info("  [Dry-run] Would swap to v2_paper — skipping execution")
        else:
            execute_swap(con, run_id)
            swap_executed = True
            # Update log row
            try:
                con.execute(
                    "UPDATE ab_test_runs SET swap_executed=1 WHERE id=?",
                    (run_id,),
                )
                con.commit()
            except Exception:
                pass
    else:
        log.info("  No swap — keeping current model")

    con.close()

    new_active = "v2_paper" if swap_executed else active_model
    return {
        "success":           True,
        "active_model":      new_active,
        "swap_executed":     swap_executed,
        "swap_recommended":  eval_result.get("swap_recommended", False),
        "evaluation":        eval_result,
        "run_id":            run_id,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="A/B Model Evaluation Framework")
    parser.add_argument("--dry-run",    action="store_true", help="Evaluate only, no swap")
    parser.add_argument("--status",     action="store_true", help="Show current A/B status")
    parser.add_argument("--force-swap", action="store_true", help="Force swap regardless of metrics")
    parser.add_argument("--db",         default=DB_PATH,     help="Database path")
    args = parser.parse_args()

    con = connect(args.db)
    ensure_ab_tables(con)

    if args.status:
        show_status(con)
        con.close()
        return 0

    con.close()

    result = run_ab_evaluation(
        db_path=args.db,
        dry_run=args.dry_run,
        force_swap=args.force_swap,
    )

    print("\n" + "=" * 60)
    print("A/B Evaluation Complete")
    print(f"  Active model:      {result.get('active_model')}")
    print(f"  Swap recommended:  {'✅' if result.get('swap_recommended') else '❌'}")
    print(f"  Swap executed:     {'✅' if result.get('swap_executed') else '❌'}")

    eval_r = result.get("evaluation", {})
    if not eval_r.get("ready"):
        print(f"  ⚠  Not ready: {eval_r.get('reason', '')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
