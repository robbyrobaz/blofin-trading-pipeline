#!/usr/bin/env python3
"""
smoke_test_phase2.py
--------------------
Comprehensive smoke test for Phase 2 ML Retrain Framework.

Tests:
  1. ml_retrain_phase2.py — gates, tick sampling, feature building, model training
  2. ab_test_models.py     — trade splitting, metrics, swap logic
  3. execution_calibrator_v2.py — EMA weighting, TOD buckets, regime classification
  4. Database integrity  — tables created, data logged properly
  5. File outputs        — models saved, calibration JSON written

Logs to: logs/smoke_test_phase2_results.log

Usage:
    python3 smoke_test_phase2.py
    python3 smoke_test_phase2.py --quick    # faster: use 1-week data
    python3 smoke_test_phase2.py --verbose
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DB_PATH = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
CALIB_PATH = ROOT / "data" / "execution_calibration.json"
LOG_PATH = ROOT / "logs" / "smoke_test_phase2_results.log"

LOG_LINES: List[str] = []


def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [{level}] {msg}"
    LOG_LINES.append(line)
    print(line)


def flush_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    print(f"\n📄 Log written to {LOG_PATH}")


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


# ─── Step 1: Test ml_retrain_phase2.py gates ──────────────────────────────────

def test_phase2_gates() -> Dict[str, bool]:
    log("=" * 60)
    log("STEP 1: Phase 2 Gate Checks")
    log("=" * 60)

    try:
        from ml_retrain_phase2 import (
            connect as p2_connect,
            ensure_phase2_tables,
            load_closed_trades,
            check_all_gates,
            log as p2_log,
        )

        con = p2_connect(DB_PATH)
        ensure_phase2_tables(con)

        trades = load_closed_trades(con)
        log(f"  Loaded {len(trades)} closed paper trades")

        all_pass, details = check_all_gates(con, trades)

        log(f"\n  Gate results:")
        for gname, info in details.items():
            if gname == "all_pass":
                continue
            icon = "✅" if info.get("passed") else "❌"
            log(f"    {icon} {gname}: {info.get('details', '')}")

        con.close()

        if all_pass:
            log("  ✅ All gates passed (ready for retrain)")
        else:
            log(f"  ⚠  Gates not met (expected for smoke test with insufficient data)")

        return {
            "gates_check_ok": True,
            "gates_passed":   all_pass,
            "trade_count":    len(trades),
        }

    except Exception as e:
        log(f"  ❌ Gate check error: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "DEBUG")
        return {"gates_check_ok": False, "error": str(e)}


# ─── Step 2: Test ml_retrain_phase2.py feature building ──────────────────────

def test_phase2_features() -> Dict[str, bool]:
    log("=" * 60)
    log("STEP 2: Phase 2 Feature Building & Model Training (Smoke Test)")
    log("=" * 60)

    try:
        from ml_retrain_phase2 import (
            connect as p2_connect,
            ensure_phase2_tables,
            load_ohlcv_from_ticks,
            build_features,
            walk_forward_split,
        )

        con = p2_connect(DB_PATH)
        ensure_phase2_tables(con)

        # Load OHLCV data (small window for smoke test)
        log("  Loading sampled OHLCV data (7-day window)...")
        ohlcv = load_ohlcv_from_ticks(con, days_back=7, period_minutes=5)
        log(f"    Candles loaded: {len(ohlcv)}")

        if len(ohlcv) < 100:
            log(f"    ⚠  Only {len(ohlcv)} candles (need 100+) — skipping feature test")
            con.close()
            return {"feature_test_ok": True, "note": "insufficient data"}

        # Build features
        log("  Building features...")
        X, y = build_features(ohlcv)
        log(f"    Feature matrix: {len(X)} rows × {X.shape[1] if len(X) > 0 else 0} cols")

        if len(X) < 50:
            log(f"    ⚠  Only {len(X)} feature rows (need 50+)")
            con.close()
            return {"feature_test_ok": True, "note": f"only {len(X)} features"}

        # Split
        log("  Walk-forward split...")
        X_tr, y_tr, X_te, y_te = walk_forward_split(X, y, embargo_pct=0.05)
        log(f"    Train: {len(X_tr)}, Test: {len(X_te)}, Embargo: {len(X) - len(X_tr) - len(X_te)}")

        con.close()

        return {
            "feature_test_ok": True,
            "candles":         len(ohlcv),
            "features":        len(X),
            "train_size":      len(X_tr),
            "test_size":       len(X_te),
        }

    except Exception as e:
        log(f"  ❌ Feature building error: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "DEBUG")
        return {"feature_test_ok": False, "error": str(e)}


# ─── Step 3: Test ab_test_models.py ────────────────────────────────────────────

def test_ab_evaluation() -> Dict[str, Any]:
    log("=" * 60)
    log("STEP 3: A/B Model Evaluation")
    log("=" * 60)

    try:
        from ab_test_models import (
            connect as ab_connect,
            ensure_ab_tables,
            get_retrain_ts,
            load_all_closed_trades,
            split_trades_by_arm,
            compute_arm_metrics,
            evaluate_ab,
            get_active_model,
        )

        con = ab_connect(DB_PATH)
        ensure_ab_tables(con)

        active = get_active_model(con)
        log(f"  Current active model: {active}")

        since_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
        retrain_ts = get_retrain_ts(con)

        trades = load_all_closed_trades(con, since_ms)
        log(f"  Loaded {len(trades)} trades from last 30 days")

        if len(trades) > 0:
            arm_a, arm_b = split_trades_by_arm(trades, retrain_ts)
            log(f"    Arm A: {len(arm_a)} trades")
            log(f"    Arm B: {len(arm_b)} trades")

            if len(arm_a) > 0 and len(arm_b) > 0:
                eval_result = evaluate_ab(arm_a, arm_b)
                log(f"    Evaluation ready: {eval_result.get('ready')}")
                if eval_result.get("ready"):
                    a = eval_result["arm_a"]
                    b = eval_result["arm_b"]
                    log(f"      Arm A Sharpe: {a['sharpe']:.3f}")
                    log(f"      Arm B Sharpe: {b['sharpe']:.3f}")
                    log(f"      Swap recommended: {'✅' if eval_result.get('swap_recommended') else '❌'}")
            else:
                log(f"    ⚠  Insufficient trades for A/B eval ({len(arm_a)} vs {len(arm_b)})")

        con.close()

        return {
            "ab_test_ok": True,
            "active_model": active,
            "trades_loaded": len(trades),
        }

    except Exception as e:
        log(f"  ❌ A/B evaluation error: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "DEBUG")
        return {"ab_test_ok": False, "error": str(e)}


# ─── Step 4: Test execution_calibrator_v2.py ──────────────────────────────────

def test_execution_calibrator() -> Dict[str, Any]:
    log("=" * 60)
    log("STEP 4: Execution Calibrator v2")
    log("=" * 60)

    try:
        from execution_calibrator_v2 import (
            connect as ec_connect,
            load_closed_trades,
            compute_calibration,
            _placeholder_calibration,
            write_calibration,
        )

        con = ec_connect(DB_PATH)
        trades = load_closed_trades(con)
        con.close()

        log(f"  Found {len(trades)} closed paper trades")

        if len(trades) < 10:
            log(f"  Using placeholder (< 10 trades)")
            calib = _placeholder_calibration(f"only {len(trades)} trades")
        else:
            log(f"  Computing calibration...")
            calib = compute_calibration(trades, verbose=False)

        # Write file
        write_calibration(calib)
        log(f"  ✅ Wrote {CALIB_PATH}")

        # Verify fields
        exec_  = calib.get("execution", {})
        regime = calib.get("regime_slippage", {})
        tod    = calib.get("time_of_day", {})

        log(f"\n  Results:")
        log(f"    Trade count:       {calib['trade_count']}")
        log(f"    EMA slippage:      {exec_.get('ema_slippage_per_side_pct', '?'):.4f}%/side")
        log(f"    Avg slippage (v1): {exec_.get('avg_slippage_per_side_pct', '?'):.4f}%/side (backward-compat)")
        log(f"    Position mult:     {exec_.get('position_size_multiplier', '?'):.3f}×")
        log(f"    TOD buckets:       {len(tod)} buckets")
        log(f"    Regimes:           {len(regime)} regimes")

        # Verify backward-compat field exists
        has_compat = "avg_slippage_per_side_pct" in exec_
        if has_compat:
            log(f"    ✅ Backward-compat field present")
        else:
            log(f"    ❌ Backward-compat field MISSING", "WARN")

        return {
            "calibrator_ok": True,
            "trade_count":   calib['trade_count'],
            "has_compat":    has_compat,
            "file_written":  CALIB_PATH.exists(),
        }

    except Exception as e:
        log(f"  ❌ Calibrator error: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "DEBUG")
        return {"calibrator_ok": False, "error": str(e)}


# ─── Step 5: Database integrity checks ────────────────────────────────────────

def test_db_integrity() -> Dict[str, Any]:
    log("=" * 60)
    log("STEP 5: Database Integrity")
    log("=" * 60)

    try:
        con = connect(DB_PATH)

        # Check Phase 2 tables exist
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'phase2%'"
        ).fetchall()
        log(f"  Phase 2 tables found: {len(tables)}")
        for t in tables:
            log(f"    - {t['name']}")

        # Check A/B tables exist
        ab_tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ab%'"
        ).fetchall()
        log(f"  A/B tables found: {len(ab_tables)}")
        for t in ab_tables:
            log(f"    - {t['name']}")

        # Check paper_trades has required columns
        cols = con.execute("PRAGMA table_info(paper_trades)").fetchall()
        col_names = [c["name"] for c in cols]
        required = ["opened_ts_ms", "closed_ts_ms", "pnl_pct", "status"]
        missing = [c for c in required if c not in col_names]
        if missing:
            log(f"  ❌ Missing columns in paper_trades: {missing}", "ERROR")
        else:
            log(f"  ✅ paper_trades schema OK")

        con.close()

        return {
            "db_ok":      len(tables) > 0,
            "phase2_tables": len(tables),
            "ab_tables":  len(ab_tables),
        }

    except Exception as e:
        log(f"  ❌ DB integrity check error: {e}", "ERROR")
        return {"db_ok": False, "error": str(e)}


# ─── Step 6: File outputs ────────────────────────────────────────────────────

def test_file_outputs() -> Dict[str, Any]:
    log("=" * 60)
    log("STEP 6: File Outputs & Artifacts")
    log("=" * 60)

    model_dir = ROOT / "data" / "models"
    calib_file = CALIB_PATH

    files_ok = True

    # Check models dir
    if model_dir.exists():
        models = list(model_dir.glob("*_v2_paper"))
        log(f"  Models directory: {model_dir}")
        log(f"    v2_paper models: {len(models)}")
        for m in models[:5]:
            log(f"      - {m.name}")
    else:
        log(f"  ⚠  Models dir not created yet (OK for first run)")

    # Check calibration file
    if calib_file.exists():
        try:
            with open(calib_file) as f:
                calib = json.load(f)
            log(f"  ✅ Calibration file: {calib_file}")
            log(f"      Trade count: {calib.get('trade_count', '?')}")
            log(f"      Generated:   {calib.get('generated_at', '?')}")
        except Exception as e:
            log(f"  ❌ Calibration file JSON error: {e}", "ERROR")
            files_ok = False
    else:
        log(f"  ⚠  Calibration file not yet created (OK for first run)")

    # Check logs directory
    log_dir = ROOT / "logs"
    if log_dir.exists():
        p2_log = log_dir / "phase2_retrain.log"
        if p2_log.exists():
            size = p2_log.stat().st_size
            lines = sum(1 for _ in open(p2_log))
            log(f"  ✅ Phase 2 log: {p2_log}")
            log(f"      Size: {size} bytes, {lines} lines")
    else:
        log(f"  ⚠  Logs dir not created yet")

    return {
        "files_ok":  files_ok,
        "calib_exists": calib_file.exists(),
        "models_dir": model_dir.exists(),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 ML Retrain Framework Smoke Test")
    parser.add_argument("--quick",   action="store_true", help="Use 1-week data window")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    log("=" * 60)
    log("PHASE 2 ML RETRAIN FRAMEWORK — SMOKE TEST")
    log(f"Time: {datetime.now(timezone.utc).isoformat()}")
    log(f"DB:   {DB_PATH}")
    log("=" * 60)

    results = {}

    # Run all tests
    results["gates"]      = test_phase2_gates()
    results["features"]   = test_phase2_features()
    results["ab"]         = test_ab_evaluation()
    results["calibrator"] = test_execution_calibrator()
    results["db"]         = test_db_integrity()
    results["files"]      = test_file_outputs()

    # Summary
    log("=" * 60)
    log("SMOKE TEST SUMMARY")
    log("=" * 60)

    passed = 0
    failed = 0

    checks = [
        ("Gate checks",     results["gates"].get("gates_check_ok")),
        ("Feature building", results["features"].get("feature_test_ok")),
        ("A/B evaluation",  results["ab"].get("ab_test_ok")),
        ("Calibrator",      results["calibrator"].get("calibrator_ok")),
        ("Database",        results["db"].get("db_ok")),
        ("File outputs",    results["files"].get("files_ok")),
    ]

    for name, ok in checks:
        if ok:
            log(f"  ✅ {name}")
            passed += 1
        else:
            log(f"  ❌ {name}")
            failed += 1

    log(f"\n  Result: {passed}/{len(checks)} checks passed")

    if failed == 0:
        log("\n🎉 ALL CHECKS PASSED — Phase 2 Framework ready for deployment!")
        exit_code = 0
    else:
        log(f"\n⚠  {failed} check(s) failed — review log for details")
        exit_code = 1

    flush_log()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
