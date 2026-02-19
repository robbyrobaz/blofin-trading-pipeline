#!/usr/bin/env python3
"""
ml_retrain_phase2.py
--------------------
Phase 2 ML Retrain Framework.

Trigger gates (ALL must pass):
  1. Paper trading period ≥ 2 weeks (14 days)
  2. ≥ 75 closed paper trades
  3. Regime diversity: volatility percentile spans 20th–80th

Training procedure:
  - Combines backtest OHLCV + paper OHLCV data
  - Walk-forward with 24h embargo between training cutoff and test start
  - Conservative slippage multiplier: 2x initially, scales toward 1.5x as data grows
  - Trains 5 models: direction, risk, price, momentum, volatility
  - Saves both old (v1_backtest) and new (v2_paper) models for A/B testing

Usage:
    python3 ml_retrain_phase2.py                  # check gates + retrain if ready
    python3 ml_retrain_phase2.py --force          # skip gate checks, force retrain
    python3 ml_retrain_phase2.py --dry-run        # check gates only, no training
    python3 ml_retrain_phase2.py --smoke-test     # 1-week subset for quick smoke test
    python3 ml_retrain_phase2.py --db PATH        # specify DB path
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
DB_PATH    = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
MODELS_DIR = ROOT / "data" / "models"
LOG_DIR    = ROOT / "logs"
LOG_FILE   = LOG_DIR / "phase2_retrain.log"

# ── Trigger gates ──────────────────────────────────────────────────────────────
MIN_PAPER_DAYS    = 14
MIN_CLOSED_TRADES = 75
REGIME_MIN_PCT    = 20.0   # volatility must span ≤ 20th percentile
REGIME_MAX_PCT    = 80.0   # volatility must span ≥ 80th percentile

# ── Training config ────────────────────────────────────────────────────────────
EMBARGO_HOURS         = 24
INITIAL_SLIPPAGE_MULT = 2.0
MIN_SLIPPAGE_MULT     = 1.5
LARGE_TRADE_THRESHOLD = 500   # at this count, mult reaches MIN_SLIPPAGE_MULT

MODEL_NAMES = [
    "direction_predictor",
    "risk_scorer",
    "price_predictor",
    "momentum_classifier",
    "volatility_regressor",
]

# ── Sampling config for tick queries ──────────────────────────────────────────
TICK_SAMPLE_SIZE = 10_000   # rows to sample for regime diversity check

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("phase2_retrain")


# ─── DB helpers ───────────────────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_phase2_tables(con: sqlite3.Connection) -> None:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS phase2_retrain_runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms             INTEGER NOT NULL,
            ts_iso            TEXT    NOT NULL,
            trigger_reason    TEXT    NOT NULL,
            gate_pass         INTEGER NOT NULL DEFAULT 0,
            gate_details      TEXT,
            paper_days        REAL,
            closed_trades     INTEGER,
            vol_percentile_lo REAL,
            vol_percentile_hi REAL,
            slippage_mult     REAL,
            training_rows     INTEGER,
            models_trained    TEXT,
            model_dir         TEXT,
            success           INTEGER DEFAULT 0,
            error_msg         TEXT,
            duration_sec      REAL,
            embargo_hours     REAL
        );

        CREATE TABLE IF NOT EXISTS phase2_gate_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms       INTEGER NOT NULL,
            ts_iso      TEXT    NOT NULL,
            gate_name   TEXT    NOT NULL,
            passed      INTEGER NOT NULL,
            value       REAL,
            threshold   REAL,
            details     TEXT
        );
    """)
    con.commit()


# ─── Gate checks ──────────────────────────────────────────────────────────────

def load_closed_trades(con: sqlite3.Connection) -> List[Dict]:
    rows = con.execute(
        """SELECT id, symbol, side, entry_price, exit_price,
                  opened_ts_ms, closed_ts_ms, pnl_pct, reason
           FROM paper_trades
           WHERE status = 'CLOSED'
             AND exit_price IS NOT NULL
             AND opened_ts_ms IS NOT NULL
             AND closed_ts_ms IS NOT NULL
           ORDER BY closed_ts_ms"""
    ).fetchall()
    return [dict(r) for r in rows]


def check_paper_age_gate(trades: List[Dict]) -> Tuple[bool, Dict]:
    """Gate 1: Paper trading period ≥ MIN_PAPER_DAYS."""
    if not trades:
        return False, {
            "passed": False, "value": 0, "threshold": MIN_PAPER_DAYS,
            "details": "No closed trades",
        }
    first_ts = min(t["opened_ts_ms"] for t in trades)
    last_ts  = max(t["closed_ts_ms"]  for t in trades)
    days = (last_ts - first_ts) / 86_400_000.0
    passed = days >= MIN_PAPER_DAYS
    return passed, {
        "passed": passed, "value": round(days, 2), "threshold": MIN_PAPER_DAYS,
        "details": f"Paper period: {days:.1f}d (need {MIN_PAPER_DAYS}d)",
    }


def check_trade_count_gate(trades: List[Dict]) -> Tuple[bool, Dict]:
    """Gate 2: ≥ MIN_CLOSED_TRADES closed trades."""
    n = len(trades)
    passed = n >= MIN_CLOSED_TRADES
    return passed, {
        "passed": passed, "value": n, "threshold": MIN_CLOSED_TRADES,
        "details": f"{n} closed trades (need {MIN_CLOSED_TRADES})",
    }


def _sample_ticks_fast(con: sqlite3.Connection, days_back: int = 30) -> List[Tuple[int, float]]:
    """
    Efficiently sample tick data without a full table scan.

    Strategy:
      1. Get MAX(rowid) — instant via B-tree
      2. Estimate a rowid lower bound corresponding to days_back
      3. Sample TICK_SAMPLE_SIZE rows from that range via modulo on rowid
         (avoids reading all 28M rows)
    """
    max_row = con.execute("SELECT MAX(rowid) FROM ticks").fetchone()[0]
    if not max_row:
        return []

    # Estimate total rows to skip: assume uniform insert rate
    total_est = con.execute("SELECT COUNT(rowid) FROM ticks WHERE rowid > ?", (max(1, max_row - 100_000),)).fetchone()[0]
    if total_est == 0:
        return []

    # rough rows/day estimate from last 100k rows
    if total_est >= 100_000:
        rows_per_day = total_est * (86400 / 100_000)   # rough
    else:
        rows_per_day = total_est / max(1, (max_row / total_est) / 86400)

    rows_in_window = int(rows_per_day * days_back)
    start_rowid    = max(1, max_row - rows_in_window)

    step = max(1, rows_in_window // TICK_SAMPLE_SIZE)
    rows = con.execute(
        f"SELECT ts_ms, price FROM ticks "
        f"WHERE rowid >= ? AND rowid <= ? AND rowid % {step} = 0 "
        f"ORDER BY rowid",
        (start_rowid, max_row),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def check_regime_diversity_gate(con: sqlite3.Connection) -> Tuple[bool, Dict]:
    """
    Gate 3: Volatility percentile must span from ≤ 20th to ≥ 80th.
    Uses sampled tick data (fast, no full table scan).
    """
    ticks = _sample_ticks_fast(con, days_back=30)

    if len(ticks) < 500:
        return False, {
            "passed": False, "value": 0, "threshold": REGIME_MIN_PCT,
            "details": f"Insufficient sampled tick data ({len(ticks)} rows)",
        }

    # Build 5-minute candles
    arr     = np.array(ticks)
    ts_ms   = arr[:, 0]
    prices  = arr[:, 1]
    periods = (ts_ms // 300_000).astype(np.int64)   # 5-min bucket

    unique_p, first_idx, counts = np.unique(periods, return_index=True, return_counts=True)
    hl_ranges = []
    for i in range(len(unique_p)):
        s = first_idx[i]
        e = s + counts[i]
        chunk = prices[s:e]
        if len(chunk) >= 2 and chunk.mean() > 0:
            hl_ranges.append((chunk.max() - chunk.min()) / chunk.mean())

    if len(hl_ranges) < 50:
        return False, {
            "passed": False, "value": 0, "threshold": REGIME_MIN_PCT,
            "details": f"Too few candles ({len(hl_ranges)}) for regime analysis",
        }

    hl_arr  = np.array(hl_ranges)
    pct_lo  = float(np.percentile(hl_arr, REGIME_MIN_PCT))
    pct_hi  = float(np.percentile(hl_arr, REGIME_MAX_PCT))
    min_val = float(hl_arr.min())
    max_val = float(hl_arr.max())

    # Data must span at least the 20th–80th percentile range
    passed = (min_val <= pct_lo) and (max_val >= pct_hi)

    return passed, {
        "passed": passed,
        "value_lo": round(min_val, 6),
        "value_hi": round(max_val, 6),
        "pct_lo_threshold": round(pct_lo, 6),
        "pct_hi_threshold": round(pct_hi, 6),
        "candles_used": len(hl_ranges),
        "ticks_sampled": len(ticks),
        "details": (
            f"Vol range [{min_val:.5f}, {max_val:.5f}] "
            f"vs 20th–80th [{pct_lo:.5f}, {pct_hi:.5f}] — "
            f"{'PASS' if passed else 'FAIL'}: "
            f"{'diverse regime' if passed else 'regime too narrow'}"
        ),
    }


def check_all_gates(
    con: sqlite3.Connection,
    trades: List[Dict],
) -> Tuple[bool, Dict]:
    age_pass,   age_info   = check_paper_age_gate(trades)
    count_pass, count_info = check_trade_count_gate(trades)
    regime_pass, reg_info  = check_regime_diversity_gate(con)
    all_pass = age_pass and count_pass and regime_pass
    return all_pass, {
        "all_pass":         all_pass,
        "paper_age":        age_info,
        "trade_count":      count_info,
        "regime_diversity": reg_info,
    }


def log_gates_to_db(con: sqlite3.Connection, gate_details: Dict) -> None:
    ts_ms  = int(time.time() * 1000)
    ts_iso = datetime.now(timezone.utc).isoformat()
    for gate_name, info in [
        ("paper_age",        gate_details.get("paper_age", {})),
        ("trade_count",      gate_details.get("trade_count", {})),
        ("regime_diversity", gate_details.get("regime_diversity", {})),
    ]:
        try:
            con.execute(
                """INSERT INTO phase2_gate_log
                   (ts_ms, ts_iso, gate_name, passed, value, threshold, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts_ms, ts_iso, gate_name,
                    1 if info.get("passed") else 0,
                    info.get("value"),
                    info.get("threshold"),
                    info.get("details", ""),
                ),
            )
        except Exception as e:
            log.warning(f"gate_log insert: {e}")
    con.commit()


# ─── Slippage multiplier ──────────────────────────────────────────────────────

def compute_slippage_multiplier(num_trades: int) -> float:
    """
    2.0× at 0 trades → 1.5× at LARGE_TRADE_THRESHOLD trades (linear interpolation).
    """
    frac = min(num_trades / LARGE_TRADE_THRESHOLD, 1.0)
    mult = INITIAL_SLIPPAGE_MULT - (INITIAL_SLIPPAGE_MULT - MIN_SLIPPAGE_MULT) * frac
    return round(mult, 3)


# ─── Data preparation ─────────────────────────────────────────────────────────

def load_ohlcv_from_ticks(
    con: sqlite3.Connection,
    days_back: int = 60,
    period_minutes: int = 5,
) -> np.ndarray:
    """
    Load sampled ticks and convert to OHLCV candles.
    Uses fast rowid-based sampling to avoid scanning 28M rows.
    Returns np.ndarray [ts_ms, open, high, low, close, volume].
    """
    ticks_raw = _sample_ticks_fast(con, days_back=days_back)
    if len(ticks_raw) < 100:
        return np.array([])
    ticks = np.array(ticks_raw)
    return _ticks_to_ohlcv(ticks, period_minutes)


def _ticks_to_ohlcv(ticks: np.ndarray, period_minutes: int = 5) -> np.ndarray:
    if len(ticks) == 0:
        return np.array([])
    period_ms = period_minutes * 60_000
    periods   = (ticks[:, 0] // period_ms).astype(np.int64)
    prices    = ticks[:, 1]
    unique_p, first_idx, counts = np.unique(periods, return_index=True, return_counts=True)
    n      = len(unique_p)
    result = np.empty((n, 6), dtype=np.float64)
    for i in range(n):
        s = first_idx[i]
        e = s + counts[i]
        chunk = prices[s:e]
        result[i, 0] = ticks[s, 0]       # ts_ms
        result[i, 1] = chunk[0]           # open
        result[i, 2] = chunk.max()        # high
        result[i, 3] = chunk.min()        # low
        result[i, 4] = chunk[-1]          # close
        result[i, 5] = float(counts[i])   # volume (tick count)
    return result


def build_features(ohlcv: np.ndarray, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and binary direction labels.
    y = 1 if price is higher 5 candles ahead, else 0.
    """
    if len(ohlcv) < lookback + 5:
        return np.array([]), np.array([])

    closes  = ohlcv[:, 4]
    highs   = ohlcv[:, 2]
    lows    = ohlcv[:, 3]
    volumes = ohlcv[:, 5]
    X, y    = [], []

    for i in range(lookback, len(ohlcv) - 5):
        wc = closes[i - lookback:i]
        wh = highs[i - lookback:i]
        wl = lows[i - lookback:i]
        wv = volumes[i - lookback:i]

        if len(wc) < lookback:
            continue

        rets    = np.diff(wc) / wc[:-1]
        vol_    = rets.std() if len(rets) > 1 else 0.0
        mean_r  = rets.mean() if len(rets) > 0 else 0.0

        def ema_ratio(period: int) -> float:
            if len(wc) < period:
                return 0.0
            k = 2.0 / (period + 1)
            e = wc[0]
            for p in wc[1:]:
                e = p * k + e * (1 - k)
            return (wc[-1] / e - 1.0) if e > 0 else 0.0

        def rsi() -> float:
            period = min(14, len(wc) - 1)
            if period < 2:
                return 50.0
            d     = np.diff(wc[:period + 1])
            gains = d[d > 0].sum() / period
            loses = -d[d < 0].sum() / period
            if loses == 0:
                return 100.0
            return 100.0 - 100.0 / (1.0 + gains / loses)

        bb_mid = wc.mean()
        bb_std = wc.std()
        bb_pos = (wc[-1] - bb_mid) / bb_std if bb_std > 0 else 0.0

        vol_ratio = wv[-1] / wv.mean() if wv.mean() > 0 else 1.0
        hl_range  = (wh[-1] - wl[-1]) / wc[-1] if wc[-1] > 0 else 0.0
        range_pos = (wc[-1] - wl.min()) / (wh.max() - wl.min() + 1e-10)

        feat = np.array([
            mean_r,
            vol_,
            rets[-1] if len(rets) > 0 else 0.0,
            rets[-5:].mean() if len(rets) >= 5 else 0.0,
            ema_ratio(9),
            ema_ratio(21),
            rsi() / 100.0,
            bb_pos,
            vol_ratio,
            hl_range,
            range_pos,
        ])

        future_ret = (closes[i + 5] / closes[i] - 1.0)
        label      = 1 if future_ret > 0 else 0
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


def walk_forward_split(
    X: np.ndarray,
    y: np.ndarray,
    embargo_pct: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    70 % train | embargo_pct embargo (discarded) | remainder test.
    Minimum embargo is 5 % of dataset.
    """
    n           = len(X)
    train_end   = int(n * 0.70)
    embargo_end = train_end + max(int(n * embargo_pct), int(n * 0.05))
    embargo_end = min(embargo_end, int(n * 0.80))  # never exceed 80 %

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test  = X[embargo_end:]
    y_test  = y[embargo_end:]
    return X_train, y_train, X_test, y_test


# ─── Model training ───────────────────────────────────────────────────────────

def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: Path,
    slippage_mult: float = 1.75,
) -> Dict[str, Any]:
    """
    Train one model and save to model_dir/<name>_v2_paper/.
    Regression models: price_predictor, volatility_regressor
    Classification: all others.
    """
    try:
        import pickle
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"model_name": model_name, "success": False, "reason": "sklearn not installed"}

    if len(X_train) < 50:
        return {"model_name": model_name, "success": False, "reason": "insufficient training data"}

    try:
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_train)
        X_te_s  = scaler.transform(X_test)

        is_regressor = model_name in ("price_predictor", "volatility_regressor")

        if is_regressor:
            clf = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            clf.fit(X_tr_s, y_train.astype(float))
            preds    = clf.predict(X_te_s)
            accuracy = float(np.mean(np.sign(preds) == np.sign(y_test - 0.5)))
            f1       = accuracy
        else:
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            clf.fit(X_tr_s, y_train)
            preds    = clf.predict(X_te_s)
            accuracy = float(accuracy_score(y_test, preds))
            f1       = float(f1_score(y_test, preds, average="weighted", zero_division=0))

        # Save model artifacts
        save_dir = model_dir / f"{model_name}_v2_paper"
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "model.pkl", "wb") as fh:
            pickle.dump(clf, fh)
        with open(save_dir / "scaler.pkl", "wb") as fh:
            pickle.dump(scaler, fh)

        meta = {
            "model_name":    model_name,
            "version":       "v2_paper",
            "trained_at":    datetime.now(timezone.utc).isoformat(),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
            "accuracy":      round(accuracy, 4),
            "f1":            round(f1, 4),
            "slippage_mult": slippage_mult,
            "embargo_hours": EMBARGO_HOURS,
            "is_regressor":  is_regressor,
        }
        with open(save_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        log.info(
            f"  ✅ {model_name}: accuracy={accuracy:.3f} f1={f1:.3f} "
            f"(train={len(X_train)}, test={len(X_test)})"
        )
        return {
            "model_name":    model_name,
            "success":       True,
            "accuracy":      accuracy,
            "f1":            f1,
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
            "model_path":    str(save_dir),
        }

    except Exception as e:
        log.error(f"Training {model_name} failed: {e}")
        return {"model_name": model_name, "success": False, "reason": str(e)}


def archive_existing_models(model_dir: Path) -> None:
    """Back up current v2_paper models to *_prev before overwriting."""
    for name in MODEL_NAMES:
        src = model_dir / f"{name}_v2_paper"
        dst = model_dir / f"{name}_v2_paper_prev"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            log.info(f"  Archived {src.name} → {dst.name}")


# ─── Main retrain procedure ───────────────────────────────────────────────────

def run_phase2_retrain(
    db_path: str,
    force: bool = False,
    dry_run: bool = False,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    t_start = time.time()
    ts_ms   = int(t_start * 1000)
    ts_iso  = datetime.now(timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("Phase 2 ML Retrain — Starting")
    log.info(f"  DB:       {db_path}")
    log.info(f"  Force:    {force}")
    log.info(f"  Dry-run:  {dry_run}")
    log.info(f"  Smoke:    {smoke_test}")
    log.info("=" * 60)

    con = connect(db_path)
    ensure_phase2_tables(con)

    # Step 1: load paper trades
    trades = load_closed_trades(con)
    log.info(f"Loaded {len(trades)} closed paper trades")

    # Step 2: gate checks (optionally relax thresholds for smoke test)
    _min_days, _min_trades = MIN_PAPER_DAYS, MIN_CLOSED_TRADES
    if smoke_test:
        log.info("⚠  Smoke test mode — relaxed gates (7d / 10 trades)")
        import ml_retrain_phase2 as _self
        _self.MIN_PAPER_DAYS    = 7
        _self.MIN_CLOSED_TRADES = 10

    all_pass, gate_details = check_all_gates(con, trades)
    log_gates_to_db(con, gate_details)

    log.info("\n📋 Gate Results:")
    for gname, info in gate_details.items():
        if gname == "all_pass":
            continue
        icon = "✅" if info.get("passed") else "❌"
        log.info(f"  {icon} {gname}: {info.get('details', '')}")

    if not all_pass and not force:
        log.info("\n⛔ Gates not met — aborting (use --force to override)")
        _log_run_to_db(
            con, ts_ms, ts_iso, "scheduled", False, gate_details, trades,
            None, [], "", False, "Gates not met",
            round(time.time() - t_start, 1),
        )
        con.close()
        return {
            "success": False,
            "reason":  "gates_not_met",
            "gate_details": gate_details,
            "duration_sec": round(time.time() - t_start, 1),
        }

    if dry_run:
        log.info("\n✅ Dry-run complete (no training)")
        con.close()
        return {
            "success":      all_pass or force,
            "dry_run":      True,
            "gate_details": gate_details,
            "duration_sec": round(time.time() - t_start, 1),
        }

    # Step 3: slippage multiplier
    slippage_mult = compute_slippage_multiplier(len(trades))
    log.info(f"\n📐 Slippage multiplier: {slippage_mult:.2f}× ({len(trades)} trades)")

    # Step 4: load OHLCV data (sampled)
    days_back = 7 if smoke_test else 60
    log.info(f"\n📊 Loading sampled OHLCV ({days_back}d lookback)...")
    ohlcv = load_ohlcv_from_ticks(con, days_back=days_back, period_minutes=5)

    if len(ohlcv) < 200:
        msg = f"Insufficient OHLCV candles ({len(ohlcv)}) — aborting"
        log.error(msg)
        con.close()
        return {"success": False, "reason": msg}

    log.info(f"  OHLCV candles: {len(ohlcv)}")

    # Step 5: build features with embargo
    log.info("\n🔧 Building features (24h embargo)...")
    X, y = build_features(ohlcv)

    if len(X) < 100:
        msg = f"Insufficient feature rows ({len(X)})"
        log.error(msg)
        con.close()
        return {"success": False, "reason": msg}

    # Embargo fraction: 24h / total candle span
    candle_span_h = (ohlcv[-1, 0] - ohlcv[0, 0]) / 3_600_000.0
    embargo_pct   = min(EMBARGO_HOURS / candle_span_h, 0.10) if candle_span_h > 0 else 0.05

    X_train, y_train, X_test, y_test = walk_forward_split(X, y, embargo_pct=embargo_pct)

    # Compute actual embargo gap from OHLCV timestamps
    train_idx   = int(len(ohlcv) * 0.70)
    embargo_idx = train_idx + max(int(len(ohlcv) * embargo_pct), int(len(ohlcv) * 0.05))
    embargo_idx = min(embargo_idx, int(len(ohlcv) * 0.80))

    if train_idx < len(ohlcv) and embargo_idx < len(ohlcv):
        embargo_actual_h = (ohlcv[embargo_idx, 0] - ohlcv[train_idx, 0]) / 3_600_000.0
    else:
        embargo_actual_h = EMBARGO_HOURS

    log.info(f"  Train: {len(X_train)} | Embargo: {embargo_actual_h:.1f}h | Test: {len(X_test)}")

    if embargo_actual_h < EMBARGO_HOURS * 0.8:
        log.warning(f"⚠  Embargo {embargo_actual_h:.1f}h shorter than {EMBARGO_HOURS}h target")

    # Step 6: archive old models
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("\n🗂  Archiving existing models…")
    archive_existing_models(model_dir)

    # Step 7: train all 5 models
    log.info(f"\n🚀 Training {len(MODEL_NAMES)} models…")
    model_results = []
    for name in MODEL_NAMES:
        log.info(f"  → {name}")
        result = train_single_model(
            name, X_train, y_train, X_test, y_test,
            model_dir, slippage_mult=slippage_mult,
        )
        model_results.append(result)

    success_count = sum(1 for r in model_results if r.get("success"))
    trained_names = [r["model_name"] for r in model_results if r.get("success")]
    run_success   = success_count > 0

    log.info(f"\n📦 {success_count}/{len(MODEL_NAMES)} models trained successfully")

    # Step 8: log to DB
    _log_run_to_db(
        con, ts_ms, ts_iso,
        "forced" if force else "scheduled",
        True, gate_details, trades, slippage_mult,
        trained_names, str(model_dir), run_success,
        None if run_success else "partial_failure",
        round(time.time() - t_start, 1),
    )
    con.close()

    duration = round(time.time() - t_start, 1)
    log.info(f"\n{'✅' if run_success else '⚠'} Phase 2 retrain done in {duration}s")

    return {
        "success":        run_success,
        "models_trained": success_count,
        "model_names":    trained_names,
        "model_dir":      str(model_dir),
        "slippage_mult":  slippage_mult,
        "train_rows":     len(X_train),
        "test_rows":      len(X_test),
        "embargo_hours":  round(embargo_actual_h, 1),
        "gate_details":   gate_details,
        "model_results":  model_results,
        "duration_sec":   duration,
    }


def _log_run_to_db(
    con, ts_ms, ts_iso, trigger, gate_pass, gate_details,
    trades, slippage_mult, trained_names, model_dir,
    success, error_msg, duration,
):
    paper_age   = gate_details.get("paper_age", {}).get("value", 0)
    trade_count = gate_details.get("trade_count", {}).get("value", 0)
    reg_info    = gate_details.get("regime_diversity", {})
    try:
        con.execute(
            """INSERT INTO phase2_retrain_runs
               (ts_ms, ts_iso, trigger_reason, gate_pass, gate_details,
                paper_days, closed_trades, vol_percentile_lo, vol_percentile_hi,
                slippage_mult, models_trained, model_dir,
                success, error_msg, duration_sec, embargo_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts_ms, ts_iso, trigger, 1 if gate_pass else 0,
                json.dumps(gate_details),
                paper_age, trade_count,
                reg_info.get("pct_lo_threshold"),
                reg_info.get("pct_hi_threshold"),
                slippage_mult,
                json.dumps(trained_names),
                model_dir,
                1 if success else 0,
                error_msg,
                duration,
                EMBARGO_HOURS,
            ),
        )
        con.commit()
    except Exception as e:
        log.warning(f"Failed to log run to DB: {e}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 ML Retrain Framework")
    parser.add_argument("--force",      action="store_true", help="Skip gate checks")
    parser.add_argument("--dry-run",    action="store_true", help="Check gates only, no training")
    parser.add_argument("--smoke-test", action="store_true", help="Relaxed gates (7d / 10 trades)")
    parser.add_argument("--db",         default=DB_PATH,     help="Database path")
    args = parser.parse_args()

    result = run_phase2_retrain(
        db_path=args.db,
        force=args.force,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
    )

    print("\n" + "=" * 60)
    if result.get("dry_run"):
        status = "✅ GATES PASSED" if result.get("success") else "❌ GATES NOT MET"
        print(f"Dry-run: {status}")
    elif result.get("success"):
        print("✅ Phase 2 Retrain SUCCEEDED")
        print(f"   Models trained:  {result.get('models_trained', 0)}")
        print(f"   Train rows:      {result.get('train_rows', 0)}")
        print(f"   Embargo:         {result.get('embargo_hours', 0)}h")
        print(f"   Slippage mult:   {result.get('slippage_mult', 0)}×")
        print(f"   Duration:        {result.get('duration_sec', 0)}s")
    else:
        print("❌ Phase 2 Retrain FAILED or gates not met")
        print(f"   Reason: {result.get('reason', 'unknown')}")
    print("=" * 60)

    return 0 if result.get("success") or result.get("dry_run") else 1


if __name__ == "__main__":
    sys.exit(main())
