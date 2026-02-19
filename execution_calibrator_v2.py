#!/usr/bin/env python3
"""
execution_calibrator_v2.py
--------------------------
Phase 2 Enhanced Execution Calibrator.

Enhancements over v1:
  - EMA-weighted slippage (recent 50 trades weighted 3× vs older)
  - Slippage tracked by time-of-day (6-hour UTC buckets)
  - Slippage tracked by volatility regime (trending / ranging / volatile)
  - Position size multipliers output per bucket and per regime
  - Backwards-compatible with existing execution_calibration.json consumers

Outputs:
    data/execution_calibration.json   — read by backtester + position sizer

Usage:
    python3 execution_calibrator_v2.py              # full update
    python3 execution_calibrator_v2.py --min-trades 20
    python3 execution_calibrator_v2.py --verbose    # per-symbol breakdown
    python3 execution_calibrator_v2.py --dry-run    # print without writing
    python3 execution_calibrator_v2.py --db PATH
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT       = Path(__file__).resolve().parent
DB_PATH    = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
CALIB_PATH = ROOT / "data" / "execution_calibration.json"

# ── Assumed baseline values (from .env / backtester defaults) ─────────────────
ASSUMED_FEE_PCT      = 0.04   # % per side
ASSUMED_SLIPPAGE_PCT = 0.02   # % per side
ASSUMED_HOLD_MIN     = 60.0   # minutes

# ── EMA config ────────────────────────────────────────────────────────────────
EMA_ALPHA_BASE    = 0.15    # smoothing factor for old trades
RECENT_N          = 50      # recent N trades get boosted weight
RECENT_WEIGHT_MULT = 3.0   # boost factor for recent trades

# ── Time-of-day buckets (UTC hours, 6-hour windows) ───────────────────────────
TOD_BUCKETS: Dict[str, tuple] = {
    "00-06": (0,  6),
    "06-12": (6,  12),
    "12-18": (12, 18),
    "18-24": (18, 24),
}

# ── Volatility regime thresholds ──────────────────────────────────────────────
# pnl_pct used as proxy when tick data unavailable
VOL_TRENDING_LO = 0.30   # pnl > 0.3% absolute → trending
VOL_VOLATILE_HI = 2.00   # pnl > 2.0% absolute → volatile


# ─── DB helpers ───────────────────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def load_closed_trades(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all closed trades ordered oldest → newest (required for EMA)."""
    rows = con.execute(
        """SELECT id, symbol, side, entry_price, exit_price,
                  opened_ts_ms, closed_ts_ms, pnl_pct, reason
           FROM paper_trades
           WHERE status = 'CLOSED'
             AND exit_price  IS NOT NULL
             AND entry_price IS NOT NULL
             AND opened_ts_ms IS NOT NULL
             AND closed_ts_ms IS NOT NULL
           ORDER BY closed_ts_ms ASC"""
    ).fetchall()
    return [dict(r) for r in rows]


# ─── Exit / regime classification ────────────────────────────────────────────

def classify_exit_reason(reason: Optional[str]) -> str:
    if not reason:
        return "OTHER"
    r = reason.upper()
    if "TP" in r:
        return "TP"
    if "SL" in r:
        return "SL"
    if "TIME" in r or "EXPIRE" in r:
        return "TIME"
    return "OTHER"


def classify_tod_bucket(ts_ms: int) -> str:
    dt   = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    hour = dt.hour
    for name, (lo, hi) in TOD_BUCKETS.items():
        if lo <= hour < hi:
            return name
    return "18-24"


def classify_vol_regime(trade: Dict[str, Any]) -> str:
    """
    Classify trade into trending / ranging / volatile.
    Uses absolute |pnl_pct| as a proxy for regime volatility.
    """
    pnl = abs(float(trade.get("pnl_pct") or 0))
    if pnl >= VOL_VOLATILE_HI:
        return "volatile"
    if pnl >= VOL_TRENDING_LO:
        return "trending"
    return "ranging"


# ─── Slippage inference ────────────────────────────────────────────────────────

def infer_slippage_per_side(trade: Dict[str, Any]) -> float:
    """
    Estimate round-trip slippage per side.
    = (raw_gross_pct - reported_pnl_pct) / 2
    """
    entry  = float(trade.get("entry_price") or 0)
    exit_  = float(trade.get("exit_price")  or 0)
    side   = trade.get("side", "BUY")
    pnl    = float(trade.get("pnl_pct")    or 0)

    if entry <= 0 or exit_ <= 0:
        return 0.0

    if side == "BUY":
        raw_gross = (exit_ - entry) / entry * 100.0
    else:
        raw_gross = (entry - exit_) / entry * 100.0

    friction  = raw_gross - pnl
    per_side  = friction / 2.0
    return max(0.0, per_side)


# ─── EMA computation ──────────────────────────────────────────────────────────

def compute_ema_slippage(
    slippages: List[float],
    recent_n: int = RECENT_N,
    recent_mult: float = RECENT_WEIGHT_MULT,
    alpha_base: float = EMA_ALPHA_BASE,
) -> float:
    """
    EMA of slippage series (oldest → newest).
    Last `recent_n` values use alpha = alpha_base × recent_mult (capped at 0.5).
    """
    if not slippages:
        return ASSUMED_SLIPPAGE_PCT

    n        = len(slippages)
    boundary = max(0, n - recent_n)
    ema      = slippages[0]

    for i, slip in enumerate(slippages[1:], 1):
        alpha = alpha_base * recent_mult if i >= boundary else alpha_base
        alpha = min(alpha, 0.50)
        ema   = alpha * slip + (1 - alpha) * ema

    return max(0.0, ema)


# ─── Position size multiplier ─────────────────────────────────────────────────

def friction_to_pos_mult(slip_per_side: float) -> float:
    """
    Convert actual slippage to position size multiplier.
      actual < assumed → allow larger positions (up to 1.2×)
      actual > assumed → reduce positions   (down to 0.5×)
    """
    assumed_total = (ASSUMED_FEE_PCT + ASSUMED_SLIPPAGE_PCT) * 2
    actual_total  = slip_per_side * 2
    ratio = assumed_total / max(actual_total, 0.001)
    return round(min(1.20, max(0.50, ratio)), 3)


# ─── Core calibration ─────────────────────────────────────────────────────────

def compute_calibration(
    trades: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Full Phase 2 calibration.
    Trades must be ordered oldest → newest for correct EMA.
    """
    if not trades:
        return _placeholder_calibration("no closed trades")

    all_slip   : List[float] = []
    all_hold   : List[float] = []
    all_pnls   : List[float] = []
    tp_pnls    : List[float] = []
    sl_pnls    : List[float] = []
    exit_types : Dict[str, int] = {"TP": 0, "SL": 0, "TIME": 0, "OTHER": 0}

    tod_slips    : Dict[str, List[float]] = {k: [] for k in TOD_BUCKETS}
    regime_slips : Dict[str, List[float]] = {
        "trending": [], "ranging": [], "volatile": []
    }
    symbol_pnls  : Dict[str, List[float]] = {}

    for t in trades:
        hold_ms  = (t["closed_ts_ms"] - t["opened_ts_ms"])
        hold_min = hold_ms / 60_000.0
        all_hold.append(hold_min)

        slip = infer_slippage_per_side(t)
        all_slip.append(slip)

        pnl = float(t.get("pnl_pct") or 0)
        all_pnls.append(pnl)

        reason = classify_exit_reason(t.get("reason"))
        exit_types[reason] = exit_types.get(reason, 0) + 1

        if reason == "TP":
            tp_pnls.append(pnl)
        elif reason == "SL":
            sl_pnls.append(pnl)

        tod    = classify_tod_bucket(t["opened_ts_ms"])
        tod_slips[tod].append(slip)

        regime = classify_vol_regime(t)
        regime_slips[regime].append(slip)

        sym = t.get("symbol", "UNKNOWN")
        symbol_pnls.setdefault(sym, []).append(pnl)

    n      = len(trades)
    wins   = sum(1 for p in all_pnls if p > 0)
    losses = n - wins

    # ── Global EMA slippage ────────────────────────────────────────────────────
    ema_slip  = compute_ema_slippage(all_slip)
    avg_slip  = min(ema_slip, 0.15)   # cap at 0.15 %/side
    pos_mult  = friction_to_pos_mult(avg_slip)

    # ── Time-of-day breakdown ──────────────────────────────────────────────────
    tod_stats: Dict[str, Dict] = {}
    for bucket, slips in tod_slips.items():
        if slips:
            ema   = min(compute_ema_slippage(slips), 0.15)
            raw   = sum(slips) / len(slips)
            pmult = friction_to_pos_mult(ema)
        else:
            ema   = ASSUMED_SLIPPAGE_PCT
            raw   = ASSUMED_SLIPPAGE_PCT
            pmult = 1.0
        tod_stats[bucket] = {
            "count":    len(slips),
            "ema_slip": round(ema, 4),
            "raw_avg":  round(raw, 4),
            "pos_mult": pmult,
        }

    # ── Volatility regime breakdown ────────────────────────────────────────────
    regime_stats: Dict[str, Dict] = {}
    for regime, slips in regime_slips.items():
        if slips:
            ema   = min(compute_ema_slippage(slips), 0.15)
            raw   = sum(slips) / len(slips)
            pmult = friction_to_pos_mult(ema)
        else:
            ema   = ASSUMED_SLIPPAGE_PCT
            raw   = ASSUMED_SLIPPAGE_PCT
            pmult = 1.0
        regime_stats[regime] = {
            "count":    len(slips),
            "ema_slip": round(ema, 4),
            "raw_avg":  round(raw, 4),
            "pos_mult": pmult,
        }

    # ── Aggregate stats ────────────────────────────────────────────────────────
    avg_hold_min    = sum(all_hold) / n
    median_hold_min = sorted(all_hold)[n // 2]
    tp_exits        = exit_types.get("TP",    0)
    sl_exits        = exit_types.get("SL",    0)
    time_exits      = exit_types.get("TIME",  0)
    fill_rate       = (tp_exits + sl_exits) / n if n > 0 else 0.0

    gains    = sum(p for p in all_pnls if p > 0)
    loss_sum = abs(sum(p for p in all_pnls if p < 0))
    pf       = gains / loss_sum if loss_sum > 0 else (gains or 1.0)

    avg_tp_pnl = sum(tp_pnls) / len(tp_pnls) if tp_pnls else 0.0
    avg_sl_pnl = sum(sl_pnls) / len(sl_pnls) if sl_pnls else 0.0
    shf        = sl_exits / n if n > 0 else 0.0

    # ── Symbol breakdown (verbose only) ───────────────────────────────────────
    symbol_breakdown: Dict[str, Any] = {}
    if verbose:
        for sym, sym_pnls in sorted(symbol_pnls.items()):
            sw = sum(1 for p in sym_pnls if p > 0)
            symbol_breakdown[sym] = {
                "trades":   len(sym_pnls),
                "win_rate": round(sw / len(sym_pnls), 3),
                "avg_pnl":  round(sum(sym_pnls) / len(sym_pnls), 4),
            }

    return {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "data_source":        "paper_trades (closed)",
        "calibrator_version": "v2",
        "trade_count":        n,
        "win_rate":           round(wins / n, 4),
        "loss_rate":          round(losses / n, 4),
        "profit_factor":      round(pf, 3),
        "execution": {
            # v2 canonical field (ema-weighted)
            "ema_slippage_per_side_pct":      round(avg_slip, 4),
            # legacy alias — keep for backward compatibility
            "avg_slippage_per_side_pct":      round(avg_slip, 4),
            "assumed_slippage_per_side_pct":  ASSUMED_SLIPPAGE_PCT,
            "assumed_fee_per_side_pct":       ASSUMED_FEE_PCT,
            "total_friction_pct":             round(avg_slip * 2, 4),
            "assumed_friction_pct":           round((ASSUMED_FEE_PCT + ASSUMED_SLIPPAGE_PCT) * 2, 4),
            "fill_rate":                      round(fill_rate, 4),
            "position_size_multiplier":       pos_mult,
            "recent_n_weighted":              RECENT_N,
            "recent_weight_mult":             RECENT_WEIGHT_MULT,
        },
        "hold_times": {
            "avg_min":                round(avg_hold_min, 2),
            "median_min":             round(median_hold_min, 2),
            "assumed_avg_min":        ASSUMED_HOLD_MIN,
            "pct_hitting_time_limit": round(time_exits / n, 4),
        },
        "exit_distribution": {
            "TP":       tp_exits,
            "SL":       sl_exits,
            "TIME":     time_exits,
            "OTHER":    exit_types.get("OTHER", 0),
            "TP_pct":   round(tp_exits  / n, 3),
            "SL_pct":   round(sl_exits  / n, 3),
            "TIME_pct": round(time_exits / n, 3),
        },
        "exit_quality": {
            "stop_hit_frequency": round(shf, 4),
            "avg_tp_pnl_pct":     round(avg_tp_pnl, 4),
            "avg_sl_pnl_pct":     round(avg_sl_pnl, 4),
            "rr_ratio": (
                round(abs(avg_tp_pnl / avg_sl_pnl), 3)
                if avg_sl_pnl != 0 else 0.0
            ),
        },
        "time_of_day":    tod_stats,
        "regime_slippage": regime_stats,
        "recommendations": _build_recommendations(avg_slip, fill_rate, shf, pf, pos_mult),
        "symbol_breakdown": symbol_breakdown,
    }


def _build_recommendations(
    slip: float,
    fill_rate: float,
    shf: float,
    pf: float,
    size_mult: float,
) -> List[str]:
    recs = []
    if slip > ASSUMED_SLIPPAGE_PCT * 2:
        recs.append(
            f"⚠  EMA slippage ({slip:.3f}%/side) is >2× assumed "
            f"({ASSUMED_SLIPPAGE_PCT}%). Widen SL or reduce entry aggression."
        )
    elif slip < ASSUMED_SLIPPAGE_PCT * 0.5:
        recs.append(
            f"✅ EMA slippage ({slip:.3f}%/side) better than assumed — model is conservative."
        )
    if fill_rate < 0.50:
        recs.append(
            f"⚠  Fill rate {fill_rate:.1%} — most exits via time limit. "
            "Consider tighter TP/SL or shorter hold window."
        )
    if shf > 0.40:
        recs.append(f"⚠  Stop hit frequency {shf:.1%} is high — stops may be too tight.")
    if pf < 1.0:
        recs.append(f"🔴 Profit factor {pf:.2f} < 1.0 — strategy losing overall.")
    elif pf >= 1.5:
        recs.append(f"✅ Profit factor {pf:.2f} is healthy.")
    if size_mult < 0.75:
        recs.append(f"📉 Position size reduced to {size_mult:.2f}× (excess friction).")
    elif size_mult > 1.0:
        recs.append(f"📈 Position size {size_mult:.2f}× — friction better than assumed.")
    if not recs:
        recs.append("✅ Execution parameters within normal range.")
    return recs


def _placeholder_calibration(reason: str = "insufficient data") -> Dict[str, Any]:
    tod_default = {
        k: {"count": 0, "ema_slip": ASSUMED_SLIPPAGE_PCT,
            "raw_avg": ASSUMED_SLIPPAGE_PCT, "pos_mult": 1.0}
        for k in TOD_BUCKETS
    }
    regime_default = {
        r: {"count": 0, "ema_slip": ASSUMED_SLIPPAGE_PCT,
            "raw_avg": ASSUMED_SLIPPAGE_PCT, "pos_mult": 1.0}
        for r in ("trending", "ranging", "volatile")
    }
    return {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "data_source":        "placeholder",
        "calibrator_version": "v2",
        "placeholder_reason": reason,
        "trade_count":        0,
        "win_rate":           0.5,
        "loss_rate":          0.5,
        "profit_factor":      1.0,
        "execution": {
            "ema_slippage_per_side_pct":     ASSUMED_SLIPPAGE_PCT,
            "avg_slippage_per_side_pct":     ASSUMED_SLIPPAGE_PCT,  # legacy alias
            "assumed_slippage_per_side_pct": ASSUMED_SLIPPAGE_PCT,
            "assumed_fee_per_side_pct":      ASSUMED_FEE_PCT,
            "total_friction_pct":            (ASSUMED_FEE_PCT + ASSUMED_SLIPPAGE_PCT) * 2,
            "assumed_friction_pct":          (ASSUMED_FEE_PCT + ASSUMED_SLIPPAGE_PCT) * 2,
            "fill_rate":                     1.0,
            "position_size_multiplier":      1.0,
            "recent_n_weighted":             RECENT_N,
            "recent_weight_mult":            RECENT_WEIGHT_MULT,
        },
        "hold_times": {
            "avg_min":                ASSUMED_HOLD_MIN,
            "median_min":             ASSUMED_HOLD_MIN,
            "assumed_avg_min":        ASSUMED_HOLD_MIN,
            "pct_hitting_time_limit": 0.0,
        },
        "exit_distribution": {
            "TP": 0, "SL": 0, "TIME": 0, "OTHER": 0,
            "TP_pct": 0.0, "SL_pct": 0.0, "TIME_pct": 0.0,
        },
        "exit_quality": {
            "stop_hit_frequency": 0.0,
            "avg_tp_pnl_pct":     0.0,
            "avg_sl_pnl_pct":     0.0,
            "rr_ratio":           0.0,
        },
        "time_of_day":     tod_default,
        "regime_slippage": regime_default,
        "recommendations": [f"Placeholder — {reason}. Updates after first paper trades close."],
        "symbol_breakdown": {},
    }


# ─── File I/O ─────────────────────────────────────────────────────────────────

def write_calibration(data: Dict[str, Any]) -> None:
    """Atomic write via temp-file rename."""
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.rename(CALIB_PATH)


def load_existing_calibration() -> Dict[str, Any]:
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as fh:
            return json.load(fh)
    return {}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution Calibrator v2 — Phase 2 enhanced feedback loop"
    )
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum closed trades required (default: 10)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Include per-symbol breakdown")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print calibration without writing file")
    parser.add_argument("--db",       default=DB_PATH,
                        help="Database path")
    args = parser.parse_args()

    print("📊 Execution Calibrator v2 — Phase 2 Enhanced")
    print(f"   DB:     {args.db}")
    print(f"   Output: {CALIB_PATH}")

    con    = connect(args.db)
    trades = load_closed_trades(con)
    con.close()

    print(f"   Found {len(trades)} closed paper trades")

    if len(trades) < args.min_trades:
        print(f"   ⚠  {len(trades)} < {args.min_trades} minimum — writing placeholder.")
        calib = _placeholder_calibration(
            f"only {len(trades)} trades (need {args.min_trades})"
        )
    else:
        calib = compute_calibration(trades, verbose=args.verbose)

    exec_  = calib["execution"]
    hold_  = calib["hold_times"]
    exit_q = calib["exit_quality"]

    print(f"\n📋 Calibration Results:")
    print(f"   Trades:           {calib['trade_count']}")
    print(f"   Win rate:         {calib['win_rate']:.1%}")
    print(f"   Profit factor:    {calib['profit_factor']:.3f}")
    print(f"   EMA slip/side:    {exec_['ema_slippage_per_side_pct']:.4f}%")
    print(f"   Fill rate:        {exec_['fill_rate']:.1%}")
    print(f"   Avg hold:         {hold_['avg_min']:.1f} min")
    print(f"   Stop-hit freq:    {exit_q['stop_hit_frequency']:.1%}")
    print(f"   Global pos mult:  {exec_['position_size_multiplier']:.3f}×")

    if calib.get("time_of_day"):
        print(f"\n🕐 Time-of-Day (6h UTC buckets):")
        for bucket, stats in sorted(calib["time_of_day"].items()):
            print(
                f"   {bucket} UTC:  "
                f"slip={stats['ema_slip']:.4f}%  "
                f"pos_mult={stats['pos_mult']:.3f}×  "
                f"n={stats['count']}"
            )

    if calib.get("regime_slippage"):
        print(f"\n📈 Volatility Regime:")
        for regime, stats in sorted(calib["regime_slippage"].items()):
            print(
                f"   {regime:<10}:  "
                f"slip={stats['ema_slip']:.4f}%  "
                f"pos_mult={stats['pos_mult']:.3f}×  "
                f"n={stats['count']}"
            )

    print(f"\n💡 Recommendations:")
    for r in calib.get("recommendations", []):
        print(f"   {r}")

    if args.verbose and calib.get("symbol_breakdown"):
        print(f"\n📊 Per-Symbol:")
        for sym, stats in sorted(calib["symbol_breakdown"].items()):
            print(
                f"   {sym:<15}  "
                f"n={stats['trades']:<5}  "
                f"wr={stats['win_rate']:.1%}  "
                f"avg={stats['avg_pnl']:+.3f}%"
            )

    if not args.dry_run:
        write_calibration(calib)
        print(f"\n✅ Written → {CALIB_PATH}")
    else:
        print("\n[dry-run] Not writing file.")
        print(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
