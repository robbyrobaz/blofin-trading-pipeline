#!/usr/bin/env python3
"""
execution_calibrator.py
-----------------------
Phase 2 Execution Calibrator — Enhanced feedback loop.

Reads CLOSED paper trades and derives realistic execution parameters.
No look-ahead bias: only reads trades that have already closed.

Phase 2 enhancements over Phase 1:
  - Slippage tracked by time-of-day bucket (00-06, 06-12, 12-18, 18-24 UTC)
  - Slippage tracked by volatility regime (trending/ranging/volatile)
  - EMA (exponential moving average) instead of flat average
  - Recent 50 trades weighted 3× heavier than older data
  - Position size multipliers output per regime

Outputs:
    data/execution_calibration.json   — read by backtester + position sizer

Usage:
    python3 execution_calibrator.py              # update from live DB
    python3 execution_calibrator.py --min-trades 20  # require at least N trades
    python3 execution_calibrator.py --verbose    # per-symbol breakdown
    python3 execution_calibrator.py --dry-run    # print without writing
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parent
DB_PATH    = os.getenv("BLOFIN_DB_PATH", str(ROOT / "data" / "blofin_monitor.db"))
CALIB_PATH = ROOT / "data" / "execution_calibration.json"

# Assumed values the backtester currently uses (from .env defaults)
ASSUMED_FEE_PCT      = 0.04   # per side
ASSUMED_SLIPPAGE_PCT = 0.02   # per side
ASSUMED_HOLD_MIN     = 60.0   # minutes

# EMA config
EMA_ALPHA_BASE    = 0.15      # base smoothing factor (lower = slower)
RECENT_N          = 50        # recent N trades weighted 3× heavier
RECENT_WEIGHT_MULT = 3.0

# Time-of-day buckets (UTC hours)
TOD_BUCKETS = {
    "00-06": (0, 6),
    "06-12": (6, 12),
    "12-18": (12, 18),
    "18-24": (18, 24),
}


# ─── DB helpers ───────────────────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def load_closed_trades(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Load all closed paper trades with full detail, ordered oldest→newest."""
    rows = con.execute(
        """SELECT id, symbol, side, entry_price, exit_price,
                  opened_ts_ms, closed_ts_ms, pnl_pct, reason
           FROM paper_trades
           WHERE status = 'CLOSED'
             AND exit_price IS NOT NULL
             AND entry_price IS NOT NULL
             AND opened_ts_ms IS NOT NULL
             AND closed_ts_ms IS NOT NULL
           ORDER BY closed_ts_ms ASC"""  # oldest first for EMA
    ).fetchall()
    return [dict(r) for r in rows]


# ─── Exit classification ───────────────────────────────────────────────────────

def classify_exit_reason(reason: str) -> str:
    if reason and "TP" in reason:
        return "TP"
    if reason and "SL" in reason:
        return "SL"
    if reason and "TIME" in reason:
        return "TIME"
    return "OTHER"


# ─── Slippage inference ────────────────────────────────────────────────────────

def infer_slippage(trade: Dict[str, Any]) -> float:
    """
    Estimate realised slippage from a paper trade.
    Compares gross price move to reported pnl_pct.
    """
    entry = trade["entry_price"]
    exit_ = trade["exit_price"]
    side  = trade["side"]
    pnl   = trade.get("pnl_pct", 0.0)

    if entry <= 0:
        return 0.0

    if side == "BUY":
        raw_pct = (exit_ - entry) / entry * 100.0
    else:
        raw_pct = (entry - exit_) / entry * 100.0

    friction = raw_pct - pnl
    per_side = friction / 2.0
    return max(0.0, per_side)


# ─── Time-of-day classification ───────────────────────────────────────────────

def classify_tod_bucket(ts_ms: int) -> str:
    """Return time-of-day bucket key for a Unix timestamp in ms."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    hour = dt.hour
    for bucket_name, (lo, hi) in TOD_BUCKETS.items():
        if lo <= hour < hi:
            return bucket_name
    return "18-24"  # fallback


# ─── Volatility regime classification ────────────────────────────────────────

def classify_vol_regime(
    trade: Dict[str, Any],
    vol_pct: Optional[float] = None,
) -> str:
    """
    Classify volatility regime as trending / ranging / volatile.
    Uses pnl_pct range and exit reason as proxies.
    vol_pct is the HL range / close if available.
    """
    pnl    = abs(trade.get("pnl_pct", 0.0))
    reason = classify_exit_reason(trade.get("reason", ""))

    if vol_pct is not None:
        if vol_pct > 0.005:   # >0.5% HL range → volatile
            return "volatile"
        elif vol_pct < 0.001:  # <0.1% HL range → ranging
            return "ranging"
        else:
            return "trending"

    # Fallback: use pnl as proxy for regime
    if pnl > 2.0:
        return "volatile"
    elif pnl < 0.3:
        return "ranging"
    else:
        return "trending"


# ─── EMA computation ──────────────────────────────────────────────────────────

def compute_ema_slippage(
    slippages: List[float],
    recent_n: int = RECENT_N,
    recent_mult: float = RECENT_WEIGHT_MULT,
    alpha_base: float = EMA_ALPHA_BASE,
) -> float:
    """
    Compute EMA of slippage, weighting recent N trades heavier.

    Strategy:
      - Process trades oldest → newest
      - For the last `recent_n` trades, use alpha = alpha_base * recent_mult
      - Otherwise use alpha = alpha_base
    """
    if not slippages:
        return ASSUMED_SLIPPAGE_PCT

    n = len(slippages)
    boundary = max(0, n - recent_n)

    ema = slippages[0]
    for i, slip in enumerate(slippages[1:], 1):
        alpha = alpha_base * recent_mult if i >= boundary else alpha_base
        alpha = min(alpha, 0.5)  # cap smoothing
        ema = alpha * slip + (1 - alpha) * ema

    return max(0.0, ema)


# ─── Core calibration ─────────────────────────────────────────────────────────

def compute_calibration(
    trades: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Full calibration with EMA slippage, time-of-day, and regime breakdown.
    Trades should be ordered oldest → newest for EMA.
    """
    if not trades:
        return _placeholder_calibration("no closed trades")

    # Per-trade computed fields
    all_slip   = []
    all_hold   = []
    all_pnls   = []
    tp_pnls    = []
    sl_pnls    = []
    exit_types = {"TP": 0, "SL": 0, "TIME": 0, "OTHER": 0}

    # Buckets
    tod_slips  : Dict[str, List[float]] = {k: [] for k in TOD_BUCKETS}
    reg_slips  : Dict[str, List[float]] = {"trending": [], "ranging": [], "volatile": []}
    symbol_stats: Dict[str, List[float]] = {}

    for t in trades:
        hold_ms  = t["closed_ts_ms"] - t["opened_ts_ms"]
        hold_min = hold_ms / 60_000.0
        all_hold.append(hold_min)

        slip   = infer_slippage(t)
        all_slip.append(slip)

        pnl    = t.get("pnl_pct", 0.0)
        all_pnls.append(pnl)

        reason = classify_exit_reason(t.get("reason", ""))
        exit_types[reason] = exit_types.get(reason, 0) + 1

        if reason == "TP":
            tp_pnls.append(pnl)
        elif reason == "SL":
            sl_pnls.append(pnl)

        # Time-of-day
        tod = classify_tod_bucket(t["opened_ts_ms"])
        tod_slips[tod].append(slip)

        # Volatility regime (pnl-proxy)
        regime = classify_vol_regime(t)
        reg_slips[regime].append(slip)

        # Symbol stats
        sym = t["symbol"]
        symbol_stats.setdefault(sym, []).append(pnl)

    n = len(trades)
    wins   = sum(1 for p in all_pnls if p > 0)
    losses = n - wins

    # ── EMA slippage (weighted: recent 50 trades × 3×) ───────────────────────
    ema_slip = compute_ema_slippage(all_slip)
    avg_slip = min(ema_slip, 0.15)  # cap at 0.15%/side

    # ── Time-of-day slippage ──────────────────────────────────────────────────
    tod_stats = {}
    for bucket, slips in tod_slips.items():
        if slips:
            ema = compute_ema_slippage(slips)
            tod_stats[bucket] = {
                "count":      len(slips),
                "ema_slip":   round(min(ema, 0.15), 4),
                "raw_avg":    round(sum(slips) / len(slips), 4),
                "pos_mult":   _friction_to_pos_mult(min(ema, 0.15)),
            }
        else:
            tod_stats[bucket] = {
                "count": 0, "ema_slip": ASSUMED_SLIPPAGE_PCT,
                "raw_avg": ASSUMED_SLIPPAGE_PCT, "pos_mult": 1.0,
            }

    # ── Volatility regime slippage ────────────────────────────────────────────
    regime_stats = {}
    for regime, slips in reg_slips.items():
        if slips:
            ema = compute_ema_slippage(slips)
            regime_stats[regime] = {
                "count":      len(slips),
                "ema_slip":   round(min(ema, 0.15), 4),
                "raw_avg":    round(sum(slips) / len(slips), 4),
                "pos_mult":   _friction_to_pos_mult(min(ema, 0.15)),
            }
        else:
            regime_stats[regime] = {
                "count": 0, "ema_slip": ASSUMED_SLIPPAGE_PCT,
                "raw_avg": ASSUMED_SLIPPAGE_PCT, "pos_mult": 1.0,
            }

    # ── Core stats ────────────────────────────────────────────────────────────
    avg_hold_min    = sum(all_hold) / n
    median_hold_min = sorted(all_hold)[n // 2]

    tp_exits   = exit_types.get("TP", 0)
    sl_exits   = exit_types.get("SL", 0)
    time_exits = exit_types.get("TIME", 0)
    fill_rate  = (tp_exits + sl_exits) / n if n > 0 else 0.0

    gains    = sum(p for p in all_pnls if p > 0)
    loss_sum = abs(sum(p for p in all_pnls if p < 0))
    pf       = (gains / loss_sum) if loss_sum > 0 else gains

    avg_tp_pnl = sum(tp_pnls) / len(tp_pnls) if tp_pnls else 0.0
    avg_sl_pnl = sum(sl_pnls) / len(sl_pnls) if sl_pnls else 0.0
    shf        = sl_exits / n if n > 0 else 0.0

    # ── Global position size multiplier ──────────────────────────────────────
    pos_mult = _friction_to_pos_mult(avg_slip)

    # ── Per-symbol breakdown ──────────────────────────────────────────────────
    symbol_breakdown = {}
    if verbose:
        for sym, sym_pnls in sorted(symbol_stats.items()):
            sw = sum(1 for p in sym_pnls if p > 0)
            symbol_breakdown[sym] = {
                "trades":   len(sym_pnls),
                "win_rate": round(sw / len(sym_pnls), 3),
                "avg_pnl":  round(sum(sym_pnls) / len(sym_pnls), 4),
            }

    return {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "data_source":     "paper_trades (closed)",
        "calibrator_version": "v2",
        "trade_count":     n,
        "win_rate":        round(wins / n, 4),
        "loss_rate":       round(losses / n, 4),
        "profit_factor":   round(pf, 3),
        "execution": {
            "ema_slippage_per_side_pct":      round(avg_slip, 4),
            "avg_slippage_per_side_pct":      round(avg_slip, 4),  # legacy alias
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
            "avg_min":             round(avg_hold_min, 2),
            "median_min":          round(median_hold_min, 2),
            "assumed_avg_min":     ASSUMED_HOLD_MIN,
            "pct_hitting_time_limit": round(time_exits / n, 4),
        },
        "exit_distribution": {
            "TP":       tp_exits,
            "SL":       sl_exits,
            "TIME":     time_exits,
            "OTHER":    exit_types.get("OTHER", 0),
            "TP_pct":   round(tp_exits / n, 3),
            "SL_pct":   round(sl_exits / n, 3),
            "TIME_pct": round(time_exits / n, 3),
        },
        "exit_quality": {
            "stop_hit_frequency": round(shf, 4),
            "avg_tp_pnl_pct":     round(avg_tp_pnl, 4),
            "avg_sl_pnl_pct":     round(avg_sl_pnl, 4),
            "rr_ratio":           round(abs(avg_tp_pnl / avg_sl_pnl), 3) if avg_sl_pnl != 0 else 0.0,
        },
        "time_of_day": tod_stats,
        "regime_slippage": regime_stats,
        "recommendations": _build_recommendations(avg_slip, fill_rate, shf, pf, pos_mult),
        "symbol_breakdown": symbol_breakdown if verbose else {},
    }


def _friction_to_pos_mult(slip_per_side: float) -> float:
    """
    Convert actual slippage to a position size multiplier.
    Lower actual friction → allow larger positions (up to 1.2×).
    Higher actual friction → reduce position (down to 0.5×).
    """
    assumed_total = (ASSUMED_FEE_PCT + ASSUMED_SLIPPAGE_PCT) * 2
    actual_total  = slip_per_side * 2
    ratio = assumed_total / max(actual_total, 0.01)
    return round(min(1.2, max(0.5, ratio)), 3)


def _build_recommendations(
    slip: float, fill_rate: float, shf: float, pf: float, size_mult: float
) -> List[str]:
    recs = []
    if slip > ASSUMED_SLIPPAGE_PCT * 2:
        recs.append(
            f"⚠  EMA slippage ({slip:.3f}%/side) is >2× assumed ({ASSUMED_SLIPPAGE_PCT}%). "
            "Widen SL or reduce entry aggression."
        )
    elif slip < ASSUMED_SLIPPAGE_PCT * 0.5:
        recs.append(f"✅ EMA slippage ({slip:.3f}%/side) better than assumed — model is conservative.")
    if fill_rate < 0.5:
        recs.append(
            f"⚠  Fill rate {fill_rate:.1%} — most exits via time limit, not TP/SL. "
            "Consider tighter TP/SL or shorter hold window."
        )
    if shf > 0.4:
        recs.append(f"⚠  Stop hit frequency {shf:.1%} is high — stops may be too tight.")
    if pf < 1.0:
        recs.append(f"🔴 Profit factor {pf:.2f} < 1.0 — strategy losing overall.")
    elif pf >= 1.5:
        recs.append(f"✅ Profit factor {pf:.2f} is healthy.")
    if size_mult < 0.75:
        recs.append(f"📉 Position size reduced to {size_mult:.2f}× due to excess friction.")
    elif size_mult > 1.0:
        recs.append(f"📈 Position size {size_mult:.2f}× — friction better than assumed.")
    if not recs:
        recs.append("✅ Execution parameters within normal range.")
    return recs


def _placeholder_calibration(reason: str = "insufficient data") -> Dict[str, Any]:
    """Return safe defaults when there's no data yet."""
    tod_default = {k: {
        "count": 0, "ema_slip": ASSUMED_SLIPPAGE_PCT,
        "raw_avg": ASSUMED_SLIPPAGE_PCT, "pos_mult": 1.0
    } for k in TOD_BUCKETS}

    regime_default = {r: {
        "count": 0, "ema_slip": ASSUMED_SLIPPAGE_PCT,
        "raw_avg": ASSUMED_SLIPPAGE_PCT, "pos_mult": 1.0
    } for r in ("trending", "ranging", "volatile")}

    return {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "data_source":         "placeholder",
        "calibrator_version":  "v2",
        "placeholder_reason":  reason,
        "trade_count":         0,
        "win_rate":            0.5,
        "loss_rate":           0.5,
        "profit_factor":       1.0,
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
            "avg_min": ASSUMED_HOLD_MIN, "median_min": ASSUMED_HOLD_MIN,
            "assumed_avg_min": ASSUMED_HOLD_MIN, "pct_hitting_time_limit": 0.0,
        },
        "exit_distribution": {
            "TP": 0, "SL": 0, "TIME": 0, "OTHER": 0,
            "TP_pct": 0.0, "SL_pct": 0.0, "TIME_pct": 0.0,
        },
        "exit_quality": {
            "stop_hit_frequency": 0.0, "avg_tp_pnl_pct": 0.0,
            "avg_sl_pnl_pct": 0.0, "rr_ratio": 0.0,
        },
        "time_of_day":    tod_default,
        "regime_slippage": regime_default,
        "recommendations": [f"Placeholder — {reason}. Will update after first paper trades close."],
        "symbol_breakdown": {},
    }


def load_existing_calibration() -> Dict[str, Any]:
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            return json.load(f)
    return {}


def write_calibration(data: Dict[str, Any]) -> None:
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIB_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(CALIB_PATH)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Execution Calibrator v2 — Phase 2 feedback loop")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum closed trades required (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Include per-symbol breakdown")
    parser.add_argument("--db",      default=DB_PATH, help="Database path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print calibration without writing file")
    args = parser.parse_args()

    print("📊 Execution Calibrator v2 — Phase 2 Enhanced Feedback Loop")
    print(f"   DB:     {args.db}")
    print(f"   Output: {CALIB_PATH}")

    con    = connect(args.db)
    trades = load_closed_trades(con)
    con.close()

    print(f"   Found {len(trades)} closed paper trades")

    if len(trades) < args.min_trades:
        print(f"   ⚠  Only {len(trades)} trades < min {args.min_trades}. Writing placeholder.")
        calib = _placeholder_calibration(
            f"only {len(trades)} closed trades (need {args.min_trades})"
        )
    else:
        calib = compute_calibration(trades, verbose=args.verbose)

    exec_  = calib["execution"]
    hold_  = calib["hold_times"]
    exit_q = calib["exit_quality"]

    print(f"\n📋 Calibration Results (v2 — EMA-weighted):")
    print(f"   Trades analysed:      {calib['trade_count']}")
    print(f"   Win rate:             {calib['win_rate']:.1%}")
    print(f"   Profit factor:        {calib['profit_factor']:.3f}")
    print(f"   EMA slippage/side:    {exec_['ema_slippage_per_side_pct']:.4f}%")
    print(f"   Fill rate:            {exec_['fill_rate']:.1%}")
    print(f"   Avg hold time:        {hold_['avg_min']:.1f} min")
    print(f"   Stop hit frequency:   {exit_q['stop_hit_frequency']:.1%}")
    print(f"   Global pos mult:      {exec_['position_size_multiplier']:.3f}×")

    if "time_of_day" in calib:
        print(f"\n🕐 Time-of-Day Slippage & Position Multipliers:")
        for bucket, stats in sorted(calib["time_of_day"].items()):
            print(f"   {bucket} UTC:  slip={stats['ema_slip']:.4f}%  pos_mult={stats['pos_mult']:.3f}×  trades={stats['count']}")

    if "regime_slippage" in calib:
        print(f"\n📈 Volatility Regime Slippage & Position Multipliers:")
        for regime, stats in sorted(calib["regime_slippage"].items()):
            print(f"   {regime:<10}: slip={stats['ema_slip']:.4f}%  pos_mult={stats['pos_mult']:.3f}×  trades={stats['count']}")

    print(f"\n💡 Recommendations:")
    for r in calib.get("recommendations", []):
        print(f"   {r}")

    if args.verbose and calib.get("symbol_breakdown"):
        print(f"\n📊 Per-Symbol Breakdown:")
        for sym, stats in sorted(calib["symbol_breakdown"].items()):
            print(f"   {sym:<15} trades={stats['trades']:<4} wr={stats['win_rate']:.1%}  avg_pnl={stats['avg_pnl']:+.3f}%")

    if not args.dry_run:
        write_calibration(calib)
        print(f"\n✅ Wrote {CALIB_PATH}")
    else:
        print("\n[dry-run] Not writing file.")
        print(json.dumps(calib, indent=2))

    return calib


if __name__ == "__main__":
    main()
