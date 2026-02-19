#!/usr/bin/env python3
"""
eep_scorer.py
-------------
Entry + Exit Package (EEP) Scoring Engine — Phase 2

Computes the full 7-metric EEP score from actual trade statistics.
Used by optimizer, strategy pruner, and dashboard ranking.

Entry metrics (70% weight):
  30%  Profit Factor      (actual wins/losses gross)
  25%  Sharpe Ratio       (risk-adjusted return)
  20%  Max Drawdown       (lower is better)
  15%  Sortino Ratio      (downside-only volatility)
  10%  Expectancy         (avg $ per trade)

Exit quality metrics (30% weight):
  25%  MPC  — % of max theoretical profit captured
  20%  RR   — R:R realization rate (actual avg / expected TP)
  15%  SHF  — Stop Hit Frequency penalty
  10%  BEUR — Breakeven exit utilization rate
  +30% remainder from exit_score residual

Hard gates (strategy EXCLUDED if any fails):
  PF ≥ 1.3
  Sharpe ≥ 0.8
  MDD ≤ 35%
  Trades ≥ 30
  Positive expectancy

Usage:
    from eep_scorer import score_eep, passes_hard_gates, compute_eep_from_trades
"""

import math
from typing import Dict, Any, List, Tuple, Optional

# ─── Constants ──────────────────────────────────────────────────────────────

HARD_GATE_PF       = 1.3
HARD_GATE_SHARPE   = 0.8
HARD_GATE_MDD_MAX  = 35.0  # %
HARD_GATE_MIN_TRADES = 30

# EEP weight allocation
ENTRY_WEIGHT = 0.70
EXIT_WEIGHT  = 0.30

ENTRY_WEIGHTS = {
    "profit_factor": 0.30,
    "sharpe":        0.25,
    "mdd":           0.20,
    "sortino":       0.15,
    "expectancy":    0.10,
}

EXIT_WEIGHTS = {
    "mpc":  0.25,
    "rr":   0.20,
    "shf":  0.15,
    "beur": 0.10,
}


# ─── Core math helpers ───────────────────────────────────────────────────────

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _sortino(pnls: List[float], target: float = 0.0) -> float:
    """Sortino ratio: mean / downside-deviation."""
    if not pnls:
        return 0.0
    mean = sum(pnls) / len(pnls)
    downside_sq = [(p - target) ** 2 for p in pnls if p < target]
    if not downside_sq:
        return mean * 10.0  # no losses → very high sortino
    downside_dev = math.sqrt(sum(downside_sq) / len(downside_sq))
    return _safe_div(mean, downside_dev, 0.0)


def _sharpe(pnls: List[float]) -> float:
    """Sharpe ratio: mean / std_dev (risk-free = 0)."""
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std = math.sqrt(variance)
    return _safe_div(mean, std, 0.0)


def _profit_factor(pnls: List[float]) -> float:
    """PF = gross_profit / abs(gross_loss)."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def _max_drawdown(pnls: List[float]) -> float:
    """Max drawdown as % (positive value = bad)."""
    if not pnls:
        return 0.0
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
    return max_dd  # in same units as pnl_pct


def _expectancy(pnls: List[float]) -> float:
    """Expected value per trade."""
    if not pnls:
        return 0.0
    return sum(pnls) / len(pnls)


# ─── Full EEP from raw trade list ────────────────────────────────────────────

def compute_eep_from_trades(
    trades: List[Dict[str, Any]],
    tp_pct: float = 1.5,
    sl_pct: float = 1.0,
    label: str = "strategy",
) -> Dict[str, Any]:
    """
    Compute full EEP score from a list of closed paper/backtest trades.

    Each trade dict must have:
        pnl_pct   (float)
        reason    (str, optional: 'TP'/'SL'/'TIME'/'OTHER')
        entry_price, exit_price (optional, for MPC calculation)
        side      (optional: 'BUY'/'SELL')
    """
    if not trades:
        return _empty_eep(label, reason="no trades")

    pnls = [float(t.get("pnl_pct", 0)) for t in trades]
    n = len(pnls)

    # Exit classification
    tp_exits = sum(1 for t in trades if "TP" in str(t.get("reason", "")))
    sl_exits = sum(1 for t in trades if "SL" in str(t.get("reason", "")))
    time_exits = sum(1 for t in trades if "TIME" in str(t.get("reason", "")))

    # ── Compute all raw metrics ──────────────────────────────────────────────
    pf      = _profit_factor(pnls)
    sharpe  = _sharpe(pnls)
    sortino = _sortino(pnls)
    mdd     = _max_drawdown(pnls)
    exp     = _expectancy(pnls)

    # ── Exit quality metrics ──────────────────────────────────────────────────
    # MPC: % of maximum theoretical profit actually captured
    # Proxy: TP exits contribute full TP amount; others contribute their actual pnl
    if tp_pct > 0:
        max_possible_pnl = tp_pct * n
        actual_captured  = max(sum(pnls), 0)
        mpc = _clamp(_safe_div(actual_captured, max_possible_pnl))
    else:
        mpc = 0.5

    # RR realization: (avg win / tp_pct) * win_rate factor
    wins = [p for p in pnls if p > 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    rr_realisation = _clamp(_safe_div(avg_win, tp_pct)) if tp_pct > 0 else 0.5

    # SHF: Stop Hit Frequency — lower is better
    shf_raw = _safe_div(sl_exits, n)
    # Convert to score: SHF=0% → 1.0, SHF=50% → 0.0, SHF=100% → 0.0
    shf_score = _clamp(1.0 - shf_raw * 2)

    # BEUR: Breakeven / early exit utilization
    # Proxy: TIME exits that ended positive (breakeven usage)
    beur_raw = _safe_div(time_exits, n)
    time_positive = sum(1 for t in trades
                       if "TIME" in str(t.get("reason","")) and t.get("pnl_pct", 0) >= 0)
    if time_exits > 0:
        beur_score = _clamp(time_positive / time_exits)  # % of time exits that were profitable
    else:
        beur_score = 0.5  # no time exits → neutral

    return score_eep(
        pf=pf,
        sharpe=sharpe,
        mdd=mdd,
        sortino=sortino,
        expectancy=exp,
        mpc=mpc,
        rr_realisation=rr_realisation,
        shf_score=shf_score,
        beur_score=beur_score,
        num_trades=n,
        label=label,
    )


# ─── EEP from aggregate metrics (for optimizer integration) ──────────────────

def compute_eep_from_metrics(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute EEP from optimizer-style strategy dict.

    Accepts: win_rate, sharpe_ratio, total_pnl_pct, avg_pnl_pct,
             max_drawdown_pct, num_trades, risk_params{sl_pct, tp_pct, hold_periods}
    
    Falls back to proxy calculations where actuals are unavailable.
    """
    wr     = float(s.get("win_rate", 0))
    sharpe = float(s.get("sharpe_ratio", 0))
    pnl    = float(s.get("total_pnl_pct", 0))
    avg_pnl = float(s.get("avg_pnl_pct", 0))
    dd     = float(s.get("max_drawdown_pct", 0))
    trades = max(int(s.get("num_trades", 0)), 1)
    risk_p = s.get("risk_params", {}) or {}
    sl     = float(risk_p.get("sl_pct", 1.0))
    tp     = float(risk_p.get("tp_pct", 2.0))
    hold_p = float(risk_p.get("hold_periods", 12))

    lr = 1.0 - wr

    # Reconstruct PF from wr and TP/SL (proxy when actual trades unavailable)
    gross_profit = wr * tp * trades
    gross_loss   = lr * sl * trades
    pf = _safe_div(gross_profit, gross_loss, gross_profit) if gross_loss > 0 else gross_profit

    # Sortino proxy: use sharpe * 1.1 (downside vol slightly less than total vol)
    sortino = sharpe * 1.15

    # Expectancy from actual avg_pnl (per trade)
    exp = avg_pnl if avg_pnl != 0 else (pnl / trades if trades > 0 else 0.0)

    # Exit quality proxies
    max_possible_pnl = tp * trades
    actual_captured  = max(pnl, 0)
    mpc = _clamp(_safe_div(actual_captured, max_possible_pnl))

    rr_realisation = _clamp(_safe_div(avg_pnl, tp)) if tp > 0 else 0.5

    # SHF proxy: invert sl/tp ratio — tighter SL relative to TP = higher SHF risk
    shf_proxy = _clamp(_safe_div(sl, tp * 2))  # 0.5 → 50% sl/tp → SHF ~ moderate
    shf_score = _clamp(1.0 - shf_proxy)

    # BEUR proxy: hold_periods < 12 suggests earlier exit management
    beur_score = _clamp(_safe_div(max(0, 24 - hold_p), 24))

    return score_eep(
        pf=pf,
        sharpe=sharpe,
        mdd=dd,
        sortino=sortino,
        expectancy=exp,
        mpc=mpc,
        rr_realisation=rr_realisation,
        shf_score=shf_score,
        beur_score=beur_score,
        num_trades=int(s.get("num_trades", 0)),
        label=s.get("strategy", "unknown"),
    )


# ─── Core EEP scorer ─────────────────────────────────────────────────────────

def score_eep(
    pf: float,
    sharpe: float,
    mdd: float,
    sortino: float,
    expectancy: float,
    mpc: float,
    rr_realisation: float,
    shf_score: float,
    beur_score: float,
    num_trades: int = 0,
    label: str = "strategy",
) -> Dict[str, Any]:
    """
    Compute EEP from pre-calculated metrics. Returns full scoring breakdown.

    Returns dict with:
        eep_score      (0–100, primary ranking metric)
        entry_score    (0–100)
        exit_score     (0–100)
        gate_pass      (bool)
        gate_fails     (list[str])
        profit_factor, sharpe, sortino, mdd_pct, expectancy
        mpc, rr_realisation, shf_score, beur_score
    """
    # ── Entry scores (scaled to 0-1) ─────────────────────────────────────────
    # PF: 1.0 → 0.0, 1.3 → 0.15, 3.0 → 1.0
    pf_score = _clamp((pf - 1.0) / 2.0)

    # Sharpe: 0 → 0, 2 → 0.4, 5 → 1.0
    sharpe_score = _clamp(sharpe / 5.0)

    # MDD: 0% → 1.0, 35% → 0.0
    mdd_score = _clamp((HARD_GATE_MDD_MAX - mdd) / HARD_GATE_MDD_MAX)

    # Sortino: 0 → 0, 5 → 1.0
    sortino_score = _clamp(sortino / 5.0)

    # Expectancy: 0% → 0.0, 1% → 0.33, 3% → 1.0
    exp_score = _clamp(expectancy / 3.0)

    entry_score = (
        pf_score      * ENTRY_WEIGHTS["profit_factor"]
        + sharpe_score  * ENTRY_WEIGHTS["sharpe"]
        + mdd_score     * ENTRY_WEIGHTS["mdd"]
        + sortino_score * ENTRY_WEIGHTS["sortino"]
        + exp_score     * ENTRY_WEIGHTS["expectancy"]
    )

    # ── Exit scores (0-1 each) ────────────────────────────────────────────────
    # MPC already 0-1
    # RR already 0-1
    # SHF already 0-1 (inverted)
    # BEUR already 0-1

    exit_score_raw = (
        mpc             * EXIT_WEIGHTS["mpc"]
        + rr_realisation  * EXIT_WEIGHTS["rr"]
        + shf_score       * EXIT_WEIGHTS["shf"]
        + beur_score      * EXIT_WEIGHTS["beur"]
    )
    # Remaining weight (0.30 total - sum of defined = 0.30; all accounted)
    # Normalize exit score to full 0-1 range
    total_exit_weight = sum(EXIT_WEIGHTS.values())  # 0.70
    exit_score = exit_score_raw / total_exit_weight if total_exit_weight > 0 else exit_score_raw

    # ── Combined EEP ─────────────────────────────────────────────────────────
    eep_raw = entry_score * ENTRY_WEIGHT + exit_score * EXIT_WEIGHT
    eep_score = round(_clamp(eep_raw) * 100, 2)

    # ── Hard gates ───────────────────────────────────────────────────────────
    gate_pass, gate_fails = check_hard_gates(pf, sharpe, mdd, num_trades, expectancy)

    return {
        "eep_score":      eep_score,
        "entry_score":    round(entry_score * 100, 2),
        "exit_score":     round(exit_score * 100, 2),
        "profit_factor":  round(pf, 3),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "mdd_pct":        round(mdd, 3),
        "expectancy":     round(expectancy, 4),
        "mpc":            round(mpc, 3),
        "rr_realisation": round(rr_realisation, 3),
        "shf_score":      round(shf_score, 3),
        "beur_score":     round(beur_score, 3),
        "num_trades":     num_trades,
        "gate_pass":      gate_pass,
        "gate_fails":     gate_fails,
        "label":          label,
    }


# ─── Hard gate check ─────────────────────────────────────────────────────────

def check_hard_gates(
    pf: float,
    sharpe: float,
    mdd: float,
    num_trades: int,
    expectancy: float,
) -> Tuple[bool, List[str]]:
    """
    Returns (pass: bool, fails: list[str]).
    All five conditions must be met for a strategy to pass.

    P2 FIX: Sharpe gate is only enforced when num_trades >= HARD_GATE_MIN_TRADES.
    With fewer than 30 trades, the Sharpe ratio is statistically unreliable
    (high variance from small sample), so we skip the Sharpe gate and instead
    fail on the trade-count gate. This prevents false passes from lucky streaks
    and false fails from high-variance small samples.
    """
    fails = []
    if pf < HARD_GATE_PF:
        fails.append(f"PF={pf:.2f} < {HARD_GATE_PF}")
    # P2: Sharpe gate requires >= 30 trades for statistical reliability
    if num_trades >= HARD_GATE_MIN_TRADES:
        if sharpe < HARD_GATE_SHARPE:
            fails.append(f"Sharpe={sharpe:.2f} < {HARD_GATE_SHARPE}")
    else:
        # Not enough trades to evaluate Sharpe — fail on trade count gate below
        pass
    if mdd > HARD_GATE_MDD_MAX:
        fails.append(f"MDD={mdd:.1f}% > {HARD_GATE_MDD_MAX}%")
    if num_trades < HARD_GATE_MIN_TRADES:
        fails.append(f"Trades={num_trades} < {HARD_GATE_MIN_TRADES} (need {HARD_GATE_MIN_TRADES} for Sharpe gate)")
    if expectancy <= 0:
        fails.append(f"Expectancy={expectancy:.4f} ≤ 0")
    return (len(fails) == 0, fails)


# ─── Backward-compat shim (for sync_optimizer_to_db.py) ─────────────────────

def score_strategy_eep(s: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for existing callers. Delegates to compute_eep_from_metrics."""
    result = compute_eep_from_metrics(s)
    # Map keys to legacy schema used by sync_optimizer_to_db
    return {
        "eep_score":      result["eep_score"],
        "entry_score":    result["entry_score"],
        "exit_score":     result["exit_score"],
        "profit_factor":  result["profit_factor"],
        "expectancy":     result["expectancy"],
        "mpc":            result["mpc"],
        "rr_realisation": result["rr_realisation"],
        "shf_score":      result["shf_score"],
        "beur_score":     result["beur_score"],
        "gate_pass":      result["gate_pass"],
        "gate_fails":     result["gate_fails"],
    }


def passes_hard_gates(s: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Backward-compat: accepts strategy dict, returns (pass, fails)."""
    result = compute_eep_from_metrics(s)
    return result["gate_pass"], result["gate_fails"]


# ─── Rank a list of strategies by EEP ────────────────────────────────────────

def rank_by_eep(strategies: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Rank a list of strategy dicts by EEP score.
    Attaches eep_score, eep_rank, gate_pass to each.
    Strategies that fail hard gates are ranked AFTER those that pass.
    """
    scored = []
    for s in strategies:
        eep = compute_eep_from_metrics(s)
        s_copy = dict(s)
        s_copy["eep_score"]  = eep["eep_score"]
        s_copy["eep_rank"]   = None  # filled below
        s_copy["gate_pass"]  = eep["gate_pass"]
        s_copy["gate_fails"] = eep["gate_fails"]
        s_copy["eep_detail"] = eep
        scored.append(s_copy)

    # Sort: gate-passing first, then by EEP descending
    scored.sort(key=lambda x: (0 if x["gate_pass"] else 1, -x["eep_score"]))

    # Assign ranks
    for i, s in enumerate(scored, 1):
        s["eep_rank"] = i

    return scored[:top_n] if top_n else scored


# ─── CLI self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Simulate a strategy that passes all gates
    good_strategy = {
        "strategy": "ema_crossover",
        "symbol": "BTC-USDT",
        "win_rate": 0.58,
        "sharpe_ratio": 1.4,
        "total_pnl_pct": 22.5,
        "avg_pnl_pct": 0.45,
        "max_drawdown_pct": 18.0,
        "num_trades": 50,
        "risk_params": {"sl_pct": 1.0, "tp_pct": 2.5, "hold_periods": 8},
    }

    # Simulate a weak strategy that fails gates
    bad_strategy = {
        "strategy": "momentum_v1",
        "symbol": "DOGE-USDT",
        "win_rate": 0.45,
        "sharpe_ratio": 0.3,
        "total_pnl_pct": -3.0,
        "avg_pnl_pct": -0.06,
        "max_drawdown_pct": 40.0,
        "num_trades": 20,
        "risk_params": {"sl_pct": 2.0, "tp_pct": 2.0, "hold_periods": 12},
    }

    print("=" * 60)
    print("EEP SCORER — Self Test")
    print("=" * 60)

    for strat in [good_strategy, bad_strategy]:
        result = compute_eep_from_metrics(strat)
        print(f"\n{strat['strategy']} ({strat['symbol']})")
        print(f"  EEP Score:    {result['eep_score']}")
        print(f"  Entry Score:  {result['entry_score']}")
        print(f"  Exit Score:   {result['exit_score']}")
        print(f"  Gate Pass:    {result['gate_pass']}")
        if not result["gate_pass"]:
            print(f"  Gate Fails:   {result['gate_fails']}")

    # Test rank
    ranked = rank_by_eep([good_strategy, bad_strategy])
    print(f"\nRanking:")
    for r in ranked:
        print(f"  #{r['eep_rank']} {r['strategy']} — EEP={r['eep_score']} gate={'✅' if r['gate_pass'] else '⛔'}")
