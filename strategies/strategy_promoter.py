#!/usr/bin/env python3
"""
Strategy promotion and demotion gate logic.

Tiers:
  0 = library    (exists as a .py file, not yet backtested)
  1 = backtest   (passing backtest gates)
  2 = forward    (passing forward-testing gates)
  3 = live       (passing live promotion gates)

Promotion gates:
  0 -> 1: file exists, imports cleanly, generates >= 1 signal on synthetic data
  1 -> 2: bt_trades >= 50, bt_win_rate >= 40%, bt_sharpe >= 0.5,
           bt_max_dd <= 30%, bt_eep_score >= 50, profit_factor >= 1.1
  2 -> 3: ft_trades >= 100, ft_win_rate >= 45%, ft_sharpe >= 1.0,
           bt-to-live convergence < 20%, ft_eep_score >= 65, profit_factor >= 1.3,
           forward-testing duration >= 14 days
"""
import importlib
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Tier 0 -> 1
# ---------------------------------------------------------------------------

def check_tier0_to_1(strategy_name: str, file_path: Optional[str]) -> Tuple[bool, str]:
    """
    Gate: strategy file exists, imports cleanly, and can instantiate.

    Returns (passes, reason).
    """
    if not file_path:
        return False, "no file_path set in registry"

    import os
    if not os.path.exists(file_path):
        return False, f"file not found: {file_path}"

    # Try importing the module
    try:
        # Build module name from path (strategies.<name>)
        mod_name = f"strategies.{strategy_name}"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            mod = importlib.import_module(mod_name)
        # Check that a class exists with a detect() method
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (
                isinstance(obj, type)
                and hasattr(obj, "detect")
                and hasattr(obj, "name")
                and getattr(obj, "name", None) == strategy_name
            ):
                # Try instantiating
                instance = obj()
                return True, "imports cleanly and instantiates"
        return False, "no matching strategy class found in module"
    except Exception as exc:
        return False, f"import/instantiation error: {exc}"


# ---------------------------------------------------------------------------
# Tier 1 -> 2 (backtest gates)
# ---------------------------------------------------------------------------

def check_tier1_to_2(bt: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Gate: backtest metrics must pass all thresholds.

    bt dict keys: trades, win_rate (0-1), sharpe, max_dd_pct (positive = bad),
                  eep_score, profit_factor.

    Returns (passes, reason).
    """
    failures = []

    trades = bt.get("trades", 0) or 0
    win_rate = (bt.get("win_rate", 0) or 0)  # expected as 0-1 fraction
    sharpe = bt.get("sharpe", 0) or 0
    max_dd = bt.get("max_dd_pct", 100) or 100    # positive number, e.g. 15.0 = 15%
    eep_score = bt.get("eep_score", 0) or 0
    profit_factor = bt.get("profit_factor", 0) or 0

    if trades < 50:
        failures.append(f"bt_trades {trades} < 50")
    if win_rate < 0.40:
        failures.append(f"bt_win_rate {win_rate:.2%} < 40%")
    if sharpe < 0.5:
        failures.append(f"bt_sharpe {sharpe:.3f} < 0.5")
    if max_dd > 30.0:
        failures.append(f"bt_max_dd {max_dd:.1f}% > 30%")
    if eep_score < 50:
        failures.append(f"bt_eep_score {eep_score:.1f} < 50")
    if profit_factor < 1.1:
        failures.append(f"profit_factor {profit_factor:.3f} < 1.1")

    if failures:
        return False, "; ".join(failures)
    return True, "all backtest gates passed"


# ---------------------------------------------------------------------------
# Tier 2 -> 3 (forward-test / live promotion gates)
# ---------------------------------------------------------------------------

def check_tier2_to_3(
    ft: Dict[str, Any],
    ft_started: Optional[str] = None,
    bt_win_rate: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Gate: forward-test metrics must pass all thresholds.

    ft dict keys: trades, win_rate (0-1), sharpe, max_dd_pct, eep_score, profit_factor.
    ft_started: ISO string of when forward-testing started.
    bt_win_rate: backtest win rate (0-1) for convergence check.

    Returns (passes, reason).
    """
    failures = []

    trades = ft.get("trades", 0) or 0
    win_rate = ft.get("win_rate", 0) or 0
    sharpe = ft.get("sharpe", 0) or 0
    max_dd = ft.get("max_dd_pct", 100) or 100
    eep_score = ft.get("eep_score", 0) or 0
    profit_factor = ft.get("profit_factor", 0) or 0

    if trades < 100:
        failures.append(f"ft_trades {trades} < 100")
    if win_rate < 0.45:
        failures.append(f"ft_win_rate {win_rate:.2%} < 45%")
    if sharpe < 1.0:
        failures.append(f"ft_sharpe {sharpe:.3f} < 1.0")
    if max_dd > 20.0:
        failures.append(f"ft_max_dd {max_dd:.1f}% > 20%")
    if eep_score < 65:
        failures.append(f"ft_eep_score {eep_score:.1f} < 65")
    if profit_factor < 1.3:
        failures.append(f"profit_factor {profit_factor:.3f} < 1.3")

    # BT-to-live convergence: |ft_win_rate - bt_win_rate| < 20%
    if bt_win_rate is not None:
        convergence = abs(win_rate - bt_win_rate)
        if convergence >= 0.20:
            failures.append(
                f"bt-to-live convergence {convergence:.2%} >= 20% "
                f"(bt={bt_win_rate:.2%}, ft={win_rate:.2%})"
            )

    # 14+ days of forward testing
    if ft_started:
        try:
            started = datetime.fromisoformat(ft_started.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_running = (now - started).days
            if days_running < 14:
                failures.append(f"only {days_running} days of forward testing (need 14)")
        except Exception:
            failures.append("could not parse ft_started date")
    else:
        failures.append("ft_started not set")

    if failures:
        return False, "; ".join(failures)
    return True, "all live promotion gates passed"


# ---------------------------------------------------------------------------
# Demotion check (Tier 3 -> 2, or 2 -> archived)
# ---------------------------------------------------------------------------

def check_demotion(
    current_tier: int,
    ft: Dict[str, Any],
    consecutive_bad_days: int = 0,
) -> Tuple[bool, str]:
    """
    Check if a strategy should be demoted.

    Simple demotion triggers:
    - ft_win_rate drops below 30% with >= 20 trades
    - ft_sharpe drops below -0.5 with >= 20 trades
    - consecutive_bad_days >= 7 (caller provides this)

    Returns (should_demote, reason).
    """
    trades = ft.get("trades", 0) or 0
    win_rate = ft.get("win_rate", 0) or 0
    sharpe = ft.get("sharpe", 0) or 0

    if trades >= 20:
        if win_rate < 0.30:
            return True, f"ft_win_rate {win_rate:.2%} < 30% (demotion trigger)"
        if sharpe < -0.5:
            return True, f"ft_sharpe {sharpe:.3f} < -0.5 (demotion trigger)"

    if consecutive_bad_days >= 7:
        return True, f"{consecutive_bad_days} consecutive bad days (demotion trigger)"

    return False, "no demotion trigger"


# ---------------------------------------------------------------------------
# Unified promotion check (auto-detect direction from current tier)
# ---------------------------------------------------------------------------

def check_promotion(
    strategy_name: str,
    current_tier: int,
    file_path: Optional[str] = None,
    bt: Optional[Dict[str, Any]] = None,
    ft: Optional[Dict[str, Any]] = None,
    ft_started: Optional[str] = None,
    bt_win_rate: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Check whether a strategy qualifies for promotion to the next tier.

    Returns (qualifies, reason).
    """
    if current_tier == 0:
        return check_tier0_to_1(strategy_name, file_path)
    elif current_tier == 1:
        if bt is None:
            return False, "no backtest metrics provided"
        return check_tier1_to_2(bt)
    elif current_tier == 2:
        if ft is None:
            return False, "no forward-test metrics provided"
        return check_tier2_to_3(ft, ft_started=ft_started, bt_win_rate=bt_win_rate)
    else:
        return False, f"tier {current_tier} is the highest tier"
