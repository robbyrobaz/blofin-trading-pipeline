#!/usr/bin/env python3
"""
run_pipeline.py — Strategy Lifecycle Pipeline Runner
====================================================

Runs hourly. Feeds strategies through the 3-tier lifecycle:
  Tier 0 (Library) → Tier 1 (Backtest) → Tier 2 (Forward Test)

Phases:
  1. Discover & Register: scan strategies/ for new .py files → register at Tier 0
  2. Backtest Tier 0: run BacktestEngine, update bt_* columns, promote if ≥1 signal
  3. Evaluate Tier 1: check backtest gates, promote to Tier 2 if passing
  4. Monitor Tier 2: check forward-test metrics, demote if degraded
  5. EEP Refresh: refresh all EEP scores (keeps dashboard current)
  6. Design New: every 6th run (~6h), call StrategyDesigner if Tier 0 < 60

Usage:
    python3 orchestration/run_pipeline.py
    python3 orchestration/run_pipeline.py --dry-run
"""

import argparse
import importlib.util
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Constants ────────────────────────────────────────────────────────────────
DB_PATH      = REPO_ROOT / "data" / "blofin_monitor.db"
STRATEGIES_DIR = REPO_ROOT / "strategies"
LOG_PATH     = REPO_ROOT / "data" / "pipeline.log"
STATE_FILE   = REPO_ROOT / "data" / "pipeline_state.json"

# Tier 0 gets quick screening (2 symbols), Tier 1+ gets full validation (5 symbols)
SCREENING_SYMBOLS = ["BTC-USDT", "ETH-USDT"]
FULL_SYMBOLS      = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "XRP-USDT"]
BACKTEST_SYMBOLS  = SCREENING_SYMBOLS  # default for phase2 (Tier 0 screening)
BACKTEST_DAYS     = 7
BACKTEST_TIMEFRAME = "1m"   # 1m candles closer to live tick behavior (strategies were tuned for tick data)

# Tier 1 promotion gates (must ALL pass)
GATE_MIN_TRADES   = 50
GATE_MIN_WIN_RATE = 0.40   # 40%
GATE_MIN_SHARPE   = 0.5
GATE_MIN_EEP      = 50.0

# Tier 2 demotion trigger (strategy degraded badly)
DEMOTE_MAX_EEP    = 10.0
DEMOTE_MIN_TRADES = 100

# Design phase cadence: run every Nth pipeline execution
DESIGN_EVERY_N_RUNS  = 6
TIER0_DESIGN_CAP     = 60   # Stop designing if Tier 0 already has this many


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("pipeline")


log = setup_logging()


# ── DB helpers ───────────────────────────────────────────────────────────────

def db_connect(timeout: int = 60) -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def db_execute_with_retry(
    sql: str,
    params: tuple = (),
    max_retries: int = 3,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Execute a write statement with retry on DB lock."""
    own_conn = conn is None
    for attempt in range(max_retries):
        try:
            if own_conn:
                conn = db_connect()
            conn.execute(sql, params)
            if own_conn:
                conn.commit()
                conn.close()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning("DB locked (attempt %d/%d), retrying in %ds…", attempt + 1, max_retries, wait)
                if own_conn and conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(wait)
            else:
                raise


def db_batch_write(statements: List[Tuple[str, tuple]], max_retries: int = 3) -> int:
    """Execute multiple write statements in a single transaction with retry."""
    for attempt in range(max_retries):
        try:
            conn = db_connect()
            cur = conn.cursor()
            count = 0
            for sql, params in statements:
                cur.execute(sql, params)
                count += cur.rowcount
            conn.commit()
            conn.close()
            return count
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning("DB locked (batch, attempt %d/%d), retrying in %ds…", attempt + 1, max_retries, wait)
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(wait)
            else:
                raise
    return 0


# ── Pipeline state (run counter for design phase) ────────────────────────────

def load_pipeline_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"run_count": 0, "last_run": None}


def save_pipeline_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


# ── Strategy loading ─────────────────────────────────────────────────────────

class StrategyAdapter:
    """
    Adapter that wraps any strategy class to work with BacktestEngine.

    BacktestEngine calls: strategy.detect(context_candles, symbol)
    where context_candles is a list of OHLCV dicts.

    This adapter handles:
      - New-style classes with detect(candles, symbol) → pass-through
      - Old BaseStrategy classes with detect(symbol, price, ...) → adapted
      - AI-generated classes with analyze(candles, indicators) → wrapped
    """

    def __init__(self, instance: Any, strategy_name: str, file_path: str = ""):
        self._inst    = instance
        self.name     = strategy_name
        self.file_path = file_path

    def detect(self, context_candles: List[Dict], symbol: str) -> Optional[Dict]:
        """
        Try to call the underlying strategy and return a dict signal or None.
        """
        inst = self._inst

        # --- Try new-style detect(candles, symbol) first ---
        if hasattr(inst, "detect"):
            try:
                result = inst.detect(context_candles, symbol)
                if result is not None:
                    # Normalise: could be a Signal namedtuple or dict
                    if hasattr(result, "signal"):
                        return {"signal": result.signal}
                    if isinstance(result, dict):
                        return result
                    return {"signal": str(result)}
            except TypeError:
                # Wrong signature — try old-style detect(symbol, price, volume, ts_ms, prices, volumes)
                # where prices/volumes are List[Tuple[int, float]] (ts_ms, value) pairs
                if context_candles:
                    last = context_candles[-1]
                    prices_tuples = [(c["ts_ms"], c["close"]) for c in context_candles]
                    volumes_tuples = [(c["ts_ms"], c.get("volume", 0)) for c in context_candles]
                    try:
                        result = inst.detect(
                            symbol,
                            last["close"],
                            last.get("volume", 0),
                            last["ts_ms"],
                            prices_tuples,
                            volumes_tuples,
                        )
                        if result is not None:
                            if hasattr(result, "signal"):
                                return {"signal": result.signal}
                            if isinstance(result, dict):
                                return result
                            return {"signal": str(result)}
                    except Exception:
                        return None
            except Exception:
                return None

        # --- Try analyze(candles, indicators) (AI-generated) ---
        if hasattr(inst, "analyze"):
            try:
                result = inst.analyze(context_candles, {})
                if result and str(result).upper() in ("BUY", "SELL"):
                    return {"signal": str(result).upper()}
            except Exception:
                return None

        return None


def _load_strategy_class_from_module(mod: Any, file_path: Path) -> Optional[StrategyAdapter]:
    """Extract the first valid strategy class from a loaded module."""
    skip_bases = {"BaseStrategy"}

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if attr is None or not isinstance(attr, type):
            continue
        if attr.__name__ in skip_bases:
            continue

        has_detect  = hasattr(attr, "detect")
        has_analyze = hasattr(attr, "analyze")
        has_name    = hasattr(attr, "name") or hasattr(attr, "__name__")

        if not (has_detect or has_analyze):
            continue

        try:
            inst = attr()
        except Exception:
            continue

        # Determine strategy name
        strat_name = None
        if hasattr(inst, "name") and inst.name and inst.name != "base":
            strat_name = str(inst.name)
        elif hasattr(attr, "name") and isinstance(getattr(attr, "name", None), str):
            n = attr.name
            if n and n != "base":
                strat_name = n
        if not strat_name:
            strat_name = file_path.stem

        return StrategyAdapter(inst, strat_name, file_path=str(file_path))

    return None


def load_strategy_from_file(file_path: Path) -> Optional[StrategyAdapter]:
    """
    Load a strategy .py file and return a StrategyAdapter, or None on failure.

    Strategy 1: import via the `strategies` package (fixes relative imports).
    Strategy 2: standalone importlib load (for self-contained files).
    Skips infrastructure files.
    """
    SKIP_STEMS = {"__init__", "base_strategy", "strategy_promoter"}
    if file_path.stem in SKIP_STEMS:
        return None

    # --- Strategy 1: package import (handles relative imports) ---
    try:
        mod_name = f"strategies.{file_path.stem}"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            import importlib as _il
            mod = _il.import_module(mod_name)
        result = _load_strategy_class_from_module(mod, file_path)
        if result is not None:
            return result
    except Exception:
        pass  # Fall through to standalone load

    # --- Strategy 2: standalone importlib (for AI-generated self-contained files) ---
    try:
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = _load_strategy_class_from_module(mod, file_path)
        if result is not None:
            return result
    except Exception as exc:
        log.warning("Failed to load %s: %s", file_path.name, exc)

    return None


def load_all_strategies() -> Dict[str, StrategyAdapter]:
    """
    Load all strategy .py files from STRATEGIES_DIR.
    Returns dict: strategy_name → StrategyAdapter (keyed by strategy name).
    Also builds a file_path → StrategyAdapter index accessible as adapters.
    """
    adapters: Dict[str, StrategyAdapter] = {}
    for py_file in sorted(STRATEGIES_DIR.glob("*.py")):
        adapter = load_strategy_from_file(py_file)
        if adapter and adapter.name not in adapters:
            adapters[adapter.name] = adapter
    return adapters


# ── Registry helpers ─────────────────────────────────────────────────────────

def get_registry(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """Return all strategy_registry rows keyed by strategy_name."""
    cur = conn.execute("SELECT * FROM strategy_registry")
    cols = [d[0] for d in cur.description]
    return {
        row["strategy_name"]: dict(zip(cols, tuple(row)))
        for row in cur.fetchall()
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Phase 1: Discover & Register ────────────────────────────────────────────

def phase1_discover_register(
    adapters: Dict[str, StrategyAdapter],
    dry_run: bool,
) -> List[str]:
    """
    Scan strategies/ dir, register any new strategies at Tier 0.
    Returns list of newly registered strategy names.
    """
    log.info("─── Phase 1: Discover & Register ───")
    new_strategies: List[str] = []

    conn = db_connect()
    registry = get_registry(conn)
    conn.close()

    registered_names = set(registry.keys())
    statements = []
    ts = now_iso()

    for name, adapter in adapters.items():
        if name in registered_names:
            continue

        # Prefer the file_path tracked during loading
        file_path = adapter.file_path or None
        if not file_path:
            # Fallback: guess from name
            candidate = STRATEGIES_DIR / f"{name}.py"
            if candidate.exists():
                file_path = str(candidate)

        log.info("  [NEW] %s (file: %s)", name, file_path or "unknown")

        if not dry_run:
            statements.append((
                """
                INSERT OR IGNORE INTO strategy_registry
                    (strategy_name, tier, source, file_path, created_at, updated_at)
                VALUES (?, 0, 'auto_discovery', ?, ?, ?)
                """,
                (name, file_path, ts, ts),
            ))
        new_strategies.append(name)

    if statements:
        written = db_batch_write(statements)
        log.info("  Registered %d new strategies (%d DB rows written)", len(new_strategies), written)
    elif new_strategies:
        log.info("  [DRY-RUN] Would register %d strategies", len(new_strategies))
    else:
        log.info("  No new strategies to register.")

    return new_strategies


# ── Backtest helper ───────────────────────────────────────────────────────────

def run_backtest_for_strategy(
    adapter: StrategyAdapter,
    symbols: List[str] = BACKTEST_SYMBOLS,
    days_back: int = BACKTEST_DAYS,
    limit_rows: int = None,
) -> Dict[str, Any]:
    """
    Run BacktestEngine on the given strategy across all symbols.
    Returns aggregated metrics dict.

    Args:
        limit_rows: If set, cap tick rows loaded per symbol (e.g. 5000 for smoke test)

    Returned keys:
        total_trades, win_rate, sharpe, pnl_pct, max_dd, eep_score,
        all_trades (list), symbol_results (list)
    """
    from backtester.backtest_engine import BacktestEngine
    from eep_scorer import compute_eep_from_trades

    all_trades = []
    symbol_results = []
    symbols_run = 0

    for symbol in symbols:
        try:
            engine = BacktestEngine(
                symbol=symbol,
                days_back=days_back,
                db_path=str(DB_PATH),
                initial_capital=10_000.0,
                limit_rows=limit_rows,
            )
            if not engine.ticks:
                log.debug("  No ticks for %s, skipping", symbol)
                continue

            result = engine.run_strategy(
                adapter,
                timeframe=BACKTEST_TIMEFRAME,
                stop_loss_pct=3.0,
                take_profit_pct=5.0,
            )
            trades = result.get("trades", [])
            metrics = result.get("metrics", {})
            all_trades.extend(trades)
            symbol_results.append({
                "symbol": symbol,
                "trades": len(trades),
                "metrics": metrics,
            })
            symbols_run += 1
            log.debug("  %s/%s: %d trades", adapter.name, symbol, len(trades))

        except Exception as exc:
            log.warning("  Backtest error %s/%s: %s", adapter.name, symbol, exc)

    if not all_trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "pnl_pct": 0.0,
            "max_dd": 0.0,
            "eep_score": 0.0,
            "all_trades": [],
            "symbol_results": symbol_results,
        }

    # Aggregate metrics from trades
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t.get("pnl_pct", 0) > 0)
    win_rate = wins / n if n > 0 else 0.0
    pnl_pct = sum(t.get("pnl_pct", 0) for t in all_trades)
    avg_pnl = pnl_pct / n

    # Sharpe (simple, not annualised)
    pnls = [t.get("pnl_pct", 0) for t in all_trades]
    if n >= 2:
        mean_p = sum(pnls) / n
        variance = sum((p - mean_p) ** 2 for p in pnls) / n
        import math
        std = math.sqrt(variance)
        sharpe = mean_p / std if std > 0 else 0.0
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

    # EEP score
    eep_result = compute_eep_from_trades(all_trades, label=adapter.name)
    eep_score = eep_result.get("eep_score", 0.0)

    return {
        "total_trades": n,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "pnl_pct": pnl_pct,
        "max_dd": max_dd,
        "eep_score": eep_score,
        "all_trades": all_trades,
        "symbol_results": symbol_results,
    }


# ── Phase 2: Backtest Tier 0 → Promote to Tier 1 ───────────────────────────

def phase2_backtest_tier0(
    adapters: Dict[str, StrategyAdapter],
    registry: Dict[str, Dict],
    dry_run: bool,
) -> Dict[str, int]:
    """
    Backtest all Tier 0 strategies.
    Promote to Tier 1 if ≥1 signal generated.
    Returns counts: {promoted, failed, skipped}
    """
    log.info("─── Phase 2: Backtest Tier 0 → Tier 1 ───")
    tier0 = [r for r in registry.values() if r["tier"] == 0 and not r.get("archived")]
    log.info("  %d Tier 0 strategies to backtest", len(tier0))

    counts = {"promoted": 0, "failed": 0, "skipped": 0}
    statements = []
    ts = now_iso()

    for row in tier0:
        name = row["strategy_name"]
        adapter = adapters.get(name)

        if adapter is None:
            log.warning("  [SKIP] %s — no .py file/class found", name)
            counts["skipped"] += 1
            continue

        # Stage 1: Smoke test with 50000 most recent rows (fast screening)
        log.info("  Smoke-testing %s (50000 rows) …", name)
        try:
            smoke = run_backtest_for_strategy(adapter, symbols=["BTC-USDT"], limit_rows=50000)
        except Exception as exc:
            log.error("  [ERROR] %s smoke test crashed: %s", name, exc)
            counts["failed"] += 1
            continue

        if smoke["total_trades"] == 0:
            log.info("  %s: 0 trades in smoke test → skip full backtest, stays tier 0", name)
            counts["failed"] += 1
            continue

        # Stage 2: Full backtest (passed smoke test — strategy generates signals)
        # Use 500K most-recent ticks (~50K 5m candles ≈ 174 days) for Tier 0 screening.
        # This is far faster than loading all 36M+ raw ticks from days_back=7.
        log.info("  %s passed smoke (%d trades) → full backtest (limit=500000 ticks) …", name, smoke["total_trades"])
        try:
            bt = run_backtest_for_strategy(adapter, limit_rows=500000)
        except Exception as exc:
            log.error("  [ERROR] %s full backtest crashed: %s", name, exc)
            counts["failed"] += 1
            continue

        total_trades = bt["total_trades"]
        promote = total_trades >= 1
        new_tier = 1 if promote else 0

        log.info(
            "  %s: %d trades, WR=%.1f%%, Sharpe=%.3f, EEP=%.1f → tier %d",
            name, total_trades,
            bt["win_rate"] * 100,
            bt["sharpe"],
            bt["eep_score"],
            new_tier,
        )

        if dry_run:
            if promote:
                log.info("    [DRY-RUN] Would promote %s to Tier 1", name)
                counts["promoted"] += 1
            continue

        # Update bt_* columns and tier
        promoted_at = ts if promote else None
        statements.append((
            """
            UPDATE strategy_registry SET
                bt_win_rate  = ?,
                bt_sharpe    = ?,
                bt_pnl_pct   = ?,
                bt_max_dd    = ?,
                bt_trades    = ?,
                bt_eep_score = ?,
                bt_last_run  = ?,
                tier         = ?,
                promoted_at  = CASE WHEN ? IS NOT NULL THEN ? ELSE promoted_at END,
                updated_at   = ?
            WHERE strategy_name = ?
            """,
            (
                round(bt["win_rate"], 6),
                round(bt["sharpe"], 6),
                round(bt["pnl_pct"], 6),
                round(bt["max_dd"], 6),
                total_trades,
                round(bt["eep_score"], 4),
                ts,
                new_tier,
                promoted_at, promoted_at,
                ts,
                name,
            ),
        ))

        if promote:
            counts["promoted"] += 1
        else:
            counts["failed"] += 1

    if not dry_run and statements:
        written = db_batch_write(statements)
        log.info("  Phase 2 complete: %d DB rows updated", written)

    log.info(
        "  Result: %d promoted to Tier 1, %d stayed at Tier 0, %d skipped",
        counts["promoted"], counts["failed"], counts["skipped"],
    )
    return counts


# ── Phase 2.5: Re-backtest Tier 1 strategies ────────────────────────────────

# How many hours before a T1 strategy is considered stale and needs re-backtest
T1_REBACKTEST_HOURS = 24
# Trades threshold above which data is presumed corrupt (pre-fix era)
T1_CORRUPT_TRADES_THRESHOLD = 10_000

def phase2b_rebacktest_tier1(
    adapters: Dict[str, StrategyAdapter],
    registry: Dict[str, Dict],
    dry_run: bool,
) -> Dict[str, int]:
    """
    Re-backtest all Tier 1 strategies with FULL_SYMBOLS so Phase 3 has fresh
    metrics to evaluate against promotion gates.

    A T1 strategy is eligible for re-backtest if:
      - Its bt_trades is absurdly large (>10K — indicates pre-BUY/SELL-fix corrupt data)
      - OR bt_last_run is more than T1_REBACKTEST_HOURS hours ago
      - OR it has never been backtested as T1 (bt_last_run < promoted_at)

    Returns counts: {backtested, skipped, failed}
    """
    import math
    from datetime import timedelta

    log.info("─── Phase 2.5: Re-backtest Tier 1 (with FULL_SYMBOLS) ───")
    tier1 = [r for r in registry.values() if r["tier"] == 1 and not r.get("archived")]
    log.info("  %d Tier 1 strategies to evaluate for re-backtest", len(tier1))

    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(hours=T1_REBACKTEST_HOURS)

    counts = {"backtested": 0, "skipped": 0, "failed": 0}
    statements = []
    ts = now_iso()

    for row in tier1:
        name = row["strategy_name"]
        bt_trades   = row.get("bt_trades") or 0
        bt_last_run = row.get("bt_last_run") or ""
        promoted_at = row.get("promoted_at") or ""

        # Determine if re-backtest is needed
        corrupt_data = bt_trades > T1_CORRUPT_TRADES_THRESHOLD
        stale = False
        if bt_last_run:
            try:
                last_run_dt = datetime.fromisoformat(bt_last_run.replace("Z", "+00:00"))
                stale = last_run_dt < stale_cutoff
            except Exception:
                stale = True
        else:
            stale = True

        # Also re-test if bt_last_run predates promoted_at (never tested as T1)
        never_tested_as_t1 = False
        if promoted_at and bt_last_run:
            try:
                promoted_dt = datetime.fromisoformat(promoted_at.replace("Z", "+00:00"))
                last_run_dt = datetime.fromisoformat(bt_last_run.replace("Z", "+00:00"))
                never_tested_as_t1 = last_run_dt < promoted_dt
            except Exception:
                never_tested_as_t1 = True

        needs_retest = corrupt_data or stale or never_tested_as_t1
        if not needs_retest:
            log.info("  [SKIP] %s — bt_last_run recent, data clean (%d trades)", name, bt_trades)
            counts["skipped"] += 1
            continue

        reason = []
        if corrupt_data:
            reason.append(f"corrupt_data({bt_trades:,} trades)")
        if stale:
            reason.append(f"stale(last_run={bt_last_run[:19]})")
        if never_tested_as_t1:
            reason.append("never_tested_as_t1")
        log.info("  Re-backtesting %s [%s] with FULL_SYMBOLS …", name, ", ".join(reason))

        adapter = adapters.get(name)
        if adapter is None:
            log.warning("  [SKIP] %s — no adapter found", name)
            counts["skipped"] += 1
            continue

        if dry_run:
            log.info("    [DRY-RUN] Would re-backtest %s with FULL_SYMBOLS", name)
            counts["backtested"] += 1
            continue

        try:
            bt = run_backtest_for_strategy(adapter, symbols=FULL_SYMBOLS, limit_rows=500_000)
        except Exception as exc:
            log.error("  [ERROR] %s re-backtest crashed: %s", name, exc)
            counts["failed"] += 1
            continue

        total_trades = bt["total_trades"]
        log.info(
            "  %s: %d trades, WR=%.1f%%, Sharpe=%.3f, MDD=%.1f%%, EEP=%.1f",
            name, total_trades,
            bt["win_rate"] * 100,
            bt["sharpe"],
            bt["max_dd"],
            bt["eep_score"],
        )

        statements.append((
            """
            UPDATE strategy_registry SET
                bt_win_rate  = ?,
                bt_sharpe    = ?,
                bt_pnl_pct   = ?,
                bt_max_dd    = ?,
                bt_trades    = ?,
                bt_eep_score = ?,
                bt_last_run  = ?,
                updated_at   = ?
            WHERE strategy_name = ?
            """,
            (
                round(bt["win_rate"], 6),
                round(bt["sharpe"], 6),
                round(bt["pnl_pct"], 6),
                round(bt["max_dd"], 6),
                total_trades,
                round(bt["eep_score"], 4),
                ts,
                ts,
                name,
            ),
        ))
        counts["backtested"] += 1

    if not dry_run and statements:
        written = db_batch_write(statements)
        log.info("  Phase 2.5 complete: %d T1 strategies re-backtested, %d DB rows updated",
                 counts["backtested"], written)
    else:
        log.info("  Phase 2.5 complete: backtested=%d, skipped=%d, failed=%d",
                 counts["backtested"], counts["skipped"], counts["failed"])

    return counts


# ── Phase 3: Evaluate Tier 1 → Promote to Tier 2 ───────────────────────────

def phase3_evaluate_tier1(
    registry: Dict[str, Dict],
    dry_run: bool,
) -> Dict[str, int]:
    """
    Evaluate all Tier 1 strategies against promotion gates.
    Promote to Tier 2 if ALL gates pass.
    Returns counts: {promoted, stayed}
    """
    log.info("─── Phase 3: Evaluate Tier 1 → Tier 2 ───")
    tier1 = [r for r in registry.values() if r["tier"] == 1 and not r.get("archived")]
    log.info("  %d Tier 1 strategies to evaluate", len(tier1))

    counts = {"promoted": 0, "stayed": 0}
    statements = []
    ts = now_iso()

    for row in tier1:
        name = row["strategy_name"]
        bt_trades   = row.get("bt_trades") or 0
        bt_win_rate = row.get("bt_win_rate") or 0.0
        bt_sharpe   = row.get("bt_sharpe") or 0.0
        bt_eep      = row.get("bt_eep_score") or 0.0

        gates_pass = (
            bt_trades   >= GATE_MIN_TRADES
            and bt_win_rate >= GATE_MIN_WIN_RATE
            and bt_sharpe   >= GATE_MIN_SHARPE
            and bt_eep      >= GATE_MIN_EEP
        )

        failures = []
        if bt_trades   < GATE_MIN_TRADES:
            failures.append(f"trades={bt_trades}<{GATE_MIN_TRADES}")
        if bt_win_rate < GATE_MIN_WIN_RATE:
            failures.append(f"win_rate={bt_win_rate:.2%}<{GATE_MIN_WIN_RATE:.0%}")
        if bt_sharpe   < GATE_MIN_SHARPE:
            failures.append(f"sharpe={bt_sharpe:.3f}<{GATE_MIN_SHARPE}")
        if bt_eep      < GATE_MIN_EEP:
            failures.append(f"eep={bt_eep:.1f}<{GATE_MIN_EEP}")

        if gates_pass:
            log.info(
                "  ✅ %s — PROMOTE to Tier 2 (trades=%d, WR=%.1f%%, sharpe=%.3f, EEP=%.1f)",
                name, bt_trades, bt_win_rate * 100, bt_sharpe, bt_eep,
            )
            counts["promoted"] += 1

            if not dry_run:
                statements.append((
                    """
                    UPDATE strategy_registry SET
                        tier        = 2,
                        promoted_at = ?,
                        ft_started  = ?,
                        updated_at  = ?
                    WHERE strategy_name = ?
                    """,
                    (ts, ts, ts, name),
                ))
        else:
            log.info("  ⏳ %s — stay at Tier 1 [%s]", name, "; ".join(failures))
            counts["stayed"] += 1

    if not dry_run and statements:
        written = db_batch_write(statements)
        log.info("  Phase 3 complete: %d promoted, %d stayed", counts["promoted"], counts["stayed"])

    return counts


# ── Phase 4: Monitor Tier 2 (Forward Test) ──────────────────────────────────

def _get_ft_metrics_from_paper_trades(conn: sqlite3.Connection, strategy_name: str) -> Dict[str, Any]:
    """
    Pull closed paper trades for a strategy and compute ft_* metrics.
    Returns dict with win_rate, sharpe, pnl_pct, max_dd, trades, eep_score.
    """
    cur = conn.execute(
        """
        SELECT pt.pnl_pct, pt.reason
        FROM paper_trades pt
        JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
        JOIN signals s ON cs.signal_id = s.id
        WHERE pt.status = 'CLOSED' AND s.strategy = ?
        ORDER BY pt.opened_ts_ms
        """,
        (strategy_name,),
    )
    rows = cur.fetchall()
    if not rows:
        return {}

    from eep_scorer import compute_eep_from_trades

    trades = []
    for r in rows:
        exit_reason = r[1] or ""
        if "EXIT:" in exit_reason:
            exit_reason = exit_reason.split("EXIT:")[-1].strip()
        trades.append({"pnl_pct": r[0] or 0.0, "reason": exit_reason})

    n = len(trades)
    pnls = [t["pnl_pct"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n
    total_pnl = sum(pnls)

    if n >= 2:
        import math
        mean_p = total_pnl / n
        variance = sum((p - mean_p) ** 2 for p in pnls) / n
        std = math.sqrt(variance)
        sharpe = mean_p / std if std > 0 else 0.0
    else:
        sharpe = 0.0

    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    eep_result = compute_eep_from_trades(trades, label=strategy_name)
    eep_score = eep_result.get("eep_score", 0.0)

    return {
        "win_rate": win_rate,
        "sharpe": sharpe,
        "pnl_pct": total_pnl,
        "max_dd": max_dd,
        "trades": n,
        "eep_score": eep_score,
    }


def phase4_monitor_tier2(
    registry: Dict[str, Dict],
    dry_run: bool,
) -> Dict[str, int]:
    """
    Update ft_* metrics for all Tier 2 strategies.
    Demote to Tier 1 if EEP < DEMOTE_MAX_EEP AND trades >= DEMOTE_MIN_TRADES.
    Returns counts: {updated, demoted}
    """
    log.info("─── Phase 4: Monitor Tier 2 ───")
    tier2 = [r for r in registry.values() if r["tier"] == 2 and not r.get("archived")]
    log.info("  %d Tier 2 strategies to monitor", len(tier2))

    counts = {"updated": 0, "demoted": 0}
    statements = []
    ts = now_iso()

    conn = db_connect()
    try:
        for row in tier2:
            name = row["strategy_name"]
            ft = _get_ft_metrics_from_paper_trades(conn, name)

            if not ft:
                log.debug("  %s: no forward-test data yet", name)
                continue

            log.info(
                "  %s: ft_trades=%d, WR=%.1f%%, sharpe=%.3f, EEP=%.1f",
                name,
                ft["trades"],
                ft["win_rate"] * 100,
                ft["sharpe"],
                ft["eep_score"],
            )

            should_demote = (
                ft.get("eep_score", 999) < DEMOTE_MAX_EEP
                and ft.get("trades", 0) >= DEMOTE_MIN_TRADES
            )

            new_tier = 1 if should_demote else 2
            demoted_at = ts if should_demote else None

            if should_demote:
                log.warning(
                    "  ⬇️  %s — DEMOTE to Tier 1 (EEP=%.1f < %.0f, trades=%d)",
                    name, ft["eep_score"], DEMOTE_MAX_EEP, ft["trades"],
                )
                counts["demoted"] += 1

            counts["updated"] += 1

            if not dry_run:
                statements.append((
                    """
                    UPDATE strategy_registry SET
                        ft_win_rate   = ?,
                        ft_sharpe     = ?,
                        ft_pnl_pct    = ?,
                        ft_max_dd     = ?,
                        ft_trades     = ?,
                        ft_eep_score  = ?,
                        ft_last_update = ?,
                        tier          = ?,
                        demoted_at    = CASE WHEN ? IS NOT NULL THEN ? ELSE demoted_at END,
                        updated_at    = ?
                    WHERE strategy_name = ?
                    """,
                    (
                        round(ft["win_rate"], 6),
                        round(ft["sharpe"], 6),
                        round(ft["pnl_pct"], 6),
                        round(ft["max_dd"], 6),
                        ft["trades"],
                        round(ft["eep_score"], 4),
                        ts,
                        new_tier,
                        demoted_at, demoted_at,
                        ts,
                        name,
                    ),
                ))
    finally:
        conn.close()

    if not dry_run and statements:
        written = db_batch_write(statements)
        log.info("  Phase 4 complete: %d DB rows updated", written)

    log.info("  Result: %d Tier-2 strategies updated, %d demoted", counts["updated"], counts["demoted"])
    return counts


# ── Phase 5: EEP Score Refresh ───────────────────────────────────────────────

def phase5_eep_refresh(dry_run: bool) -> None:
    """
    Refresh all EEP scores in strategy_registry from strategy_scores and paper_trades.
    Delegates to populate_registry_eep logic.
    """
    log.info("─── Phase 5: EEP Score Refresh ───")

    if dry_run:
        log.info("  [DRY-RUN] Would refresh all EEP scores")
        return

    try:
        # Import and reuse logic from populate_registry_eep
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "populate_registry_eep",
            REPO_ROOT / "populate_registry_eep.py",
        )
        mod = _ilu.module_from_spec(spec)

        # Monkey-patch the DB_PATH in that module before running
        spec.loader.exec_module(mod)

        # Call their helpers with our DB path
        ts = now_iso()

        conn_ro = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)
        bt_metrics  = mod.get_backtest_metrics(conn_ro)
        ft_trades   = mod.get_forward_test_trades(conn_ro)
        all_strats  = [r[0] for r in conn_ro.execute("SELECT strategy_name FROM strategy_registry").fetchall()]
        conn_ro.close()

        from eep_scorer import compute_eep_from_metrics, compute_eep_from_trades

        bt_results = {
            s: {"eep": compute_eep_from_metrics(m), "metrics": m}
            for s, m in bt_metrics.items()
        }
        ft_results = {}
        for strat, trades in ft_trades.items():
            eep = compute_eep_from_trades(trades, label=strat)
            from populate_registry_eep import compute_ft_metrics_from_trades
            ft_m = compute_ft_metrics_from_trades(trades)
            ft_results[strat] = {"eep": eep, "metrics": ft_m}

        updated = mod.write_updates_with_retry(all_strats, bt_results, ft_results, ts)
        log.info("  EEP refresh complete: %d rows updated", updated)

    except Exception as exc:
        log.error("  EEP refresh failed: %s", exc)


# ── Phase 6: Design New Strategies ──────────────────────────────────────────

def phase6_design_new_strategy(
    registry: Dict[str, Dict],
    run_count: int,
    dry_run: bool,
) -> Optional[Dict]:
    """
    On every DESIGN_EVERY_N_RUNS-th pipeline run, call StrategyDesigner.
    Only if current Tier 0 count < TIER0_DESIGN_CAP.
    Returns the result dict from design_new_strategy(), or None if skipped/failed.
    """
    if run_count % DESIGN_EVERY_N_RUNS != 0:
        log.info(
            "─── Phase 6: Strategy Design (skip — run %d, design on every %d) ───",
            run_count, DESIGN_EVERY_N_RUNS,
        )
        return None

    tier0_count = sum(1 for r in registry.values() if r["tier"] == 0 and not r.get("archived"))
    log.info("─── Phase 6: Design New Strategy (run=%d, Tier0=%d) ───", run_count, tier0_count)

    if tier0_count >= TIER0_DESIGN_CAP:
        log.info("  Tier 0 has %d strategies (cap=%d), skipping design", tier0_count, TIER0_DESIGN_CAP)
        return None

    if dry_run:
        log.info("  [DRY-RUN] Would call StrategyDesigner.design_new_strategy()")
        return None

    try:
        from orchestration.strategy_designer import StrategyDesigner
        designer = StrategyDesigner(
            db_path=str(DB_PATH),
            strategies_dir=str(STRATEGIES_DIR),
        )
        result = designer.design_new_strategy()
        if result:
            # Register the new strategy in strategy_registry
            ts = now_iso()
            strat_name = result.get("strategy_name", "")
            file_path  = result.get("filepath", "")
            if strat_name:
                try:
                    db_batch_write([(
                        """
                        INSERT OR IGNORE INTO strategy_registry
                            (strategy_name, tier, source, file_path, description, created_at, updated_at)
                        VALUES (?, 0, 'strategy_designer', ?, 'Auto-designed by StrategyDesigner', ?, ?)
                        """,
                        (strat_name, file_path, ts, ts),
                    )])
                    log.info("  Registered new strategy: %s at Tier 0", strat_name)
                except Exception as db_exc:
                    log.warning("  Could not register new strategy in registry: %s", db_exc)
            return result
        else:
            log.warning("  StrategyDesigner returned None (design failed)")
            return None

    except Exception as exc:
        log.error("  Phase 6 failed: %s", exc, exc_info=True)
        return None


# ── Summary printer ──────────────────────────────────────────────────────────

def print_summary(
    new_registered: List[str],
    ph2: Dict[str, int],
    ph2b: Dict[str, int],
    ph3: Dict[str, int],
    ph4: Dict[str, int],
    designed: Optional[Dict],
    dry_run: bool,
    elapsed: float,
) -> None:
    mode = " [DRY-RUN]" if dry_run else ""
    log.info("")
    log.info("═" * 60)
    log.info("PIPELINE SUMMARY%s  (%.1fs)", mode, elapsed)
    log.info("═" * 60)
    log.info("  Phase 1   — New strategies registered : %d", len(new_registered))
    if new_registered:
        for n in new_registered:
            log.info("               • %s", n)
    log.info("  Phase 2   — Tier 0 backtested         : promoted=%d, stayed=%d, skipped=%d",
             ph2["promoted"], ph2["failed"], ph2["skipped"])
    log.info("  Phase 2.5 — Tier 1 re-backtested      : backtested=%d, skipped=%d, failed=%d",
             ph2b["backtested"], ph2b["skipped"], ph2b["failed"])
    log.info("  Phase 3   — Tier 1 evaluated          : promoted=%d, stayed=%d",
             ph3["promoted"], ph3["stayed"])
    log.info("  Phase 4   — Tier 2 monitored          : updated=%d, demoted=%d",
             ph4["updated"], ph4["demoted"])
    log.info("  Phase 6   — Strategies designed       : %s",
             designed.get("strategy_name") if designed else "none")
    log.info("═" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy Lifecycle Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run hourly to feed strategies through the 3-tier lifecycle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what WOULD happen without making any changes",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run only a specific phase (1-6; use 2.5 via --rephase flag for T1 re-backtest)",
    )
    parser.add_argument(
        "--skip-t1-rebacktest",
        action="store_true",
        help="Skip Phase 2.5 (T1 re-backtest). Use if you want fast Phase 3 evaluation only.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run
    skip_t1_rebacktest = args.skip_t1_rebacktest

    start = datetime.now(timezone.utc)
    log.info("=" * 60)
    log.info("PIPELINE START  %s%s", start.isoformat(), "  [DRY-RUN]" if dry_run else "")
    log.info("  DB:         %s", DB_PATH)
    log.info("  Strategies: %s", STRATEGIES_DIR)
    log.info("  Symbols:    %s", BACKTEST_SYMBOLS)
    log.info("=" * 60)

    # Load pipeline state (run counter)
    state = load_pipeline_state()
    run_count = state.get("run_count", 0) + 1

    # Load all strategy adapters from disk
    log.info("Loading strategy files…")
    adapters = load_all_strategies()
    log.info("Loaded %d strategy adapters from disk", len(adapters))

    # Load current registry snapshot
    conn = db_connect()
    registry = get_registry(conn)
    conn.close()
    log.info("Registry has %d strategies (Tier0=%d, Tier1=%d, Tier2=%d)",
             len(registry),
             sum(1 for r in registry.values() if r["tier"] == 0),
             sum(1 for r in registry.values() if r["tier"] == 1),
             sum(1 for r in registry.values() if r["tier"] == 2))

    # ── Execute phases ────────────────────────────────────────────────────────
    only_phase = args.phase
    new_registered: List[str] = []
    ph2  = {"promoted": 0, "failed": 0, "skipped": 0}
    ph2b = {"backtested": 0, "skipped": 0, "failed": 0}
    ph3  = {"promoted": 0, "stayed": 0}
    ph4  = {"updated": 0, "demoted": 0}
    designed = None

    if not only_phase or only_phase == 1:
        new_registered = phase1_discover_register(adapters, dry_run)
        # Refresh registry after registration
        if new_registered and not dry_run:
            conn = db_connect()
            registry = get_registry(conn)
            conn.close()

    if not only_phase or only_phase == 2:
        ph2 = phase2_backtest_tier0(adapters, registry, dry_run)
        # Refresh registry after promotions
        if not dry_run:
            conn = db_connect()
            registry = get_registry(conn)
            conn.close()

    # Phase 2.5: Re-backtest Tier 1 strategies with FULL_SYMBOLS before gate check.
    # Runs unless --skip-t1-rebacktest is passed or a specific other phase is requested.
    if not only_phase and not skip_t1_rebacktest:
        ph2b = phase2b_rebacktest_tier1(adapters, registry, dry_run)
        # Always refresh registry so Phase 3 sees the fresh bt_* data
        if not dry_run:
            conn = db_connect()
            registry = get_registry(conn)
            conn.close()

    if not only_phase or only_phase == 3:
        ph3 = phase3_evaluate_tier1(registry, dry_run)
        if not dry_run:
            conn = db_connect()
            registry = get_registry(conn)
            conn.close()

    if not only_phase or only_phase == 4:
        ph4 = phase4_monitor_tier2(registry, dry_run)
        if not dry_run:
            conn = db_connect()
            registry = get_registry(conn)
            conn.close()

    if not only_phase or only_phase == 5:
        phase5_eep_refresh(dry_run)

    if not only_phase or only_phase == 6:
        designed = phase6_design_new_strategy(registry, run_count, dry_run)

    # ── Save updated state ────────────────────────────────────────────────────
    if not dry_run:
        state["run_count"] = run_count
        state["last_run"]  = now_iso()
        save_pipeline_state(state)
        log.info("Pipeline state saved (run_count=%d)", run_count)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print_summary(new_registered, ph2, ph2b, ph3, ph4, designed, dry_run, elapsed)
    log.info("PIPELINE DONE")


if __name__ == "__main__":
    main()
