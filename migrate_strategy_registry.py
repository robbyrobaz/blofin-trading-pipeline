#!/usr/bin/env python3
"""
One-time migration script: creates strategy_registry table and populates it
from existing strategies/*.py files and strategy_scores data.
"""
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = "/home/rob/.openclaw/workspace/blofin-stack/data/blofin_monitor.db"
STRATEGIES_DIR = "/home/rob/.openclaw/workspace/blofin-stack/strategies"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS strategy_registry (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT UNIQUE NOT NULL,
    tier INTEGER DEFAULT 0,
    strategy_type TEXT,
    description TEXT,
    source TEXT,
    file_path TEXT,
    ml_entry_model TEXT,
    ml_exit_model TEXT,
    bt_win_rate REAL, bt_sharpe REAL, bt_pnl_pct REAL, bt_max_dd REAL,
    bt_trades INTEGER, bt_eep_score REAL, bt_last_run TEXT,
    ft_win_rate REAL, ft_sharpe REAL, ft_pnl_pct REAL, ft_max_dd REAL,
    ft_trades INTEGER, ft_eep_score REAL, ft_started TEXT, ft_last_update TEXT,
    promoted_at TEXT, demoted_at TEXT,
    archived INTEGER DEFAULT 0, archive_reason TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""

# Map strategy names to metadata: (strategy_type, description, source)
STRATEGY_METADATA = {
    # Core strategies
    "momentum":              ("pattern", "Detects strong upward or downward price momentum", "designed"),
    "breakout":              ("pattern", "Detects price breakouts above/below resistance/support", "designed"),
    "breakout_v2":           ("pattern", "Improved breakout detection with volume confirmation", "tuned"),
    "reversal":              ("pattern", "Detects price reversal patterns", "designed"),
    "vwap_reversion":        ("pattern", "VWAP mean reversion strategy", "designed"),
    "rsi_divergence":        ("pattern", "RSI overbought/oversold conditions", "designed"),
    "bb_squeeze":            ("pattern", "Bollinger Band squeeze breakout detection", "designed"),
    "bb_squeeze_v2":         ("pattern", "Improved Bollinger Band squeeze with wider filters", "tuned"),
    "ema_crossover":         ("pattern", "EMA crossover signals", "designed"),
    "volume_spike":          ("pattern", "Detects abnormal volume spikes", "designed"),
    "support_resistance":    ("pattern", "Support and resistance level detection", "designed"),
    "macd_divergence":       ("pattern", "MACD divergence signals", "designed"),
    "candle_patterns":       ("pattern", "Candlestick pattern detection", "designed"),
    "volume_mean_reversion": ("pattern", "Volume-weighted mean reversion", "designed"),
    # Batch 2
    "cross_asset_correlation":   ("pattern", "Cross-asset correlation signals", "librarian"),
    "volatility_regime_switch":  ("pattern", "Regime switching based on volatility", "librarian"),
    "ml_random_forest_15m":      ("ml_entry", "Random forest classifier on 15m features", "librarian"),
    "orderflow_imbalance":       ("pattern", "Order flow imbalance detection", "librarian"),
    "ensemble_top3":             ("ensemble", "Ensemble of top 3 performing strategies", "librarian"),
    # Ghost strategies (implemented in this migration)
    "mtf_trend_align":       ("pattern", "Multi-timeframe trend alignment: 5m RSI + 1h EMA filter", "librarian"),
    "ml_gbt_5m":             ("ml_entry", "Gradient boosted tree on 5m features", "librarian"),
    "mtf_momentum_confirm":  ("pattern", "Multi-timeframe momentum: 15m signal + 4h trend confirmation", "librarian"),
    # Versioned momentum/rsi strategies
    "momentum_v1":           ("pattern", "Momentum strategy version 1", "tuned"),
    "momentum_v2":           ("pattern", "Momentum strategy version 2", "tuned"),
    "rsi_divergence_v1":     ("pattern", "RSI divergence strategy version 1", "tuned"),
}

# Strategy_NNN files are tuned variants
STRATEGY_NNN_TYPE = ("pattern", "Auto-tuned strategy variant", "tuned")


def get_strategy_scores(conn):
    """Get best aggregated scores per strategy from strategy_scores table."""
    rows = conn.execute("""
        SELECT strategy,
               MAX(trades) as trades,
               MAX(win_rate) as win_rate,
               MAX(sharpe_ratio) as sharpe,
               MIN(max_drawdown_pct) as min_dd,
               MAX(total_pnl_pct) as total_pnl,
               MAX(ts_iso) as last_update
        FROM strategy_scores
        WHERE window = 'all' AND symbol IS NULL
        GROUP BY strategy
    """).fetchall()
    return {r[0]: r for r in rows}


def get_strategy_files():
    """Scan strategies/ directory for .py files."""
    skip = {"base_strategy.py", "__init__.py", "strategy_promoter.py"}
    files = {}
    for f in Path(STRATEGIES_DIR).glob("*.py"):
        if f.name in skip:
            continue
        name = f.stem
        files[name] = str(f)
    return files


def get_metadata(name):
    if name in STRATEGY_METADATA:
        return STRATEGY_METADATA[name]
    if name.startswith("strategy_"):
        return STRATEGY_NNN_TYPE
    return ("pattern", None, "designed")


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc).isoformat()

    print("Creating strategy_registry table...")
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()

    scores = get_strategy_scores(conn)
    strategy_files = get_strategy_files()

    print(f"Found {len(strategy_files)} strategy files")
    print(f"Found {len(scores)} strategies with score data")

    inserted = 0
    skipped = 0

    # Insert all strategies that have .py files
    for name, file_path in sorted(strategy_files.items()):
        stype, desc, source = get_metadata(name)
        score_row = scores.get(name)

        # Tier: 2 if we have paper trade data, 0 otherwise
        tier = 2 if score_row and score_row[1] and score_row[1] > 0 else 0

        ft_trades = score_row[1] if score_row else None
        ft_win_rate = score_row[2] if score_row else None
        ft_sharpe = score_row[3] if score_row else None
        ft_max_dd = score_row[4] if score_row else None
        ft_pnl_pct = score_row[5] if score_row else None
        ft_last_update = score_row[6] if score_row else None

        try:
            conn.execute("""
                INSERT OR IGNORE INTO strategy_registry (
                    strategy_name, tier, strategy_type, description, source, file_path,
                    ft_win_rate, ft_sharpe, ft_pnl_pct, ft_max_dd, ft_trades,
                    ft_last_update,
                    archived, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                name, tier, stype, desc, source, file_path,
                ft_win_rate, ft_sharpe, ft_pnl_pct, ft_max_dd, ft_trades,
                ft_last_update,
                now, now,
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    # Insert ghost strategies (in strategy_scores but no .py file)
    # After this migration they will have .py files, but mark archived ones separately
    for name, score_row in scores.items():
        if name in strategy_files:
            continue  # already handled above
        # Ghost with no file
        stype, desc, source = get_metadata(name)
        ft_trades = score_row[1]
        ft_win_rate = score_row[2]
        ft_sharpe = score_row[3]
        ft_max_dd = score_row[4]
        ft_pnl_pct = score_row[5]
        ft_last_update = score_row[6]
        try:
            conn.execute("""
                INSERT OR IGNORE INTO strategy_registry (
                    strategy_name, tier, strategy_type, description, source, file_path,
                    ft_win_rate, ft_sharpe, ft_pnl_pct, ft_max_dd, ft_trades,
                    ft_last_update,
                    archived, archive_reason, created_at, updated_at
                ) VALUES (?, 0, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, 1, 'no_file', ?, ?)
            """, (
                name, stype, desc, source,
                ft_win_rate, ft_sharpe, ft_pnl_pct, ft_max_dd, ft_trades,
                ft_last_update, now, now,
            ))
            inserted += 1
            print(f"  Ghost (archived): {name}")
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    print(f"Inserted {inserted} strategies, skipped {skipped} (already existed)")

    # Show summary
    print("\n=== strategy_registry summary ===")
    for row in conn.execute("""
        SELECT tier, COUNT(*) as cnt, GROUP_CONCAT(strategy_name, ', ') as names
        FROM strategy_registry
        WHERE archived = 0
        GROUP BY tier
        ORDER BY tier
    """):
        print(f"  Tier {row[0]}: {row[1]} strategies")
        for n in (row[2] or "").split(", "):
            print(f"    - {n}")

    archived_count = conn.execute("SELECT COUNT(*) FROM strategy_registry WHERE archived = 1").fetchone()[0]
    print(f"  Archived: {archived_count}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
