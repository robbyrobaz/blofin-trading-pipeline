"""
Build training dataset for entry classifier.

Joins paper_trades -> confirmed_signals -> signals to create
feature vectors at signal time T with labels from actual trade outcomes.

Key design decisions:
- All features come from data available AT signal time (no future leakage)
- Labels are binary: pnl_pct > 0 (win=1, loss=0)
- Dataset is sorted chronologically (never shuffled)
- 24h temporal embargo between train and test sets
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# Strategy-specific feature columns extracted from details_json.
# Each strategy has different keys; we build a unified flat feature set
# with NaN for missing fields (filled to 0 at training time).
STRATEGY_DETAIL_KEYS = [
    # bb_squeeze
    "band_width_pct", "mean",
    # breakout
    "prev_high", "prev_low", "threshold",
    # candle_patterns
    "body_pct", "upper_wick", "lower_wick",
    # momentum
    "change_pct",
    # reversal
    "bounce_pct", "reject_pct",
    # rsi_divergence
    "rsi",
    # support_resistance
    "distance_pct", "touches",
    # volume_mean_reversion
    "deviation_pct", "volume_ratio", "volatility_pct",
    # vwap_reversion
    # deviation_pct already included above
]

# Deduplicate while preserving order
STRATEGY_DETAIL_KEYS = list(dict.fromkeys(STRATEGY_DETAIL_KEYS))

STRATEGIES = [
    "bb_squeeze", "breakout", "candle_patterns", "ema_crossover",
    "momentum", "reversal", "rsi_divergence", "support_resistance",
    "volume_mean_reversion", "vwap_reversion",
]


def load_entry_dataset(db_path: str) -> pd.DataFrame:
    """
    Load signal->trade pairs from database.

    Returns DataFrame sorted by signal timestamp (ascending) with:
    - Feature columns: strategy, symbol, side, confidence, score,
      hour_of_day, day_of_week, and strategy-specific detail fields
    - Label column: won (1 if pnl_pct > 0, else 0)
    - Metadata: signal_ts_ms (for temporal splitting)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            s.id           AS signal_id,
            s.ts_ms        AS signal_ts_ms,
            s.symbol       AS symbol,
            s.signal       AS side,
            s.strategy     AS strategy,
            s.confidence   AS confidence,
            s.details_json AS details_json,
            cs.score       AS cs_score,
            pt.pnl_pct     AS pnl_pct,
            pt.opened_ts_ms AS opened_ts_ms,
            pt.closed_ts_ms AS closed_ts_ms
        FROM paper_trades pt
        JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
        JOIN signals s            ON cs.signal_id = s.id
        WHERE pt.status = 'CLOSED'
        ORDER BY s.ts_ms ASC
    """).fetchall()

    conn.close()

    records = []
    for row in rows:
        rec = {
            "signal_ts_ms": row["signal_ts_ms"],
            "symbol": row["symbol"],
            "side": 1 if row["side"] == "BUY" else 0,
            "confidence": float(row["confidence"] or 0.5),
            "cs_score": float(row["cs_score"] or 0.5),
            "won": 1 if (row["pnl_pct"] or 0) > 0 else 0,
        }

        # Time-of-day features (hour, day of week) derived from signal timestamp
        dt = datetime.fromtimestamp(row["signal_ts_ms"] / 1000, tz=timezone.utc)
        rec["hour_sin"] = float(np.sin(2 * np.pi * dt.hour / 24))
        rec["hour_cos"] = float(np.cos(2 * np.pi * dt.hour / 24))
        rec["dow_sin"] = float(np.sin(2 * np.pi * dt.weekday() / 7))
        rec["dow_cos"] = float(np.cos(2 * np.pi * dt.weekday() / 7))

        # One-hot encode strategy
        strategy = row["strategy"] or "unknown"
        for s in STRATEGIES:
            rec[f"strat_{s}"] = 1 if strategy == s else 0

        # Extract strategy-specific detail fields (NaN if absent)
        details = {}
        if row["details_json"]:
            try:
                details = json.loads(row["details_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        for key in STRATEGY_DETAIL_KEYS:
            val = details.get(key)
            rec[f"detail_{key}"] = float(val) if val is not None else np.nan

        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values("signal_ts_ms").reset_index(drop=True)

    print(f"Loaded {len(df)} signal->trade pairs")
    print(f"  Win rate: {df['won'].mean():.1%}  ({df['won'].sum()} wins / {(1-df['won']).sum()} losses)")
    print(f"  Date range: {datetime.fromtimestamp(df['signal_ts_ms'].min()/1000, tz=timezone.utc).date()} "
          f"to {datetime.fromtimestamp(df['signal_ts_ms'].max()/1000, tz=timezone.utc).date()}")

    return df


def temporal_split_with_embargo(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    embargo_hours: float = 24.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically with a temporal embargo gap.

    The embargo prevents any training sample within `embargo_hours` of the
    first test sample from being used, eliminating look-ahead bias from
    autocorrelated features.

    Args:
        df: DataFrame sorted by signal_ts_ms ascending
        test_ratio: Fraction of samples for the test set
        embargo_hours: Hours of gap between last train and first test sample

    Returns:
        (train_df, test_df) — no overlap, no shuffle
    """
    split_idx = int(len(df) * (1.0 - test_ratio))
    test_start_ts = int(df.iloc[split_idx]["signal_ts_ms"])
    embargo_ms = int(embargo_hours * 3600 * 1000)
    embargo_cutoff = test_start_ts - embargo_ms

    train_df = df[df["signal_ts_ms"] <= embargo_cutoff].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Temporal split (24h embargo):")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Embargo gap: {embargo_hours:.0f}h ({split_idx - len(train_df)} samples dropped)")
    print(f"  Test:  {len(test_df)} samples")

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return numeric feature column names (everything except metadata, label, and string cols)."""
    exclude = {"signal_ts_ms", "won", "symbol"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in (float, int, "float64", "int64", "int32")
    ]


if __name__ == "__main__":
    import os
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "blofin_monitor.db"
    )
    df = load_entry_dataset(db_path)
    train_df, test_df = temporal_split_with_embargo(df)
    print(f"\nFeature columns ({len(get_feature_columns(df))}):")
    print(get_feature_columns(df))
