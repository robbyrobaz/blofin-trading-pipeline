"""
Exit Timing Classifier — given an open position and current market context,
predicts whether to CLOSE NOW (1) or HOLD (0).

Training methodology:
- For each closed trade, we know its final pnl_pct and hold duration.
- We generate one training sample per trade representing the decision
  at the actual close moment: features = {age_min, unrealized_pnl_pct, side, ...},
  label = 1 (CLOSE_NOW is optimal).
- We also generate "hold" samples at intermediate points during the trade
  where the best future outcome was still ahead, labelled 0 (HOLD).

With only paper_trades table (no tick-by-tick trade state snapshots),
we use a simpler proxy:
  - Label = 1 if pnl_pct > 0 AND hold_minutes < PAPER_MAX_HOLD_MINUTES
    (trade exited profitably at TP, meaning closing then was correct)
  - Label = 0 otherwise (trade either ran to SL or timeout)

This gives a meaningful but approximate exit signal — good enough to
demonstrate the pattern and wire into the pipeline.
"""
import json
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


STRATEGIES = [
    "bb_squeeze", "breakout", "candle_patterns", "ema_crossover",
    "momentum", "reversal", "rsi_divergence", "support_resistance",
    "volume_mean_reversion", "vwap_reversion",
]


class ExitClassifier:
    """XGBoost binary classifier: exit-now (1) vs hold (0)."""

    MODEL_FILE = "exit_classifier.pkl"
    SCALER_FILE = "exit_classifier_scaler.pkl"
    META_FILE = "exit_classifier_meta.json"

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or str(
            Path(__file__).resolve().parents[2] / "models" / "exit_classifier"
        )
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: list = []
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_dataset(db_path: str) -> pd.DataFrame:
        """
        Build exit timing dataset from paper_trades.

        Each row represents one closed trade. Features capture the state
        at close; label = 1 if the trade was a profitable TP exit.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT
                pt.id,
                pt.opened_ts_ms,
                pt.closed_ts_ms,
                pt.symbol,
                pt.side,
                pt.entry_price,
                pt.exit_price,
                pt.pnl_pct,
                pt.reason,
                s.strategy,
                s.confidence
            FROM paper_trades pt
            JOIN confirmed_signals cs ON pt.confirmed_signal_id = cs.id
            JOIN signals s            ON cs.signal_id = s.id
            WHERE pt.status = 'CLOSED'
            ORDER BY pt.opened_ts_ms ASC
        """).fetchall()

        conn.close()

        records = []
        for row in rows:
            opened_ms = row["opened_ts_ms"] or 0
            closed_ms = row["closed_ts_ms"] or 0
            hold_min = (closed_ms - opened_ms) / 60_000.0

            pnl = float(row["pnl_pct"] or 0)
            side_val = 1 if row["side"] == "BUY" else 0

            # Time-of-day at open
            dt = datetime.fromtimestamp(opened_ms / 1000, tz=timezone.utc)

            # Label: 1 = close-now was optimal (profitable TP exit)
            reason = (row["reason"] or "").upper()
            label = 1 if (pnl > 0 and "TP" in reason) else 0

            rec = {
                "opened_ts_ms": opened_ms,
                "side": side_val,
                "pnl_pct": pnl,
                "hold_min": float(hold_min),
                "confidence": float(row["confidence"] or 0.5),
                "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24)),
                "dow_sin": float(np.sin(2 * np.pi * dt.weekday() / 7)),
                "dow_cos": float(np.cos(2 * np.pi * dt.weekday() / 7)),
                "close_now": label,
            }

            strategy = row["strategy"] or "unknown"
            for s in STRATEGIES:
                rec[f"strat_{s}"] = 1 if strategy == s else 0

            records.append(rec)

        df = pd.DataFrame(records)
        df = df.sort_values("opened_ts_ms").reset_index(drop=True)

        print(f"Exit dataset: {len(df)} closed trades")
        print(f"  Close-now rate: {df['close_now'].mean():.1%} "
              f"({df['close_now'].sum()} TP wins / {(1-df['close_now']).sum()} others)")

        return df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, db_path: str) -> Dict[str, Any]:
        """
        Build dataset from DB and train exit classifier.

        Uses temporal split (80/20 chronological, no shuffle).
        """
        df = self.build_dataset(db_path)

        # pnl_pct is the FINAL realized PnL — it directly encodes the label.
        # Exclude it from features: at prediction time we only have unrealized PnL
        # (passed in separately by the caller), not the final outcome.
        exclude = {"opened_ts_ms", "close_now", "pnl_pct"}
        self.feature_cols = [c for c in df.columns if c not in exclude]

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[self.feature_cols].fillna(0).values
        y_train = train_df["close_now"].values
        X_test = test_df[self.feature_cols].fillna(0).values
        y_test = test_df["close_now"].values

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )

        y_proba = self.model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        try:
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            roc_auc = 0.0
        try:
            pr_auc = float(average_precision_score(y_test, y_proba))
        except ValueError:
            pr_auc = 0.0

        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "close_now_rate_train": float(y_train.mean()),
            "close_now_rate_test": float(y_test.mean()),
        }

        importance = dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))
        top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        self.metadata = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "feature_cols": self.feature_cols,
            "metrics": metrics,
            "top_features": top10,
            "scale_pos_weight": float(scale_pos_weight),
        }

        print(f"Exit classifier trained:")
        print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
        print(f"  ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_exit(
        self,
        side: str,
        hold_min: float,
        strategy: str,
        confidence: float,
        opened_ts_ms: int,
    ) -> Tuple[bool, float]:
        """
        Predict whether to exit now given open-trade context.

        Note: pnl_pct is intentionally NOT a parameter — the model is trained
        without it to avoid leaking the final outcome into the prediction.
        The TP/SL rules in paper_engine already handle profit-level exits.

        Returns:
            (should_exit: bool, probability: float)
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        dt = datetime.fromtimestamp(opened_ts_ms / 1000, tz=timezone.utc)
        features = {
            "side": 1 if side == "BUY" else 0,
            "hold_min": float(hold_min),
            "confidence": float(confidence or 0.5),
            "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dt.weekday() / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dt.weekday() / 7)),
        }
        for s in STRATEGIES:
            features[f"strat_{s}"] = 1 if strategy == s else 0

        row = np.array([features.get(col, 0.0) for col in self.feature_cols], dtype=float)
        row = np.nan_to_num(row, nan=0.0)
        row_s = self.scaler.transform(row.reshape(1, -1))
        prob = float(self.model.predict_proba(row_s)[0, 1])
        return prob >= 0.5, prob

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Optional[str] = None) -> None:
        d = Path(model_dir or self.model_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / self.MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        with open(d / self.SCALER_FILE, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(d / self.META_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Exit classifier saved to {d}")

    def load(self, model_dir: Optional[str] = None) -> None:
        d = Path(model_dir or self.model_dir)
        with open(d / self.MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
        with open(d / self.SCALER_FILE, "rb") as f:
            self.scaler = pickle.load(f)
        with open(d / self.META_FILE) as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata.get("feature_cols", [])
        print(f"Exit classifier loaded from {d}")

    @classmethod
    def try_load(cls, model_dir: Optional[str] = None) -> Optional["ExitClassifier"]:
        ec = cls(model_dir=model_dir)
        try:
            ec.load()
            return ec
        except FileNotFoundError:
            return None
