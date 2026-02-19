"""
Entry Classifier — predicts whether a confirmed signal will result in a winning trade.

Training data: historical signals (from signals table) matched to paper trade
outcomes (from paper_trades table), linked via:
  paper_trades.confirmed_signal_id -> confirmed_signals.signal_id -> signals.strategy

Features: all available at signal time T (strategy, symbol, side, confidence,
score, time-of-day, strategy-specific indicator values from details_json).

Label: 1 if pnl_pct > 0 (win), 0 otherwise.

Temporal split with 24h embargo — no shuffling.
"""
import json
import os
import pickle
import sys
from datetime import datetime
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

# Allow running from ml_pipeline/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class EntryClassifier:
    """XGBoost binary classifier: signal -> win probability."""

    MODEL_FILE = "entry_classifier.pkl"
    SCALER_FILE = "entry_classifier_scaler.pkl"
    META_FILE = "entry_classifier_meta.json"

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or str(
            Path(__file__).resolve().parents[2] / "models" / "entry_classifier"
        )
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: list = []
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list,
    ) -> Dict[str, Any]:
        """
        Train on signal->trade outcome pairs.

        Args:
            train_df: Chronologically earlier portion (no shuffle)
            test_df:  Chronologically later portion, separated by 24h embargo
            feature_cols: Feature column names (no label, no ts metadata)

        Returns:
            Dict of evaluation metrics
        """
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["won"].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["won"].values

        # Scale — fit ONLY on training data
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Class imbalance: weight loss class down (more losses than wins)
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=200,
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

        # Evaluate
        y_proba = self.model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = self._compute_metrics(y_test, y_pred, y_proba)
        metrics["n_train"] = int(len(y_train))
        metrics["n_test"] = int(len(y_test))
        metrics["train_win_rate"] = float(y_train.mean())
        metrics["test_win_rate"] = float(y_test.mean())

        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_.tolist()))
        top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        self.metadata = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "feature_cols": feature_cols,
            "metrics": metrics,
            "top_features": top10,
            "scale_pos_weight": float(scale_pos_weight),
        }

        print(f"Entry classifier trained:")
        print(f"  Train: {len(y_train)} samples  |  Test: {len(y_test)} samples")
        print(f"  Test win rate (base): {y_test.mean():.1%}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}  |  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}  |  Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}")
        print(f"  Top features: {[f[0] for f in top10[:5]]}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: Dict[str, Any]) -> float:
        """
        Return probability of winning (0.0 - 1.0) for a single signal.

        Args:
            features: Dict with keys matching feature_cols (missing keys -> 0)

        Returns:
            Float probability in [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        row = np.array([features.get(col, 0.0) or 0.0 for col in self.feature_cols], dtype=float)
        row = np.nan_to_num(row, nan=0.0)
        row_s = self.scaler.transform(row.reshape(1, -1))
        return float(self.model.predict_proba(row_s)[0, 1])

    def predict_from_signal(
        self,
        strategy: str,
        symbol: str,
        side: str,
        confidence: float,
        cs_score: float,
        details_json: Optional[str],
        signal_ts_ms: int,
    ) -> float:
        """
        High-level interface: build feature vector from raw signal fields,
        return win probability.
        """
        from ml_pipeline.build_entry_dataset import STRATEGIES, STRATEGY_DETAIL_KEYS
        from datetime import timezone

        dt = datetime.fromtimestamp(signal_ts_ms / 1000, tz=timezone.utc)
        features: Dict[str, Any] = {
            "side": 1 if side == "BUY" else 0,
            "confidence": float(confidence or 0.5),
            "cs_score": float(cs_score or 0.5),
            "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dt.weekday() / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dt.weekday() / 7)),
        }
        for s in STRATEGIES:
            features[f"strat_{s}"] = 1 if strategy == s else 0

        details = {}
        if details_json:
            try:
                details = json.loads(details_json)
            except (json.JSONDecodeError, TypeError):
                pass
        for key in STRATEGY_DETAIL_KEYS:
            val = details.get(key)
            features[f"detail_{key}"] = float(val) if val is not None else 0.0

        return self.predict_proba(features)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Optional[str] = None) -> None:
        """Save model, scaler, and metadata to disk."""
        d = Path(model_dir or self.model_dir)
        d.mkdir(parents=True, exist_ok=True)

        with open(d / self.MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        with open(d / self.SCALER_FILE, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(d / self.META_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Entry classifier saved to {d}")

    def load(self, model_dir: Optional[str] = None) -> None:
        """Load model, scaler, and metadata from disk."""
        d = Path(model_dir or self.model_dir)

        with open(d / self.MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
        with open(d / self.SCALER_FILE, "rb") as f:
            self.scaler = pickle.load(f)
        with open(d / self.META_FILE) as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata.get("feature_cols", [])
        print(f"Entry classifier loaded from {d}")

    @classmethod
    def try_load(cls, model_dir: Optional[str] = None) -> Optional["EntryClassifier"]:
        """Load if model files exist, else return None (graceful fallback)."""
        ec = cls(model_dir=model_dir)
        try:
            ec.load()
            return ec
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
