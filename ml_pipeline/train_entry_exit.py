"""
Training script for entry classifier and exit timing classifier.

Usage:
    cd /home/rob/.openclaw/workspace/blofin-stack
    python -m ml_pipeline.train_entry_exit

Trains both models and saves them to:
    models/entry_classifier/
    models/exit_classifier/
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = os.getenv(
    "BLOFIN_DB_PATH",
    str(ROOT / "data" / "blofin_monitor.db"),
)
MODEL_BASE = str(ROOT / "models")


def train_entry_classifier() -> dict:
    from ml_pipeline.build_entry_dataset import (
        get_feature_columns,
        load_entry_dataset,
        temporal_split_with_embargo,
    )
    from ml_pipeline.models.entry_classifier import EntryClassifier

    print("\n" + "=" * 60)
    print("TRAINING ENTRY CLASSIFIER")
    print("=" * 60)

    df = load_entry_dataset(DB_PATH)
    if len(df) < 100:
        print(f"Insufficient data: {len(df)} samples (need >= 100)")
        return {"success": False, "error": "insufficient data"}

    train_df, test_df = temporal_split_with_embargo(df, test_ratio=0.2, embargo_hours=24.0)

    if len(train_df) < 50:
        print(f"Train set too small after embargo: {len(train_df)} samples")
        return {"success": False, "error": "train set too small after embargo"}

    feature_cols = get_feature_columns(df)
    ec = EntryClassifier(model_dir=os.path.join(MODEL_BASE, "entry_classifier"))
    metrics = ec.train(train_df, test_df, feature_cols)
    ec.save()

    return {"success": True, "metrics": metrics}


def train_exit_classifier() -> dict:
    from ml_pipeline.models.exit_classifier import ExitClassifier

    print("\n" + "=" * 60)
    print("TRAINING EXIT CLASSIFIER")
    print("=" * 60)

    ec = ExitClassifier(model_dir=os.path.join(MODEL_BASE, "exit_classifier"))
    metrics = ec.train(DB_PATH)
    ec.save()

    return {"success": True, "metrics": metrics}


def save_results_to_db(entry_result: dict, exit_result: dict) -> None:
    """Save training results to ml_model_results table."""
    import sqlite3
    import time

    conn = sqlite3.connect(DB_PATH)
    ts_ms = int(time.time() * 1000)
    ts_iso = __import__("datetime").datetime.utcnow().isoformat() + "Z"

    for model_name, result in [("entry_classifier", entry_result), ("exit_classifier", exit_result)]:
        if not result.get("success"):
            continue
        m = result.get("metrics", {})
        conn.execute("""
            INSERT INTO ml_model_results (
                ts_ms, ts_iso, model_name, model_type, symbol,
                features_json, train_accuracy, test_accuracy,
                f1_score, precision_score, recall_score, roc_auc,
                config_json, metrics_json, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            ts_ms, ts_iso, model_name, "xgboost", "ALL",
            None,
            m.get("train_win_rate") or m.get("close_now_rate_train"),
            m.get("test_win_rate") or m.get("close_now_rate_test"),
            m.get("f1"),
            m.get("precision"),
            m.get("recall"),
            m.get("roc_auc"),
            json.dumps({"model_dir": os.path.join(MODEL_BASE, model_name)}),
            json.dumps(m),
        ))

    conn.commit()
    conn.close()
    print("\nResults saved to ml_model_results table.")


def main():
    print("Entry/Exit Classifier Training Pipeline")
    print("DB:", DB_PATH)
    print("Models:", MODEL_BASE)

    entry_result = train_entry_classifier()
    exit_result = train_exit_classifier()

    save_results_to_db(entry_result, exit_result)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in [("entry_classifier", entry_result), ("exit_classifier", exit_result)]:
        if result.get("success"):
            m = result["metrics"]
            print(f"  {name}: ROC-AUC={m.get('roc_auc', 0):.4f}  F1={m.get('f1', 0):.4f}")
        else:
            print(f"  {name}: FAILED — {result.get('error')}")

    return entry_result, exit_result


if __name__ == "__main__":
    main()
