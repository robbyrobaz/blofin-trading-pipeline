"""
Database connector for ML pipeline - saves training results to ml_model_results table.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List
import time


class MLDatabaseConnector:
    """Handles saving ML model training results to database."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connector.
        
        Args:
            db_path: Path to blofin_monitor.db
        """
        self.db_path = db_path
    
    def save_training_result(self, result: Dict[str, Any]) -> int:
        """
        Save a single model training result to database.
        
        Args:
            result: Training result dict from TrainingPipeline
            
        Returns:
            Row ID of inserted record
        """
        if not result.get("success", False):
            print(f"⚠ Skipping failed model: {result.get('model_name')}")
            return None
        
        model_name = result["model_name"]
        metrics = result.get("metrics", {})
        
        # Prepare data
        ts_ms = int(time.time() * 1000)
        ts_iso = datetime.utcnow().isoformat() + 'Z'
        
        # Extract metrics (handle both classification and regression)
        train_accuracy = metrics.get("train_accuracy", metrics.get("train_r2"))
        test_accuracy = metrics.get("test_accuracy", metrics.get("test_r2"))
        f1_score = metrics.get("f1_score")
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        roc_auc = metrics.get("roc_auc")
        
        # Build config and full metrics JSON
        config_json = json.dumps({
            "model_dir": result.get("model_dir"),
            "training_time": result.get("training_time"),
            "timestamp": ts_iso
        })
        
        metrics_json = json.dumps(metrics)
        
        # Determine model type from name
        model_type_map = {
            "direction_predictor": "classification",
            "risk_scorer": "regression",
            "price_predictor": "regression",
            "momentum_classifier": "classification",
            "volatility_regressor": "regression"
        }
        model_type = model_type_map.get(model_name, "unknown")
        
        # Insert into database
        con = sqlite3.connect(self.db_path)
        cursor = con.cursor()
        
        cursor.execute("""
            INSERT INTO ml_model_results (
                ts_ms, ts_iso, model_name, model_type, symbol,
                features_json, train_accuracy, test_accuracy,
                f1_score, precision_score, recall_score, roc_auc,
                config_json, metrics_json, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts_ms, ts_iso, model_name, model_type, "BTC-USDT",  # Default symbol
            None,  # features_json (can add later if needed)
            train_accuracy, test_accuracy, f1_score, precision, recall, roc_auc,
            config_json, metrics_json, 0  # archived=0
        ))
        
        row_id = cursor.lastrowid
        con.commit()
        con.close()
        
        print(f"✓ Saved {model_name} to database (id={row_id})")
        return row_id
    
    def save_all_results(self, training_results: Dict[str, Any]) -> List[int]:
        """
        Save all model training results to database.
        
        Args:
            training_results: Dict from TrainingPipeline.train_all_models()
            
        Returns:
            List of inserted row IDs
        """
        results = training_results.get("results", {})
        
        row_ids = []
        for model_name, result in results.items():
            row_id = self.save_training_result(result)
            if row_id:
                row_ids.append(row_id)
        
        return row_ids
    
    def get_latest_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest training results from database.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of result dicts
        """
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        
        cursor = con.execute("""
            SELECT * FROM ml_model_results
            WHERE archived = 0
            ORDER BY ts_ms DESC
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        con.close()
        
        return results
