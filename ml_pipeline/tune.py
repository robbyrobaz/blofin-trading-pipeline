"""
Tuning Pipeline - Drift detection and model retraining.
"""
import os
import sys
import json
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common.predictor import load_model
from ml_pipeline.validate import ValidationPipeline
from ml_pipeline.train import TrainingPipeline


class TuningPipeline:
    """Detects model drift and triggers retraining."""
    
    def __init__(
        self,
        base_model_dir: str = "models",
        drift_threshold: float = 0.10,
        history_file: str = "ml_pipeline/performance_history.json"
    ):
        """
        Initialize tuning pipeline.
        
        Args:
            base_model_dir: Base directory containing trained models
            drift_threshold: Performance drop threshold to trigger retraining (10% default)
            history_file: Path to performance history JSON
        """
        self.base_model_dir = base_model_dir
        self.drift_threshold = drift_threshold
        self.history_file = history_file
        self.performance_history = self._load_history()
        self.drift_log = []
    
    def _load_history(self) -> Dict:
        """Load performance history from disk."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_history(self) -> None:
        """Save performance history to disk."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def detect_drifting_models(
        self,
        features_df: pd.DataFrame,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Detect which models are experiencing performance drift.
        
        Args:
            features_df: DataFrame with features and targets
            days_back: Days to use for recent performance evaluation
            
        Returns:
            List of dicts with drift information for each model
        """
        print("\n" + "="*60)
        print("DETECTING MODEL DRIFT")
        print("="*60)
        
        # Validate current performance
        validator = ValidationPipeline(self.base_model_dir)
        current_results = validator.validate_all_models(features_df, days_back=days_back)
        
        drifting_models = []
        
        for model_name, result in current_results["results"].items():
            if not result.get("success"):
                continue
            
            current_metrics = result["metrics"]
            
            # Get historical performance
            if model_name not in self.performance_history:
                # First run, save baseline
                self.performance_history[model_name] = {
                    "baseline": current_metrics,
                    "history": [current_metrics],
                }
                print(f"✓ {model_name}: Baseline established")
                continue
            
            baseline = self.performance_history[model_name]["baseline"]
            
            # Determine primary metric based on model type
            if "accuracy" in current_metrics:
                primary_metric = "accuracy"
                # For accuracy, lower is worse
                baseline_value = baseline.get(primary_metric, 0)
                current_value = current_metrics.get(primary_metric, 0)
                drift_amount = baseline_value - current_value
                is_drifting = drift_amount > self.drift_threshold
            elif "mae" in current_metrics:
                primary_metric = "mae"
                # For MAE, higher is worse
                baseline_value = baseline.get(primary_metric, float('inf'))
                current_value = current_metrics.get(primary_metric, float('inf'))
                drift_amount = current_value - baseline_value
                # Calculate relative drift
                relative_drift = drift_amount / (baseline_value + 1e-10)
                is_drifting = relative_drift > self.drift_threshold
            else:
                print(f"⚠ {model_name}: No suitable metric for drift detection")
                continue
            
            # Update history
            self.performance_history[model_name]["history"].append(current_metrics)
            
            if is_drifting:
                drift_info = {
                    "model_name": model_name,
                    "primary_metric": primary_metric,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "drift_amount": drift_amount,
                    "drift_percent": (drift_amount / (baseline_value + 1e-10)) * 100,
                    "timestamp": datetime.now().isoformat(),
                }
                drifting_models.append(drift_info)
                
                self.drift_log.append(drift_info)
                
                print(f"⚠ {model_name}: DRIFT DETECTED")
                print(f"  - {primary_metric}: {baseline_value:.4f} → {current_value:.4f}")
                print(f"  - Drift: {drift_amount:.4f} ({drift_info['drift_percent']:.1f}%)")
            else:
                print(f"✓ {model_name}: Performance stable")
                print(f"  - {primary_metric}: {baseline_value:.4f} → {current_value:.4f}")
        
        # Save updated history
        self._save_history()
        
        print("\n" + "="*60)
        print(f"DRIFT DETECTION COMPLETE: {len(drifting_models)} models need retraining")
        print("="*60)
        
        return drifting_models
    
    def retrain_model(
        self,
        model_name: str,
        features_df: pd.DataFrame,
        update_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Retrain a specific model on fresh data.
        
        Args:
            model_name: Name of the model to retrain
            features_df: Fresh training data
            update_baseline: Whether to update baseline after retraining
            
        Returns:
            Dict with retraining results
        """
        print(f"\n{'='*60}")
        print(f"RETRAINING {model_name}")
        print(f"{'='*60}")
        
        try:
            # Get model-target mapping
            model_targets = {
                "direction_predictor": "target_direction",
                "risk_scorer": "target_risk",
                "price_predictor": "target_price",
                "momentum_classifier": "target_momentum",
                "volatility_regressor": "target_volatility",
            }
            
            if model_name not in model_targets:
                return {"success": False, "error": f"Unknown model: {model_name}"}
            
            target_col = model_targets[model_name]
            
            if target_col not in features_df.columns:
                return {"success": False, "error": f"Target column '{target_col}' not found"}
            
            # Prepare training data
            y = features_df[target_col]
            X = features_df.drop(columns=[target_col], errors="ignore")
            
            # Feature selection (drop low-importance features)
            X_selected = self._select_features(model_name, X)
            
            # Initialize training pipeline
            trainer = TrainingPipeline(self.base_model_dir)
            
            # Train single model
            result = trainer.train_single_model(model_name, X_selected, y)
            
            if result.get("success"):
                # Update baseline if requested
                if update_baseline and model_name in self.performance_history:
                    new_baseline = result["metrics"]
                    old_baseline = self.performance_history[model_name]["baseline"]
                    
                    self.performance_history[model_name]["baseline"] = new_baseline
                    self.performance_history[model_name]["retrained_at"] = datetime.now().isoformat()
                    self._save_history()
                    
                    print(f"✓ Baseline updated for {model_name}")
                
                # Log retraining
                retrain_log = {
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "old_performance": self.performance_history.get(model_name, {}).get("baseline", {}),
                    "new_performance": result.get("metrics", {}),
                }
                
                log_file = "ml_pipeline/retrain_log.json"
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                else:
                    log_data = []
                
                log_data.append(retrain_log)
                
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                print(f"✓ Retraining logged to {log_file}")
            
            return result
            
        except Exception as e:
            print(f"✗ Retraining failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _select_features(self, model_name: str, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select important features, drop unimportant ones.
        
        Args:
            model_name: Name of the model
            X: Feature dataframe
            
        Returns:
            DataFrame with selected features
        """
        try:
            # Load model to get feature importance
            model = load_model(model_name, self.base_model_dir)
            importance = model.get_feature_importance()
            
            if not importance:
                # No feature importance available, return all features
                return X
            
            # Drop features with very low importance (< 1% of max)
            max_importance = max(importance.values())
            threshold = max_importance * 0.01
            
            selected_features = [
                feat for feat, imp in importance.items()
                if imp >= threshold and feat in X.columns
            ]
            
            if selected_features:
                print(f"  Selected {len(selected_features)}/{len(importance)} features")
                return X[selected_features]
            else:
                return X
                
        except Exception as e:
            print(f"  ⚠ Feature selection failed: {str(e)}, using all features")
            return X
    
    def auto_tune(
        self,
        features_df: pd.DataFrame,
        days_back: int = 7,
        auto_retrain: bool = True
    ) -> Dict[str, Any]:
        """
        Automatically detect drift and retrain models.
        
        Args:
            features_df: DataFrame with features and targets
            days_back: Days for drift detection
            auto_retrain: Whether to automatically retrain drifting models
            
        Returns:
            Dict with tuning results
        """
        print("\n" + "="*60)
        print("AUTO-TUNING PIPELINE")
        print("="*60)
        
        # Detect drift
        drifting_models = self.detect_drifting_models(features_df, days_back)
        
        retrain_results = {}
        
        if auto_retrain and drifting_models:
            print(f"\nAuto-retraining {len(drifting_models)} models...")
            
            for drift_info in drifting_models:
                model_name = drift_info["model_name"]
                result = self.retrain_model(model_name, features_df)
                retrain_results[model_name] = result
        
        return {
            "drifting_models": drifting_models,
            "retrain_results": retrain_results,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """Main entry point for tuning."""
    print("ML Tuning Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TuningPipeline(
        base_model_dir="models",
        drift_threshold=0.10
    )
    
    # Generate synthetic data
    from ml_pipeline.train import TrainingPipeline
    train_pipeline = TrainingPipeline()
    features_df = train_pipeline.generate_synthetic_data(n_samples=5000)
    
    # Run auto-tuning
    results = pipeline.auto_tune(features_df, days_back=7, auto_retrain=True)
    
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    print(f"Drifting models: {len(results['drifting_models'])}")
    print(f"Retrained models: {len(results['retrain_results'])}")
    
    return results


if __name__ == "__main__":
    main()
