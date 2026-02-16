"""
Validation Pipeline - Backtests models on holdout data.
"""
import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common.predictor import load_model


class ValidationPipeline:
    """Validates models on holdout data and calculates performance metrics."""
    
    def __init__(self, base_model_dir: str = "models"):
        """
        Initialize validation pipeline.
        
        Args:
            base_model_dir: Base directory containing trained models
        """
        self.base_model_dir = base_model_dir
        self.validation_results = {}
    
    def backtest_model(
        self,
        model_name: str,
        features_df: pd.DataFrame,
        target_col: str,
        symbols: List[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Backtest a model on holdout data.
        
        Args:
            model_name: Name of the model to backtest
            features_df: DataFrame with features and targets
            target_col: Name of the target column
            symbols: List of symbols to test on (if applicable)
            days_back: Number of days to use for backtesting
            
        Returns:
            Dict containing validation metrics
        """
        print(f"\nBacktesting {model_name}...")
        
        try:
            # Load model
            model = load_model(model_name, self.base_model_dir)
            
            # Prepare data
            # Assume features_df is sorted by time, use last days_back for validation
            if "timestamp" in features_df.columns:
                cutoff_date = pd.Timestamp.now() - timedelta(days=days_back)
                holdout_df = features_df[features_df["timestamp"] >= cutoff_date]
            else:
                # Use last 20% as holdout
                split_idx = int(len(features_df) * 0.8)
                holdout_df = features_df.iloc[split_idx:]
            
            if len(holdout_df) == 0:
                return {
                    "success": False,
                    "error": "No holdout data available",
                }
            
            # Separate features and target
            y_true = holdout_df[target_col]
            X = holdout_df.drop(columns=[target_col], errors="ignore")
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate metrics based on model type
            if model.model_type in ["xgboost", "svm"]:
                # Classification metrics
                if isinstance(predictions, dict):
                    if "direction" in predictions:
                        y_pred = [predictions["direction"]]
                    elif "state" in predictions:
                        y_pred = [predictions["state"]]
                    elif "directions" in predictions:
                        y_pred = predictions["directions"]
                    elif "states" in predictions:
                        y_pred = predictions["states"]
                    else:
                        y_pred = [0]
                else:
                    y_pred = predictions
                
                # Convert string labels to numeric if needed
                if isinstance(y_pred[0], str):
                    label_map = {"DOWN": 0, "UP": 1} if model_name == "direction_predictor" else {
                        "decelerating": 0, "neutral": 1, "accelerating": 2
                    }
                    y_pred = [label_map.get(p, 0) for p in y_pred]
                
                # Ensure same length
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                accuracy = accuracy_score(y_true, y_pred)
                
                # For binary classification
                if len(set(y_true)) == 2:
                    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
                    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
                    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
                else:
                    # Multi-class
                    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                
                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
                
            else:
                # Regression metrics
                if isinstance(predictions, dict):
                    if "predicted_price" in predictions:
                        y_pred = [predictions["predicted_price"]]
                    elif "predicted_vol" in predictions:
                        y_pred = [predictions["predicted_vol"]]
                    elif "predicted_prices" in predictions:
                        y_pred = predictions["predicted_prices"]
                    elif "predicted_vols" in predictions:
                        y_pred = predictions["predicted_vols"]
                    else:
                        y_pred = [0]
                else:
                    y_pred = predictions
                
                # Ensure same length
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                
                # MAPE
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
                
                metrics = {
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "mape": float(mape),
                }
            
            # Add metadata
            metrics["n_samples"] = len(y_true)
            metrics["model_name"] = model_name
            metrics["timestamp"] = datetime.now().isoformat()
            
            print(f"✓ {model_name} backtest complete")
            for key, value in metrics.items():
                if isinstance(value, float) and key != "timestamp":
                    print(f"  - {key}: {value:.4f}")
            
            return {
                "success": True,
                "metrics": metrics,
            }
            
        except Exception as e:
            print(f"✗ {model_name} backtest failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def validate_all_models(
        self,
        features_df: pd.DataFrame,
        symbols: List[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Validate all trained models.
        
        Args:
            features_df: DataFrame with features and targets
            symbols: List of symbols to test on
            days_back: Number of days for backtesting
            
        Returns:
            Dict with validation results for all models
        """
        print("\n" + "="*60)
        print("VALIDATING ALL MODELS")
        print("="*60)
        
        # Define model-target mappings
        model_targets = {
            "direction_predictor": "target_direction",
            "risk_scorer": "target_risk",
            "price_predictor": "target_price",
            "momentum_classifier": "target_momentum",
            "volatility_regressor": "target_volatility",
        }
        
        results = {}
        for model_name, target_col in model_targets.items():
            if target_col not in features_df.columns:
                print(f"⚠ Skipping {model_name} - target column '{target_col}' not found")
                continue
            
            result = self.backtest_model(
                model_name=model_name,
                features_df=features_df,
                target_col=target_col,
                symbols=symbols,
                days_back=days_back
            )
            results[model_name] = result
        
        # Summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print(f"Validated: {successful}/{len(results)} models")
        
        self.validation_results = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
        
        return self.validation_results
    
    def compare_models(self, metric: str = "accuracy") -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        
        Args:
            metric: Metric to compare (accuracy, f1, mae, etc.)
            
        Returns:
            DataFrame with model comparison
        """
        if not self.validation_results:
            print("No validation results available")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, result in self.validation_results.get("results", {}).items():
            if result.get("success") and "metrics" in result:
                metrics = result["metrics"]
                if metric in metrics:
                    comparison_data.append({
                        "model": model_name,
                        metric: metrics[metric],
                        "n_samples": metrics.get("n_samples", 0),
                    })
        
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values(by=metric, ascending=False)
        
        return df


def main():
    """Main entry point for validation."""
    print("ML Validation Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ValidationPipeline(base_model_dir="models")
    
    # Generate synthetic data for testing
    from ml_pipeline.train import TrainingPipeline
    train_pipeline = TrainingPipeline()
    features_df = train_pipeline.generate_synthetic_data(n_samples=5000)
    
    # Validate all models
    results = pipeline.validate_all_models(features_df, days_back=7)
    
    # Compare models
    print("\nModel Comparison (by primary metric):")
    for model_name in results["results"].keys():
        if results["results"][model_name].get("success"):
            metrics = results["results"][model_name]["metrics"]
            if "accuracy" in metrics:
                comp_df = pipeline.compare_models("accuracy")
            elif "mae" in metrics:
                comp_df = pipeline.compare_models("mae")
            print(comp_df)
            break
    
    return results


if __name__ == "__main__":
    main()
