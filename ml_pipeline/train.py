"""
Training Pipeline - Orchestrates training of all models in parallel.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.models.direction_predictor import DirectionPredictor
from ml_pipeline.models.risk_scorer import RiskScorer
from ml_pipeline.models.price_predictor import PricePredictor
from ml_pipeline.models.momentum_classifier import MomentumClassifier
from ml_pipeline.models.volatility_regressor import VolatilityRegressor


class TrainingPipeline:
    """Orchestrates parallel training of all ML models."""
    
    def __init__(self, base_model_dir: str = "models"):
        """
        Initialize training pipeline.
        
        Args:
            base_model_dir: Base directory for saving models
        """
        self.base_model_dir = base_model_dir
        self.models = {
            "direction_predictor": DirectionPredictor(),
            "risk_scorer": RiskScorer(),
            "price_predictor": PricePredictor(),
            "momentum_classifier": MomentumClassifier(),
            "volatility_regressor": VolatilityRegressor(),
        }
        self.training_results = {}
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Dict[str, Tuple]:
        """
        Prepare training data for each model.
        
        Args:
            features_df: DataFrame with all features and targets
            
        Returns:
            Dict mapping model names to (X, y) tuples
        """
        print("Preparing training data for all models...")
        
        # Auto-generate targets if missing
        if "target_direction" not in features_df.columns:
            from ml_pipeline.target_generator import add_targets
            print("  ⋯ Generating targets from price data...")
            features_df = add_targets(features_df, lookback=5)
        
        training_data = {}
        exclude_cols = ["target_direction", "target_price", "target_momentum", "target_volatility", "momentum"]
        
        # Direction Predictor - predict if price goes UP/DOWN in next 5 candles
        if "target_direction" in features_df.columns:
            X = features_df.drop(columns=exclude_cols, errors="ignore")
            y = features_df["target_direction"]
            training_data["direction_predictor"] = (X, y)
            print(f"  ✓ Direction: {len(X)} samples")
        
        # Risk Scorer - use volatility as risk proxy
        if "target_volatility" in features_df.columns:
            X = features_df.drop(columns=exclude_cols, errors="ignore")
            y = (features_df["target_volatility"] > features_df["target_volatility"].median()).astype(int)
            training_data["risk_scorer"] = (X, y)
            print(f"  ✓ Risk scorer: {len(X)} samples")
        
        # Price Predictor - predict future price
        if "target_price" in features_df.columns:
            X = features_df.drop(columns=exclude_cols, errors="ignore")
            y = features_df["target_price"]
            training_data["price_predictor"] = (X, y)
            print(f"  ✓ Price: {len(X)} samples")
        
        # Momentum Classifier - classify momentum state
        if "target_momentum" in features_df.columns:
            X = features_df.drop(columns=exclude_cols, errors="ignore")
            y = features_df["target_momentum"]
            training_data["momentum_classifier"] = (X, y)
            print(f"  ✓ Momentum: {len(X)} samples")
        
        # Volatility Regressor - predict future volatility
        if "target_volatility" in features_df.columns:
            X = features_df.drop(columns=exclude_cols, errors="ignore")
            y = features_df["target_volatility"]
            training_data["volatility_regressor"] = (X, y)
            print(f"  ✓ Volatility: {len(X)} samples")
        
        return training_data
    
    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dict with training results
        """
        start_time = time.time()
        
        try:
            model = self.models[model_name]
            
            # Train model
            metrics = model.train(X, y)
            
            # Save model
            model_dir = os.path.join(self.base_model_dir, f"model_{model_name}")
            model.save(model_dir)
            
            training_time = time.time() - start_time
            
            result = {
                "success": True,
                "model_name": model_name,
                "metrics": metrics,
                "training_time": training_time,
                "model_dir": model_dir,
            }
            
            print(f"✓ {model_name} completed in {training_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e),
                "training_time": time.time() - start_time,
            }
    
    def train_all_models(
        self,
        features_df: pd.DataFrame,
        max_workers: int = 5
    ) -> Dict[str, Any]:
        """
        Train all models in parallel.
        
        Args:
            features_df: DataFrame with all features and targets
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dict with training results for all models
        """
        print("\n" + "="*60)
        print("STARTING PARALLEL MODEL TRAINING")
        print("="*60)
        
        start_time = time.time()
        
        # Prepare data for each model
        training_data = self.prepare_training_data(features_df)
        
        if not training_data:
            print("✗ No training data available. Check target columns in features_df.")
            return {"success": False, "error": "No training data"}
        
        print(f"\nTraining {len(training_data)} models in parallel...")
        print(f"Models: {', '.join(training_data.keys())}")
        
        # Train models in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training jobs
            futures = {
                executor.submit(self.train_single_model, model_name, X, y): model_name
                for model_name, (X, y) in training_data.items()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as e:
                    print(f"✗ Exception training {model_name}: {str(e)}")
                    results[model_name] = {
                        "success": False,
                        "model_name": model_name,
                        "error": str(e),
                    }
        
        total_time = time.time() - start_time
        
        # Summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        failed = len(results) - successful
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time:.1f}s")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        
        if successful > 0:
            print("\nModel Performance Summary:")
            for model_name, result in results.items():
                if result.get("success"):
                    metrics = result.get("metrics", {})
                    print(f"  {model_name}:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and "importance" not in key:
                            print(f"    - {key}: {value:.4f}")
        
        self.training_results = {
            "total_time": total_time,
            "successful": successful,
            "failed": failed,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
        
        return self.training_results
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic training data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic features and targets
        """
        print(f"Generating {n_samples} synthetic training samples...")
        
        np.random.seed(42)
        
        # Generate base features
        data = {
            # Price features
            "close": np.random.uniform(40000, 50000, n_samples),
            "high": np.random.uniform(40000, 50000, n_samples),
            "low": np.random.uniform(40000, 50000, n_samples),
            "volume": np.random.uniform(1e6, 1e9, n_samples),
            
            # Momentum features
            "momentum_roc": np.random.uniform(-5, 5, n_samples),
            "momentum_rsi": np.random.uniform(20, 80, n_samples),
            "momentum_stoch": np.random.uniform(20, 80, n_samples),
            
            # Trend features
            "trend_ema_diff": np.random.uniform(-1000, 1000, n_samples),
            "trend_macd": np.random.uniform(-500, 500, n_samples),
            "trend_adx": np.random.uniform(10, 50, n_samples),
            "trend_cci": np.random.uniform(-200, 200, n_samples),
            
            # Volatility features
            "volatility_std": np.random.uniform(100, 1000, n_samples),
            "volatility_parkinson": np.random.uniform(0.01, 0.05, n_samples),
            "atr_14": np.random.uniform(200, 800, n_samples),
            
            # Volume features
            "volume_sma": np.random.uniform(1e6, 1e9, n_samples),
            "volume_ratio": np.random.uniform(0.5, 2.0, n_samples),
            "volume_std": np.random.uniform(1e5, 1e8, n_samples),
            "volume_obv": np.random.uniform(-1e9, 1e9, n_samples),
            "volume_mfi": np.random.uniform(20, 80, n_samples),
            "volume_cmf": np.random.uniform(-0.5, 0.5, n_samples),
            "volume_volatility": np.random.uniform(0.1, 2.0, n_samples),
            
            # Additional features
            "returns": np.random.uniform(-0.05, 0.05, n_samples),
            "returns_abs": np.random.uniform(0, 0.05, n_samples),
            "acceleration": np.random.uniform(-0.01, 0.01, n_samples),
            "rsi_14": np.random.uniform(20, 80, n_samples),
            "macd": np.random.uniform(-500, 500, n_samples),
            "macd_signal": np.random.uniform(-500, 500, n_samples),
            "macd_hist": np.random.uniform(-200, 200, n_samples),
            "max_drawdown": np.random.uniform(0, 0.2, n_samples),
            "drawdown_duration": np.random.randint(0, 100, n_samples),
            "sharpe_ratio": np.random.uniform(-1, 3, n_samples),
            "high_low_range": np.random.uniform(100, 1000, n_samples),
            "close_variance": np.random.uniform(1e5, 1e7, n_samples),
            "price_range": np.random.uniform(500, 2000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate targets
        # Direction: binary classification (0=DOWN, 1=UP)
        df["target_direction"] = (df["returns"] > 0).astype(int)
        
        # Risk: 0-100 score based on volatility
        df["target_risk"] = np.clip(
            (df["volatility_std"] / 10) + np.random.uniform(-10, 10, n_samples),
            0, 100
        )
        
        # Price: future price (current + some change)
        df["target_price"] = df["close"] * (1 + np.random.uniform(-0.05, 0.05, n_samples))
        
        # Momentum: 3-class classification (0=decel, 1=neutral, 2=accel)
        df["target_momentum"] = pd.cut(
            df["momentum_roc"],
            bins=[-np.inf, -1, 1, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        # Volatility: future volatility
        df["target_volatility"] = df["volatility_std"] * np.random.uniform(0.8, 1.2, n_samples) / 1000
        
        print(f"✓ Generated synthetic data with shape {df.shape}")
        return df


def main():
    """Main entry point for training."""
    print("ML Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(base_model_dir="models")
    
    # Generate synthetic data for testing
    features_df = pipeline.generate_synthetic_data(n_samples=5000)
    
    # Train all models
    results = pipeline.train_all_models(features_df, max_workers=5)
    
    return results


if __name__ == "__main__":
    main()
