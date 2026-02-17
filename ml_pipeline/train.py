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
    
    def load_real_data(self, symbols: list = None, lookback_bars: int = 2000) -> pd.DataFrame:
        """
        Load REAL training data from database using FeatureManager.
        
        Args:
            symbols: List of symbols to load (default: BTC, ETH, SOL)
            lookback_bars: Number of candles to load per symbol
            
        Returns:
            DataFrame with real features and generated targets
        """
        from features.feature_manager import FeatureManager
        
        if symbols is None:
            symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
        
        print(f"Loading REAL data for {len(symbols)} symbols ({lookback_bars} bars each)...")
        
        fm = FeatureManager()
        all_features = []
        
        for symbol in symbols:
            print(f"  Loading {symbol}...")
            
            try:
                # Load features from real tick data
                features = fm.get_features(
                    symbol=symbol,
                    timeframe='1m',
                    lookback_bars=lookback_bars,
                    fill_nan=True
                )
                
                # Add symbol column for tracking
                features['symbol'] = symbol
                
                all_features.append(features)
                print(f"    ✓ {len(features)} samples loaded")
                
            except Exception as e:
                print(f"    ✗ Failed to load {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("Failed to load any real data. Check database and symbols.")
        
        # Combine all symbols
        df = pd.concat(all_features, ignore_index=True)
        
        print(f"\n✓ Loaded {len(df)} total samples from {len(all_features)} symbols")
        
        # Generate targets from real price data
        print("  Generating targets from real price movement...")
        
        # Direction: will price go up in next 5 candles?
        df['target_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
        
        # Risk: volatility over next 20 candles
        df['target_risk'] = df['close'].rolling(20).std().fillna(0) * 100
        
        # Price: actual price 5 candles ahead
        df['target_price'] = df['close'].shift(-5)
        
        # Momentum: categorize price change rate
        price_change = df['close'].pct_change().fillna(0)
        df['target_momentum'] = pd.cut(
            price_change,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        # Volatility: standard deviation over next 10 candles
        df['target_volatility'] = df['close'].rolling(10).std().fillna(0) / 1000
        
        # Drop rows with NaN targets (from forward shifts)
        initial_count = len(df)
        df = df.dropna()
        dropped = initial_count - len(df)
        
        print(f"  ✓ Targets generated ({dropped} rows dropped for forward-looking targets)")
        print(f"\n✓ Final dataset: {len(df)} clean samples ready for training")
        
        return df


def main():
    """Main entry point for training."""
    print("ML Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(base_model_dir="models")
    
    # Load REAL data from database (24.7M ticks!)
    symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'AVAX-USDT', 'LINK-USDT']
    features_df = pipeline.load_real_data(symbols=symbols, lookback_bars=2000)
    
    # Train all models
    results = pipeline.train_all_models(features_df, max_workers=5)
    
    return results


if __name__ == "__main__":
    main()
