"""
Volatility Regressor using Gradient Boosting.
Predicts future volatility.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.common.base_model import BaseModel


class VolatilityRegressor(BaseModel):
    """Gradient Boosting-based volatility regressor."""
    
    def __init__(self):
        super().__init__(
            model_name="volatility_regressor",
            model_type="gradient_boosting"
        )
        self.feature_names = [
            "volatility_std", "volatility_parkinson", "atr_14",
            "volume_std", "volume_ratio", "returns_abs",
            "high_low_range", "close_variance"
        ]
        self.config = {
            "features": self.feature_names,
            "hyperparams": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "subsample": 0.8,
                "random_state": 42,
            },
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train Gradient Boosting volatility regressor.
        
        Args:
            X: Feature dataframe
            y: Target volatility values
            **kwargs: Additional training parameters
            
        Returns:
            Dict containing training metrics
        """
        print(f"Training {self.model_name}...")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Gradient Boosting
        self.model = GradientBoostingRegressor(**self.config["hyperparams"])
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Predictions for additional metrics
        y_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-10))) * 100
        
        # Get feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist()
        ))
        
        # Update metadata
        self.metadata["trained_at"] = datetime.now().isoformat()
        self.metadata["performance"] = {
            "train_accuracy": float(train_score),  # R² as accuracy for regression
            "test_accuracy": float(val_score),  # Standardized name
            "train_r2": float(train_score),
            "test_r2": float(val_score),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_mape": float(mape),
            "n_samples": len(X),
        }
        self.metadata["feature_importance"] = feature_importance
        
        metrics = {
            "train_accuracy": train_score,  # R² as accuracy for regression
            "test_accuracy": val_score,  # Standardized name for DB
            "train_r2": train_score,
            "test_r2": val_score,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "feature_importance": feature_importance,
        }
        
        print(f"✓ {self.model_name} trained - Test R²: {val_score:.4f}, MAE: {mae:.6f}")
        return metrics
    
    def predict(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Predict future volatility.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional prediction parameters
            
        Returns:
            Dict with predicted volatility and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict
        volatility = self.model.predict(X_scaled)
        
        # Calculate confidence based on validation performance
        val_mape = self.metadata["performance"].get("test_mape", 100)
        confidence = max(0.5, min(0.95, 1.0 - (val_mape / 100)))
        
        # Format output
        if isinstance(volatility, np.ndarray) and len(volatility) == 1:
            return {
                "predicted_vol": float(volatility[0]),
                "confidence": float(confidence),
            }
        else:
            return {
                "predicted_vols": volatility.tolist(),
                "confidence": float(confidence),
            }
