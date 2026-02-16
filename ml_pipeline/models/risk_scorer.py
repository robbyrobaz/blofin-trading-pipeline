"""
Risk Scorer using Random Forest.
Predicts risk level 0-100.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.common.base_model import BaseModel


class RiskScorer(BaseModel):
    """Random Forest-based risk scorer."""
    
    def __init__(self):
        super().__init__(
            model_name="risk_scorer",
            model_type="random_forest"
        )
        self.feature_names = [
            "volatility_std", "volatility_parkinson", "atr_14",
            "max_drawdown", "drawdown_duration", "sharpe_ratio",
            "volume_volatility", "price_range"
        ]
        self.config = {
            "features": self.feature_names,
            "hyperparams": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "risk_scale": [0, 100],
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train Random Forest risk scorer.
        
        Args:
            X: Feature dataframe
            y: Target risk scores (0-100)
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
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            **self.config["hyperparams"],
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Predictions for additional metrics
        y_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
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
            "feature_importance": feature_importance,
        }
        
        print(f"✓ {self.model_name} trained - Test R²: {val_score:.4f}, MAE: {mae:.2f}")
        return metrics
    
    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """
        Predict risk score.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional prediction parameters
            
        Returns:
            Risk score (0-100) or array of scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict
        risk_scores = self.model.predict(X_scaled)
        
        # Clip to valid range
        risk_scores = np.clip(risk_scores, 0, 100)
        
        # Return single value or array
        if isinstance(risk_scores, np.ndarray) and len(risk_scores) == 1:
            return int(risk_scores[0])
        else:
            return risk_scores.astype(int).tolist()
