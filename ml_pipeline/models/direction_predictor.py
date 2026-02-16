"""
Direction Predictor using XGBoost.
Predicts UP/DOWN for next 5 candles.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from models.common.base_model import BaseModel


class DirectionPredictor(BaseModel):
    """XGBoost-based direction predictor."""
    
    def __init__(self):
        super().__init__(
            model_name="direction_predictor",
            model_type="xgboost"
        )
        self.feature_names = [
            "close", "returns", "rsi_14", "macd", "macd_signal",
            "macd_hist", "volume_sma", "volume_ratio"
        ]
        self.config = {
            "features": self.feature_names,
            "hyperparams": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
            "lookahead_candles": 5,
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost direction predictor.
        
        Args:
            X: Feature dataframe
            y: Target labels (0=DOWN, 1=UP)
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
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            **self.config["hyperparams"],
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        # Calculate predictions for additional metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Calculate classification metrics
        f1 = f1_score(y_val, y_pred_val, average='binary', zero_division=0)
        precision = precision_score(y_val, y_pred_val, average='binary', zero_division=0)
        recall = recall_score(y_val, y_pred_val, average='binary', zero_division=0)
        
        # Get feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist()
        ))
        
        # Update metadata
        self.metadata["trained_at"] = datetime.now().isoformat()
        self.metadata["performance"] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(val_acc),  # Standardized name
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "n_samples": len(X),
        }
        self.metadata["feature_importance"] = feature_importance
        
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": val_acc,  # Standardized name for DB
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "feature_importance": feature_importance,
        }
        
        print(f"✓ {self.model_name} trained - Test Accuracy: {val_acc:.4f}, F1: {f1:.4f}")
        return metrics
    
    def predict(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Predict direction for next candles.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional prediction parameters
            
        Returns:
            Dict with direction and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict probabilities
        proba = self.model.predict_proba(X_scaled)
        
        # Get prediction (0=DOWN, 1=UP)
        prediction = self.model.predict(X_scaled)
        
        # Format output
        if isinstance(prediction, np.ndarray) and len(prediction) == 1:
            direction = "UP" if prediction[0] == 1 else "DOWN"
            confidence = float(max(proba[0]))
        else:
            # Multiple predictions
            directions = ["UP" if p == 1 else "DOWN" for p in prediction]
            confidences = [float(max(p)) for p in proba]
            return {
                "directions": directions,
                "confidences": confidences,
            }
        
        return {
            "direction": direction,
            "confidence": confidence,
        }
