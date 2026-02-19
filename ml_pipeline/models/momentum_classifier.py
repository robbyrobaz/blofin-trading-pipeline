"""
Momentum Classifier using SVM.
Predicts momentum direction (accelerating, decelerating, neutral).
"""
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from models.common.base_model import BaseModel


class MomentumClassifier(BaseModel):
    """SVM-based momentum classifier."""
    
    def __init__(self):
        super().__init__(
            model_name="momentum_classifier",
            model_type="svm"
        )
        self.feature_names = [
            "momentum_roc", "momentum_rsi", "momentum_stoch",
            "trend_adx", "trend_cci", "volume_cmf"
        ]
        self.config = {
            "features": self.feature_names,
            "hyperparams": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "random_state": 42,
            },
            "classes": ["decelerating", "neutral", "accelerating"],
        }
        # Map numeric labels to class names
        self.label_map = {0: "decelerating", 1: "neutral", 2: "accelerating"}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train SVM momentum classifier.
        
        Args:
            X: Feature dataframe
            y: Target labels (0=decelerating, 1=neutral, 2=accelerating)
            **kwargs: Additional training parameters
            
        Returns:
            Dict containing training metrics
        """
        print(f"Training {self.model_name}...")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # DATA LEAKAGE GUARD: Temporal split (no shuffle) to prevent future data in training set.
        # Note: stratify=y is intentionally removed — it requires shuffle which leaks time-series data.
        MIN_SAMPLES = 50
        if len(X_selected) < MIN_SAMPLES:
            raise ValueError(
                f"Insufficient training data for {self.model_name}: "
                f"{len(X_selected)} samples (minimum {MIN_SAMPLES} required)"
            )
        split_idx = int(len(X_selected) * 0.8)
        self.scaler = StandardScaler()
        X_train_raw = X_selected.iloc[:split_idx] if hasattr(X_selected, 'iloc') else X_selected[:split_idx]
        y_train = y.iloc[:split_idx] if hasattr(y, 'iloc') else y[:split_idx]
        y_val = y.iloc[split_idx:] if hasattr(y, 'iloc') else y[split_idx:]
        self.scaler.fit(X_train_raw)
        X_scaled = self.scaler.transform(X_selected.iloc[:] if hasattr(X_selected, 'iloc') else X_selected)
        X_train = X_scaled[:split_idx]
        X_val = X_scaled[split_idx:]
        
        # Train SVM
        self.model = SVC(**self.config["hyperparams"])
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        # Per-class accuracy
        y_pred_train = self.model.predict(X_train)
        y_pred = self.model.predict(X_val)
        class_accuracies = {}
        for i, class_name in self.label_map.items():
            mask = y_val == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == y_val[mask]).sum() / mask.sum()
                class_accuracies[class_name] = float(class_acc)
        
        # Calculate classification metrics (macro average for multi-class)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        
        # Update metadata
        self.metadata["trained_at"] = datetime.now().isoformat()
        self.metadata["performance"] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(val_acc),  # Standardized name
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "class_accuracies": class_accuracies,
            "n_samples": len(X),
            "class_distribution": {
                self.label_map[i]: int((y == i).sum())
                for i in range(len(self.label_map))
            },
        }
        
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": val_acc,  # Standardized name for DB
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "class_accuracies": class_accuracies,
        }
        
        print(f"✓ {self.model_name} trained - Test Accuracy: {val_acc:.4f}, F1: {f1:.4f}")
        return metrics
    
    def predict(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Predict momentum state.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional prediction parameters
            
        Returns:
            Dict with state and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Format output
        if len(predictions) == 1:
            state = self.label_map[predictions[0]]
            confidence = float(max(probabilities[0]))
            
            return {
                "state": state,
                "confidence": confidence,
            }
        else:
            states = [self.label_map[p] for p in predictions]
            confidences = [float(max(p)) for p in probabilities]
            
            return {
                "states": states,
                "confidences": confidences,
            }
