"""
Base model interface for all ML models in the Blofin Stack.
All models must inherit from BaseModel and implement the abstract methods.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime
import pickle


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, model_name: str, model_type: str):
        """
        Initialize base model.
        
        Args:
            model_name: Unique identifier for the model (e.g., "direction_predictor")
            model_type: Type of model (e.g., "xgboost", "random_forest", "neural_net")
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.config = {}
        self.metadata = {
            "created_at": None,
            "trained_at": None,
            "performance": {},
            "feature_importance": {},
        }
        self.scaler = None
        
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional training parameters
            
        Returns:
            Dict containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions (format depends on model type)
        """
        pass
    
    def save(self, model_dir: str) -> None:
        """
        Save model, config, and metadata to disk.
        
        Args:
            model_dir: Directory to save model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler if exists
        if self.scaler is not None:
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save metadata
        self.metadata["saved_at"] = datetime.now().isoformat()
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        print(f"✓ Model saved to {model_dir}")
    
    def load(self, model_dir: str) -> None:
        """
        Load model, config, and metadata from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        # Load model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler if exists
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"✓ Model loaded from {model_dir}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (if available).
        
        Returns:
            Dict mapping feature names to importance scores
        """
        return self.metadata.get("feature_importance", {})
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from last training/validation.
        
        Returns:
            Dict containing performance metrics
        """
        return self.metadata.get("performance", {})
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update metadata field.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
