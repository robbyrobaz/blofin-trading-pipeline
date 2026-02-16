"""
Model loading and ensemble prediction utilities.
"""
import os
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np
from models.common.base_model import BaseModel


def load_model(model_name: str, base_dir: str = "models") -> BaseModel:
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model to load (e.g., "direction_predictor")
        base_dir: Base directory containing model folders
        
    Returns:
        Loaded BaseModel instance
    """
    # Map model names to their implementations
    model_registry = {
        "direction_predictor": "ml_pipeline.models.direction_predictor.DirectionPredictor",
        "risk_scorer": "ml_pipeline.models.risk_scorer.RiskScorer",
        "price_predictor": "ml_pipeline.models.price_predictor.PricePredictor",
        "momentum_classifier": "ml_pipeline.models.momentum_classifier.MomentumClassifier",
        "volatility_regressor": "ml_pipeline.models.volatility_regressor.VolatilityRegressor",
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Dynamically import the model class
    module_path, class_name = model_registry[model_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)
    
    # Instantiate and load
    model = model_class()
    model_dir = os.path.join(base_dir, f"model_{model_name}")
    model.load(model_dir)
    
    return model


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models.
    Supports weighted averaging, voting, and stacking.
    """
    
    def __init__(self, ensemble_type: str = "weighted_avg"):
        """
        Initialize ensemble predictor.
        
        Args:
            ensemble_type: Type of ensemble ("weighted_avg", "voting", "stacking")
        """
        self.ensemble_type = ensemble_type
        self.models = []
        self.weights = []
        self.config = {
            "ensemble_type": ensemble_type,
            "models": [],
            "weights": [],
        }
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: BaseModel instance
            weight: Weight for this model (used in weighted_avg)
        """
        self.models.append(model)
        self.weights.append(weight)
        self.config["models"].append(model.model_name)
        self.config["weights"].append(weight)
    
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make ensemble prediction.
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Ensemble prediction
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X, **kwargs)
            predictions.append(pred)
        
        if self.ensemble_type == "weighted_avg":
            return self._weighted_average(predictions)
        elif self.ensemble_type == "voting":
            return self._majority_voting(predictions)
        elif self.ensemble_type == "stacking":
            return self._stacking(predictions)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _weighted_average(self, predictions: List[Any]) -> Any:
        """
        Compute weighted average of predictions.
        
        Args:
            predictions: List of predictions from each model
            
        Returns:
            Weighted average prediction
        """
        # Normalize weights
        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]
        
        # Handle different prediction formats
        if isinstance(predictions[0], dict):
            # Assume dict has numeric values we can average
            result = {}
            for key in predictions[0].keys():
                if isinstance(predictions[0][key], (int, float)):
                    weighted_sum = sum(
                        pred.get(key, 0) * w 
                        for pred, w in zip(predictions, normalized_weights)
                    )
                    result[key] = weighted_sum
                else:
                    # For non-numeric values, use first model's prediction
                    result[key] = predictions[0][key]
            return result
        
        elif isinstance(predictions[0], (int, float)):
            # Simple numeric predictions
            return sum(p * w for p, w in zip(predictions, normalized_weights))
        
        elif isinstance(predictions[0], np.ndarray):
            # Array predictions
            weighted_sum = np.zeros_like(predictions[0], dtype=float)
            for pred, weight in zip(predictions, normalized_weights):
                weighted_sum += pred * weight
            return weighted_sum
        
        else:
            # Fallback: return first prediction
            return predictions[0]
    
    def _majority_voting(self, predictions: List[Any]) -> Any:
        """
        Majority voting for classification tasks.
        
        Args:
            predictions: List of predictions from each model
            
        Returns:
            Most common prediction
        """
        # Handle dict predictions (e.g., {"direction": "UP"})
        if isinstance(predictions[0], dict):
            # Vote on first key's value
            first_key = list(predictions[0].keys())[0]
            values = [pred[first_key] for pred in predictions]
            most_common = max(set(values), key=values.count)
            
            # Return dict with voted value
            result = predictions[0].copy()
            result[first_key] = most_common
            
            # Average confidence if available
            if "confidence" in result:
                confidences = [pred.get("confidence", 0) for pred in predictions]
                result["confidence"] = np.mean(confidences)
            
            return result
        
        else:
            # Simple voting
            return max(set(predictions), key=predictions.count)
    
    def _stacking(self, predictions: List[Any]) -> Any:
        """
        Stacking ensemble (meta-model on base predictions).
        For now, falls back to weighted average.
        
        Args:
            predictions: List of predictions from each model
            
        Returns:
            Stacked prediction
        """
        # TODO: Implement proper stacking with meta-model
        # For now, use weighted average
        return self._weighted_average(predictions)
    
    def save_config(self, config_path: str) -> None:
        """
        Save ensemble configuration.
        
        Args:
            config_path: Path to save config file
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Ensemble config saved to {config_path}")
    
    def load_config(self, config_path: str, base_dir: str = "models") -> None:
        """
        Load ensemble configuration and models.
        
        Args:
            config_path: Path to config file
            base_dir: Base directory containing model folders
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.ensemble_type = self.config["ensemble_type"]
        self.models = []
        self.weights = []
        
        for model_name, weight in zip(self.config["models"], self.config["weights"]):
            model = load_model(model_name, base_dir)
            self.add_model(model, weight)
        
        print(f"✓ Ensemble loaded from {config_path}")
