"""
Price Predictor using Neural Network.
Predicts future price.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.common.base_model import BaseModel


class PriceNet(nn.Module):
    """Neural network for price prediction."""
    
    def __init__(self, input_dim: int):
        super(PriceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class PricePredictor(BaseModel):
    """Neural Network-based price predictor."""
    
    def __init__(self):
        super().__init__(
            model_name="price_predictor",
            model_type="neural_net"
        )
        self.feature_names = [
            "momentum_roc", "momentum_rsi", "trend_ema_diff",
            "trend_macd", "volume_obv", "volume_mfi",
            "close", "high", "low", "volume"
        ]
        self.config = {
            "features": self.feature_names,
            "hyperparams": {
                "hidden_layers": [128, 64, 32],
                "dropout": 0.2,
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32,
            },
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train Neural Network price predictor.
        
        Args:
            X: Feature dataframe
            y: Target prices
            **kwargs: Additional training parameters
            
        Returns:
            Dict containing training metrics
        """
        print(f"Training {self.model_name} on {self.device}...")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val.values).reshape(-1, 1).to(self.device)
        
        # Create model
        input_dim = X_train.shape[1]
        self.model = PriceNet(input_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["hyperparams"]["learning_rate"]
        )
        
        # Training loop
        epochs = self.config["hyperparams"]["epochs"]
        batch_size = self.config["hyperparams"]["batch_size"]
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            train_losses.append(epoch_loss / (len(X_train) / batch_size))
            val_losses.append(val_loss.item())
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
        
        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_t).cpu().numpy()
            val_pred = self.model(X_val_t).cpu().numpy()
        
        train_mae = np.mean(np.abs(y_train.values.reshape(-1, 1) - train_pred))
        val_mae = np.mean(np.abs(y_val.values.reshape(-1, 1) - val_pred))
        train_rmse = np.sqrt(np.mean((y_train.values.reshape(-1, 1) - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val.values.reshape(-1, 1) - val_pred) ** 2))
        
        # Calculate R² score for consistency
        from sklearn.metrics import r2_score
        train_r2 = r2_score(y_train.values, train_pred.flatten())
        val_r2 = r2_score(y_val.values, val_pred.flatten())
        
        # Update metadata
        self.metadata["trained_at"] = datetime.now().isoformat()
        self.metadata["performance"] = {
            "train_accuracy": float(train_r2),  # R² as accuracy for regression
            "test_accuracy": float(val_r2),  # Standardized name
            "train_mae": float(train_mae),
            "test_mae": float(val_mae),
            "train_rmse": float(train_rmse),
            "test_rmse": float(val_rmse),
            "best_val_loss": float(best_val_loss),
            "n_samples": len(X),
        }
        
        metrics = {
            "train_accuracy": train_r2,  # R² as accuracy for regression
            "test_accuracy": val_r2,  # Standardized name for DB
            "train_mae": train_mae,
            "test_mae": val_mae,
            "train_rmse": train_rmse,
            "test_rmse": val_rmse,
        }
        
        print(f"✓ {self.model_name} trained - Test R²: {val_r2:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
        return metrics
    
    def predict(self, X: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Predict future price.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional prediction parameters
            
        Returns:
            Dict with predicted price and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Select features
        X_selected = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        
        # Normalize
        X_scaled = self.scaler.transform(X_selected)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Calculate confidence (based on recent performance)
        val_mae = self.metadata["performance"].get("test_mae", 0)
        val_rmse = self.metadata["performance"].get("test_rmse", 1)
        
        # Confidence inversely related to error
        confidence = max(0.5, min(0.95, 1.0 - (val_mae / (val_rmse * 2))))
        
        # Format output
        if len(predictions) == 1:
            return {
                "predicted_price": float(predictions[0][0]),
                "confidence": float(confidence),
            }
        else:
            return {
                "predicted_prices": predictions.flatten().tolist(),
                "confidence": float(confidence),
            }
