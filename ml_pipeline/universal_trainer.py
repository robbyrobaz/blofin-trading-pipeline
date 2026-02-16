"""Universal ML trainer - works with any feature set."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import pickle
import json


class UniversalMLTrainer:
    """Train ML models using available features."""
    
    def __init__(self, base_model_dir='models'):
        self.base_model_dir = base_model_dir
        Path(base_model_dir).mkdir(exist_ok=True)
    
    def train_models(self, df: pd.DataFrame) -> int:
        """
        Train 5 ML models using available features.
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Number of models successfully trained
        """
        successful = 0
        
        # Select best available features (top predictive)
        feature_cols = self._select_features(df)
        
        if len(feature_cols) < 10:
            print(f"✗ Not enough features: {len(feature_cols)}")
            return 0
        
        # Prepare X, y
        X = df[feature_cols].fillna(0)
        X = StandardScaler().fit_transform(X)
        
        if 'target_direction' in df.columns:
            y_direction = df['target_direction'].fillna(0)
            successful += self._train_direction_model(X, y_direction, feature_cols)
        
        if 'target_volatility' in df.columns:
            y_volatility = df['target_volatility'].fillna(0)
            successful += self._train_volatility_model(X, y_volatility, feature_cols)
        
        if 'target_price' in df.columns:
            y_price = df['target_price'].fillna(df['close'].mean())
            successful += self._train_price_model(X, y_price, feature_cols)
        
        if 'target_momentum' in df.columns:
            y_momentum = df['target_momentum'].fillna(0)
            successful += self._train_momentum_model(X, y_momentum, feature_cols)
        
        # Generic classifier
        if 'target_direction' in df.columns:
            y = (df['close'].pct_change() > 0).astype(int).fillna(0)
            successful += self._train_generic_model(X, y, 'risk_scorer', feature_cols)
        
        return successful
    
    def _select_features(self, df: pd.DataFrame, n_features: int = 30):
        """Select best features for training."""
        # Exclude non-numeric and target columns
        exclude = {'timestamp', 'open', 'high', 'low', 'close', 'tick_count', 'volume',
                   'target_direction', 'target_price', 'target_momentum', 'target_volatility', 'momentum'}
        
        features = [col for col in df.columns 
                   if col not in exclude and df[col].dtype in ['float64', 'int64']]
        
        # Use top features (by variance)
        if len(features) > n_features:
            variances = df[features].var()
            features = variances.nlargest(n_features).index.tolist()
        
        return features
    
    def _train_direction_model(self, X, y, feature_cols):
        """Train direction classifier."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            self._save_model('direction_predictor', model, acc, {'accuracy': acc})
            print(f"  ✓ Direction: {acc:.2%}")
            return 1
        except Exception as e:
            print(f"  ✗ Direction: {e}")
            return 0
    
    def _train_volatility_model(self, X, y, feature_cols):
        """Train volatility regressor."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self._save_model('volatility_regressor', model, rmse, {'rmse': rmse})
            print(f"  ✓ Volatility: RMSE={rmse:.4f}")
            return 1
        except Exception as e:
            print(f"  ✗ Volatility: {e}")
            return 0
    
    def _train_price_model(self, X, y, feature_cols):
        """Train price predictor."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self._save_model('price_predictor', model, rmse, {'rmse': rmse})
            print(f"  ✓ Price: RMSE={rmse:.2f}")
            return 1
        except Exception as e:
            print(f"  ✗ Price: {e}")
            return 0
    
    def _train_momentum_model(self, X, y, feature_cols):
        """Train momentum classifier."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            self._save_model('momentum_classifier', model, acc, {'accuracy': acc})
            print(f"  ✓ Momentum: {acc:.2%}")
            return 1
        except Exception as e:
            print(f"  ✗ Momentum: {e}")
            return 0
    
    def _train_generic_model(self, X, y, name, feature_cols):
        """Train generic model."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            self._save_model(name, model, acc, {'accuracy': acc})
            print(f"  ✓ {name}: {acc:.2%}")
            return 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            return 0
    
    def _save_model(self, name, model, score, metrics):
        """Save trained model."""
        model_dir = Path(self.base_model_dir) / f'{name}_{score:.3f}'
        model_dir.mkdir(exist_ok=True)
        
        with open(model_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        metadata = {
            'name': name,
            'score': float(score),
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
