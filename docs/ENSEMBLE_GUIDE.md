# Ensemble Guide

## Overview

Ensembles combine multiple ML models to achieve better predictions than any single model.

Why ensembles?
- Reduce overfitting (multiple models learn different patterns)
- Leverage strengths of different algorithms
- Test combinations without retraining (cheap and fast)
- Flexibility (swap models in/out, adjust weights)

---

## Ensemble Directory Structure

```
models/ensembles/
├── ensemble_001.json              # Config: which models + weights
├── ensemble_001_results.json      # Validation results
├── ensemble_002.json
├── ensemble_002_results.json
└── ...
```

---

## Creating an Ensemble

### Basic Weighted Average

Simplest approach: take the average of model predictions, weighted by their performance.

```python
from models.common.predictor import EnsemblePredictor

# Create ensemble of top 3 direction predictors
ensemble = EnsemblePredictor()

ensemble.config = {
    "name": "ensemble_001",
    "description": "Weighted average of 3 direction predictors",
    "models": ["model_001", "model_002", "model_003"],
    "weights": [0.5, 0.3, 0.2],      # Weighted by backtest accuracy
    "method": "weighted_avg",
    "prediction_type": "direction"    # 'direction', 'risk', 'price', etc.
}

ensemble.save()
```

### Majority Voting

For classification (UP/DOWN), use voting.

```python
ensemble.config = {
    "name": "ensemble_002",
    "models": ["model_004", "model_005", "model_006"],
    "method": "voting",
    "voting_rule": "majority",       # Or "weighted_majority"
    "threshold": 0.5                 # Confidence needed (0.5 = 50%+)
}

ensemble.save()

# Usage:
pred = ensemble.predict(...)
# Returns: "UP" (3 models agree) or "ABSTAIN" (tied/uncertain)
```

### Stacking

Meta-model learns from individual model outputs.

```python
from ml_pipeline.train import StackingEnsemble

# Train meta-model on predictions from base models
meta_ensemble = StackingEnsemble(
    base_models=["model_001", "model_002", "model_003"],
    meta_model_type="logistic_regression"  # or 'xgboost', 'neural_net'
)

meta_ensemble.train(X=training_features, y=training_labels)
meta_ensemble.save("ensemble_003")

# Usage:
pred = meta_ensemble.predict(...)
# Meta-model learned optimal combination
```

---

## Ensemble Configurations

### Direction Prediction Ensemble

Predicts: UP or DOWN in next 5 candles

```json
{
  "name": "ensemble_direction_001",
  "description": "Ensemble of direction predictors",
  "models": [
    {
      "name": "model_001",
      "type": "xgboost",
      "weight": 0.5,
      "recent_accuracy": 0.568
    },
    {
      "name": "model_002",
      "type": "random_forest",
      "weight": 0.3,
      "recent_accuracy": 0.541
    },
    {
      "name": "model_003",
      "type": "neural_net",
      "weight": 0.2,
      "recent_accuracy": 0.528
    }
  ],
  "method": "weighted_avg",
  "voting_threshold": 0.5,
  "confidence_required": 0.60,
  "status": "active",
  "created_ts": "2026-02-15",
  "backtest_accuracy": 0.575,
  "backtest_f1": 0.571
}
```

### Risk Scoring Ensemble

Predicts: Risk level (0-100)

```json
{
  "name": "ensemble_risk_001",
  "description": "Risk scoring ensemble",
  "models": [
    {"name": "model_004", "type": "random_forest", "weight": 0.4},
    {"name": "model_005", "type": "svm", "weight": 0.6}
  ],
  "method": "weighted_avg",
  "output_scaling": "0-100",
  "status": "active"
}
```

### Price Prediction Ensemble

Predicts: Future price

```json
{
  "name": "ensemble_price_001",
  "description": "Price prediction ensemble",
  "models": [
    {"name": "model_006", "type": "neural_net", "weight": 0.5},
    {"name": "model_007", "type": "gradient_boosting", "weight": 0.5}
  ],
  "method": "weighted_avg",
  "output_type": "continuous",
  "metric": "rmse",
  "status": "active"
}
```

---

## Building Ensembles

### Option 1: Weighted Average (Recommended for Speed)

```python
from models.common.predictor import load_model
import numpy as np

def create_weighted_ensemble(model_names, weights, prediction_type='direction'):
    """
    Create weighted average ensemble.
    
    model_names: ['model_001', 'model_002', 'model_003']
    weights: [0.5, 0.3, 0.2]  (should sum to 1.0)
    """
    
    config = {
        "name": f"ensemble_{len(model_names)}_weighted",
        "models": model_names,
        "weights": weights,
        "method": "weighted_avg",
        "prediction_type": prediction_type
    }
    
    def predict(features):
        predictions = []
        for model_name in model_names:
            model = load_model(model_name)
            pred = model.predict(features)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=weights, axis=0)
        return ensemble_pred
    
    config['predict_fn'] = predict
    return config
```

### Option 2: Majority Voting (Good for Classification)

```python
def create_voting_ensemble(model_names, voting_rule='majority'):
    """
    Create voting ensemble for classification.
    """
    
    config = {
        "name": f"ensemble_{len(model_names)}_voting",
        "models": model_names,
        "method": "voting",
        "voting_rule": voting_rule
    }
    
    def predict(features):
        votes = []
        for model_name in model_names:
            model = load_model(model_name)
            pred = model.predict(features)  # Returns 'UP' or 'DOWN'
            votes.append(pred)
        
        # Count votes
        if voting_rule == 'majority':
            if votes.count('UP') > votes.count('DOWN'):
                return 'UP'
            elif votes.count('DOWN') > votes.count('UP'):
                return 'DOWN'
            else:
                return 'ABSTAIN'  # Tie
        
        return votes[0]  # First vote wins if not majority rule
    
    config['predict_fn'] = predict
    return config
```

### Option 3: Stacking (Most Powerful, Slower)

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def create_stacking_ensemble(model_names, meta_model_type='logistic_regression'):
    """
    Create stacking ensemble with meta-model.
    Requires training on historical data first.
    """
    
    # Train meta-model on base model outputs
    def train_meta_model(X_features, y_labels):
        # Get predictions from each base model
        meta_X = []
        for model_name in model_names:
            model = load_model(model_name)
            preds = model.predict(X_features)
            meta_X.append(preds)
        
        meta_X = np.column_stack(meta_X)
        
        # Train meta-model
        if meta_model_type == 'logistic_regression':
            meta_model = LogisticRegression()
        elif meta_model_type == 'xgboost':
            from xgboost import XGBClassifier
            meta_model = XGBClassifier()
        
        meta_model.fit(meta_X, y_labels)
        return meta_model
    
    def predict(features):
        meta_features = []
        for model_name in model_names:
            model = load_model(model_name)
            pred = model.predict(features)
            meta_features.append(pred)
        
        meta_features = np.array(meta_features).reshape(1, -1)
        return meta_model.predict(meta_features)
    
    config = {
        "name": f"ensemble_{len(model_names)}_stacking",
        "models": model_names,
        "method": "stacking",
        "meta_model_type": meta_model_type,
        "predict_fn": predict,
        "train_fn": train_meta_model
    }
    
    return config
```

---

## Validating Ensembles

### Backtest Ensemble

```python
from backtester.backtest_engine import BacktestEngine
from models.common.predictor import EnsemblePredictor

def backtest_ensemble(ensemble_name, symbols=['BTC-USDT'], days_back=7):
    """
    Backtest ensemble on historical data.
    """
    
    ensemble = EnsemblePredictor.load(ensemble_name)
    
    results = {
        'symbol': symbol,
        'timeframe': '1m',
        'total_predictions': 0,
        'correct': 0,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    
    for symbol in symbols:
        engine = BacktestEngine(symbol=symbol, days_back=days_back)
        
        correct = 0
        total = 0
        
        for i in range(len(engine.data) - 5):
            # Get features
            current_candle = engine.data[i]
            next_candles = engine.data[i+1:i+5]
            
            # Predict
            pred = ensemble.predict(current_candle)
            
            # Actual
            future_price = next_candles[-1]['close']
            actual = 'UP' if future_price > current_candle['close'] else 'DOWN'
            
            if pred == actual:
                correct += 1
            total += 1
        
        accuracy = correct / total
        results['accuracy'] = accuracy
    
    return results
```

### Compare Ensemble vs Individual Models

```python
def compare_ensemble_vs_individuals(ensemble_name):
    """
    Compare ensemble accuracy to each base model.
    """
    
    ensemble_results = backtest_ensemble(ensemble_name)
    ensemble_accuracy = ensemble_results['accuracy']
    
    ensemble_config = db.get_ensemble_config(ensemble_name)
    model_names = ensemble_config['models']
    
    print(f"\n{ensemble_name} Comparison:")
    print(f"Ensemble accuracy: {ensemble_accuracy:.2%}")
    print("\nBase models:")
    
    for model_name in model_names:
        model_results = db.get_last_backtest(model_name)
        model_accuracy = model_results['accuracy']
        diff = (ensemble_accuracy - model_accuracy) * 100
        symbol = "✓" if diff > 0 else "✗"
        print(f"  {symbol} {model_name}: {model_accuracy:.2%} ({diff:+.1f}%)")
    
    # If ensemble wins, keep it
    if ensemble_accuracy > max(m['accuracy'] for m in model_results.values()):
        db.set_ensemble_status(ensemble_name, 'active')
        return True
    else:
        db.set_ensemble_status(ensemble_name, 'archived')
        return False
```

---

## Ensemble Management

### Keep Top Ensembles

```python
def select_top_ensembles(count=3):
    """
    Keep top N ensembles by F1 score.
    """
    all_ensembles = db.get_all_ensembles()
    
    # Sort by F1
    sorted_ensembles = sorted(
        all_ensembles,
        key=lambda e: e['backtest_f1'],
        reverse=True
    )
    
    # Keep top N
    for i, ensemble in enumerate(sorted_ensembles):
        if i < count:
            db.set_ensemble_status(ensemble['name'], 'active')
        else:
            db.set_ensemble_status(ensemble['name'], 'archived')
    
    return sorted_ensembles[:count]
```

### Generate New Ensembles

When new models are trained, create new ensemble combinations:

```python
def generate_ensemble_candidates(available_models, max_ensemble_size=4):
    """
    Generate candidate ensembles from available models.
    """
    from itertools import combinations
    
    ensembles_to_test = []
    
    # Test all combinations up to max size
    for size in range(2, max_ensemble_size + 1):
        for model_combo in combinations(available_models, size):
            # Weight by individual accuracy
            weights = [
                model['accuracy'] 
                for model in model_combo
            ]
            
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
            
            ensemble_config = {
                'models': [m['name'] for m in model_combo],
                'weights': weights,
                'method': 'weighted_avg'
            }
            
            ensembles_to_test.append(ensemble_config)
    
    return ensembles_to_test
```

### Test and Rank

```python
def test_ensemble_candidates(candidates, symbols=['BTC-USDT']):
    """
    Backtest all candidate ensembles.
    """
    
    results = []
    
    for i, config in enumerate(candidates):
        ensemble = EnsemblePredictor(config)
        
        backtest_results = backtest_ensemble(ensemble, symbols)
        
        results.append({
            'ensemble_id': f"ensemble_{i}",
            'models': config['models'],
            'weights': config['weights'],
            'accuracy': backtest_results['accuracy'],
            'f1': backtest_results['f1']
        })
    
    # Sort by F1
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    return results
```

---

## Integration with Strategies

Ensembles can enhance strategy decisions:

```python
from strategies.base_strategy import BaseStrategy, Signal
from models.common.predictor import EnsemblePredictor

class EnsembleStrategy(BaseStrategy):
    name = "ensemble_driven_trading"
    
    def __init__(self):
        self.direction_ensemble = EnsemblePredictor.load('ensemble_direction_001')
        self.risk_ensemble = EnsemblePredictor.load('ensemble_risk_001')
    
    def detect(self, symbol, price, volume, ts_ms, df, fm):
        features = fm.get_features(symbol, '1m', [
            'close', 'rsi_14', 'macd_histogram', 
            'volume_sma_20', 'atr_14'
        ])
        
        # Ensemble predictions
        direction = self.direction_ensemble.predict(features)
        risk = self.risk_ensemble.predict(features)
        
        # Only trade if ensemble confident AND risk acceptable
        if direction in ['UP', 'DOWN'] and risk < 50:
            confidence = direction_pred['confidence']
            
            return Signal(
                symbol=symbol,
                signal=direction,
                strategy=self.name,
                confidence=confidence,
                details={
                    'ensemble_direction': direction,
                    'ensemble_risk': risk
                }
            )
        
        return None
```

---

## Best Practices

1. **Diverse models** — Combine different algorithm types (XGBoost + Neural Net + Random Forest)
2. **Low correlation** — Choose models with different weaknesses
3. **Weighted by performance** — Better models get higher weights
4. **Test combinations** — Not all combinations improve accuracy
5. **Keep it simple** — 3-4 models usually better than 10
6. **Retrain together** — When base models retrain, revalidate ensemble
7. **Monitor drift** — Check ensemble accuracy on live data regularly

---

## Summary

Ensembles are how you go from "decent" to "great". By combining multiple imperfect models, you get something more robust and reliable.

Start with 2-3 top models, test combinations, keep what works.
