# ML Pipeline Guide

## Overview

The ML pipeline **continuously builds, validates, and tunes machine learning models** to predict price direction and risk.

Every 12 hours:
1. **Build** — Train 4-5 ML models in parallel on rolling 7-day data
2. **Validate** — Backtest on holdout data (not used in training)
3. **Rank** — Keep top 5 models, archive bottom performers
4. **Tune** — If accuracy drifting, retrain with new data or redesign features

---

## Directory Structure

```
models/
├── model_001/
│   ├── model.pkl              # Trained model (sklearn/xgb/torch)
│   ├── config.json            # Features, hyperparams
│   ├── metadata.json          # Performance, creation_ts, status
│   └── README.md              # What this model does
├── model_002/
├── ...
├── common/
│   ├── base_model.py          # Abstract base class
│   ├── trainer.py             # Training loop
│   └── predictor.py           # Load and predict
└── ensembles/
    ├── ensemble_001.json      # Config: which models + weights
    └── ensemble_001_results.json
```

---

## Model Types

### 1. Direction Predictor (XGBoost)

Predicts: Does price go UP or DOWN in next 1-5 minutes?

```python
from ml_pipeline.direction_predictor import DirectionPredictor

predictor = DirectionPredictor()

# Train
predictor.train(
    symbols=['BTC-USDT', 'ETH-USDT'],
    timeframe='1m',
    lookback_bars=1000,
    prediction_horizon=5  # Predict next 5 candles
)

# Predict
pred = predictor.predict(symbol='BTC-USDT', current_features=df)
# Returns: {"direction": "UP", "confidence": 0.72}

# Backtest
results = predictor.backtest(symbols=['BTC-USDT'], days_back=7)
# Returns: accuracy, precision, recall, F1
```

### 2. Risk Scorer (Random Forest)

Predicts: How risky is this trade? (0-100)

```python
from ml_pipeline.risk_scorer import RiskScorer

scorer = RiskScorer()

# Train
scorer.train(symbols=['BTC-USDT'], lookback_bars=1000)

# Predict
risk = scorer.predict(symbol='BTC-USDT', current_features=df)
# Returns: 35 (low risk) to 85 (high risk)

# Only execute trades with risk < 60
if risk < 60:
    execute_trade()
```

### 3. Price Predictor (Neural Network)

Predicts: What will the price be in 1 hour?

```python
from ml_pipeline.price_predictor import PricePredictor

predictor = PricePredictor()

# Train
predictor.train(symbols=['BTC-USDT'], timeframe='1m')

# Predict
price_pred = predictor.predict(symbol='BTC-USDT')
# Returns: {"predicted_price": 45240.50, "confidence": 0.68}

# Backtest
results = predictor.backtest(symbols=['BTC-USDT'], days_back=7)
# Returns: RMSE, MAE, R²
```

### 4. Momentum Classifier (SVM)

Predicts: Is momentum increasing or decreasing?

```python
from ml_pipeline.momentum_classifier import MomentumClassifier

classifier = MomentumClassifier()

# Train
classifier.train(symbols=['BTC-USDT'], lookback_bars=1000)

# Predict
momentum_state = classifier.predict(symbol='BTC-USDT', current_features=df)
# Returns: {"state": "accelerating", "confidence": 0.74}
```

### 5. Volatility Regressor (Gradient Boosting)

Predicts: What will volatility be in next 1-4 hours?

```python
from ml_pipeline.volatility_regressor import VolatilityRegressor

regressor = VolatilityRegressor()

# Train
regressor.train(symbols=['BTC-USDT'], timeframe='5m')

# Predict
vol = regressor.predict(symbol='BTC-USDT')
# Returns: {"predicted_volatility": 0.0145, "confidence": 0.71}
```

---

## Training Pipeline

### Step 1: Data Preparation

```python
from ml_pipeline.train import TrainingPipeline
from features.feature_manager import FeatureManager

fm = FeatureManager()

# Get training data for all symbols
training_data = {}
for symbol in ['BTC-USDT', 'ETH-USDT', 'XRP-USDT']:
    df = fm.get_features(
        symbol=symbol,
        timeframe='1m',
        feature_list=[
            'close', 'returns', 'rsi_14', 'macd_histogram',
            'volume_sma_20', 'atr_14', 'vwap', 'adx_14'
        ],
        lookback_bars=2000  # Last 1.4 days of 1m data
    )
    training_data[symbol] = df

print(f"Loaded {len(training_data)} symbols")
for sym, df in training_data.items():
    print(f"  {sym}: {len(df)} rows, {len(df.columns)} features")
```

### Step 2: Build Models

```python
from ml_pipeline.train import TrainingPipeline
import concurrent.futures

pipeline = TrainingPipeline()

# Define models to train
models_to_train = [
    {
        'type': 'xgboost',
        'name': 'direction_predictor_001',
        'config': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    },
    {
        'type': 'random_forest',
        'name': 'risk_scorer_001',
        'config': {
            'n_estimators': 100,
            'max_depth': 10
        }
    },
    # ... more models
]

# Train in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {}
    for model_def in models_to_train:
        future = executor.submit(
            pipeline.train_model,
            model_def,
            training_data
        )
        futures[model_def['name']] = future
    
    # Collect results
    trained_models = {}
    for name, future in futures.items():
        trained_models[name] = future.result()
        print(f"✓ {name} trained")

# Save all models
for name, model in trained_models.items():
    pipeline.save_model(name, model)
```

### Step 3: Validation (Backtest)

```python
from ml_pipeline.validate import ValidationPipeline

validator = ValidationPipeline()

validation_results = {}
for model_name, model in trained_models.items():
    results = validator.backtest_model(
        model,
        symbols=['BTC-USDT', 'ETH-USDT'],
        days_back=7
    )
    validation_results[model_name] = results
    
    print(f"{model_name}:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Precision: {results['precision']:.2%}")
    print(f"  F1: {results['f1']:.2f}")
```

### Step 4: Ranking

```python
from orchestration.ranker import ModelRanker

ranker = ModelRanker()

# Get top 5 models
top_models = ranker.get_top_models(
    results=validation_results,
    count=5
)

print("Top 5 Models:")
for i, (name, score) in enumerate(top_models, 1):
    print(f"{i}. {name}: {score:.1f}")

# Archive bottom 2 (if > 7 models)
to_remove = ranker.get_to_replace(count=2)
for model_name in to_remove:
    db.update_model_status(model_name, 'archived')
    print(f"Archived: {model_name}")
```

---

## Tuning & Drift Detection

### Monitor Live Accuracy

```python
from ml_pipeline.validate import ValidationPipeline

validator = ValidationPipeline()

# Every 12 hours, check if model accuracy is drifting
def check_model_drift():
    for model_name in active_models:
        model = load_model(model_name)
        
        # Validate on last 24h of data
        live_results = validator.validate_on_live_data(
            model,
            symbol='BTC-USDT',
            hours_back=24
        )
        
        # Compare to last backtest
        last_backtest = db.get_last_backtest(model_name)
        drift = last_backtest['accuracy'] - live_results['accuracy']
        
        if drift > 0.05:  # 5%+ drift
            print(f"⚠️ {model_name} accuracy drifting: {drift:.2%}")
            db.flag_for_retraining(model_name)
```

### Retrain Drifting Models

```python
def retrain_drifting_models():
    flagged = db.get_flagged_for_retraining()
    
    for model_name in flagged:
        old_model = load_model(model_name)
        new_version = model_name.replace('_001', '_002')
        
        # Retrain with latest data
        retrained = trainer.train_model(
            model_type=old_model.type,
            config=old_model.config,
            training_data=get_latest_data()
        )
        
        # Validate
        new_results = validator.backtest_model(retrained, ...)
        old_results = db.get_validation_results(model_name)
        
        if new_results['f1'] > old_results['f1']:
            # Better! Keep it
            save_model(new_version, retrained)
            db.update_model_status(model_name, 'superseded')
            db.update_model_status(new_version, 'active')
            print(f"✓ Retrained {model_name} → {new_version}")
        else:
            # Worse, discard
            print(f"✗ Retraining didn't help, keeping {model_name}")
```

---

## Building Ensembles

Ensembles combine multiple models for better predictions.

### Create an Ensemble

```python
from models.common.predictor import EnsemblePredictor

ensemble = EnsemblePredictor(
    models=['model_001', 'model_002', 'model_003'],
    weights=[0.5, 0.3, 0.2],  # Weighted average
    method='weighted_avg'      # or 'voting', 'stacking'
)

# Save config
ensemble.save_config('ensemble_001')

# Predict
pred = ensemble.predict(symbol='BTC-USDT', df=features)
# Returns: blended prediction from 3 models
```

### Ensemble Methods

| Method | Best For | Example |
|--------|----------|---------|
| **Weighted Avg** | Continuous predictions (price, volatility) | `(0.5*m1 + 0.3*m2 + 0.2*m3)` |
| **Majority Vote** | Classification (direction, risk) | `3 out of 5 vote UP` |
| **Stacking** | Complex combinations | `Meta-model learns from outputs` |

### Test Ensemble

```python
validator = ValidationPipeline()

results = validator.backtest_ensemble(
    'ensemble_001',
    symbols=['BTC-USDT', 'ETH-USDT'],
    days_back=7
)

print(f"Ensemble accuracy: {results['accuracy']:.2%}")
print(f"Ensemble F1: {results['f1']:.2f}")

# Compare to individual models
for model in ['model_001', 'model_002', 'model_003']:
    individual = db.get_validation_results(model)
    print(f"{model}: {individual['accuracy']:.2%}")

# If ensemble > individuals → use it!
if results['f1'] > individual['f1']:
    db.set_ensemble_active('ensemble_001')
```

---

## Integration with Strategies

ML models can inform strategy decisions:

```python
from strategies.base_strategy import BaseStrategy, Signal
from models.common.predictor import load_model

class MLEnhancedStrategy(BaseStrategy):
    name = "ml_enhanced_momentum"
    
    def __init__(self):
        self.direction_model = load_model('model_001')
        self.risk_model = load_model('model_002')
    
    def detect(self, symbol, price, volume, ts_ms, df, fm):
        # Get technical signal
        rsi = fm.get_features(symbol, '5m', ['rsi_14'])
        if rsi.iloc[-1]['rsi_14'] > 70:  # Overbought
            signal_type = 'SHORT'
        elif rsi.iloc[-1]['rsi_14'] < 30:  # Oversold
            signal_type = 'LONG'
        else:
            return None
        
        # Validate with ML
        features = fm.get_features(symbol, '1m', [
            'close', 'returns', 'volume_sma_20', 'atr_14'
        ])
        
        direction_pred = self.direction_model.predict(features.iloc[-1:])
        risk = self.risk_model.predict(features.iloc[-1:])
        
        # Only trade if ML agrees AND risk is low
        if (direction_pred['direction'] == signal_type and 
            risk < 50):
            
            return Signal(
                symbol=symbol,
                signal=signal_type,
                strategy=self.name,
                confidence=direction_pred['confidence'],
                details={
                    'technical': rsi.iloc[-1]['rsi_14'],
                    'ml_direction': direction_pred['direction'],
                    'risk_score': risk
                }
            )
        
        return None
```

---

## Model Metadata

Each model should document itself:

```json
{
  "model_001": {
    "name": "direction_predictor_001",
    "type": "xgboost",
    "description": "Predicts UP/DOWN direction for next 5 candles",
    "created_ts": "2026-02-15T21:30:00",
    "version": 1,
    "status": "active",
    "features": [
      "close", "rsi_14", "macd_histogram", 
      "volume_sma_20", "atr_14", "adx_14"
    ],
    "feature_count": 6,
    "hyperparams": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1,
      "subsample": 0.8
    },
    "training": {
      "symbols": ["BTC-USDT", "ETH-USDT", "XRP-USDT"],
      "data_points": 6000,
      "train_ratio": 0.8,
      "validation_ratio": 0.2
    },
    "performance": {
      "backtest_accuracy": 0.562,
      "backtest_precision": 0.58,
      "backtest_recall": 0.54,
      "backtest_f1": 0.56,
      "backtest_window": "7d"
    },
    "live_performance": {
      "accuracy": 0.545,
      "last_updated": "2026-02-15T21:00:00",
      "trades_evaluated": 145
    },
    "dependencies": []
  }
}
```

---

## Troubleshooting

### Accuracy < 55%
- Features aren't predictive → try different features
- Not enough training data → collect more history
- Market regime changed → retrain on fresh data
- Model too complex → simplify (fewer features, less depth)

### Consistent drift (backtest ≠ live)
- Overfitting → use more training data, regularization
- Data leakage → ensure training data doesn't include future info
- Market regime change → add regime detection features
- Slippage not accounted for → lower target accuracy threshold

### Memory issues
- Too much training data loaded → batch training
- Large feature set → feature selection to reduce dimensionality
- Model too complex → use simpler model types

---

## Next Steps

1. Implement `ml_pipeline/train.py` (training loop)
2. Implement `ml_pipeline/validate.py` (backtesting)
3. Create model trainer classes for each model type
4. Build `models/common/base_model.py` (abstract interface)
5. Build ensemble predictor
6. Integrate with daily orchestration
7. Set up model monitoring dashboard

The ML pipeline is what separates amateur traders from professionals. Get it right.
