# ML Pipeline Module

Complete machine learning training, validation, and tuning system for Blofin Stack.

## Overview

The ML Pipeline provides:
- **5 specialized models** trained in parallel
- **Automated validation** with comprehensive metrics
- **Drift detection** and automatic retraining
- **Ensemble predictions** with multiple strategies

## Architecture

```
ml_pipeline/
├── train.py           # Training orchestration (parallel execution)
├── validate.py        # Backtesting and validation
├── tune.py           # Drift detection and retraining
├── models/           # Model implementations
│   ├── direction_predictor.py    (XGBoost)
│   ├── risk_scorer.py            (Random Forest)
│   ├── price_predictor.py        (Neural Network)
│   ├── momentum_classifier.py    (SVM)
│   └── volatility_regressor.py   (Gradient Boosting)
└── tests/
    └── test_train.py  # Comprehensive test suite

models/
└── common/
    ├── base_model.py  # Abstract base class
    └── predictor.py   # Model loading & ensemble
```

## Models

### 1. Direction Predictor (XGBoost)
- **Purpose**: Predicts UP/DOWN for next 5 candles
- **Features**: close, returns, rsi_14, macd, volume_sma
- **Output**: `{"direction": "UP", "confidence": 0.75}`

### 2. Risk Scorer (Random Forest)
- **Purpose**: Predicts risk level 0-100
- **Features**: volatility features, drawdown history
- **Output**: `35` (low) to `85` (high)

### 3. Price Predictor (Neural Network)
- **Purpose**: Predicts future price
- **Features**: momentum, trend, volume features
- **Output**: `{"predicted_price": 45240, "confidence": 0.68}`

### 4. Momentum Classifier (SVM)
- **Purpose**: Predicts momentum direction
- **Features**: momentum indicators, ROC, trend
- **Output**: `{"state": "accelerating", "confidence": 0.74}`

### 5. Volatility Regressor (Gradient Boosting)
- **Purpose**: Predicts future volatility
- **Features**: volatility features, volume
- **Output**: `{"predicted_vol": 0.0145, "confidence": 0.71}`

## Usage

### Training All Models

```python
from ml_pipeline.train import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(base_model_dir="models")

# Prepare your features DataFrame
# Must include target columns: target_direction, target_risk, 
# target_price, target_momentum, target_volatility
features_df = get_features()  # Your feature engineering

# Train all models in parallel (5 workers, ~60 min total)
results = pipeline.train_all_models(features_df, max_workers=5)

print(f"Successful: {results['successful']}/5")
```

### Validating Models

```python
from ml_pipeline.validate import ValidationPipeline

# Initialize validator
validator = ValidationPipeline(base_model_dir="models")

# Backtest all models on holdout data
results = validator.validate_all_models(
    features_df, 
    symbols=["BTC/USDT", "ETH/USDT"],
    days_back=7
)

# Compare models
accuracy_comparison = validator.compare_models("accuracy")
print(accuracy_comparison)
```

### Drift Detection & Retraining

```python
from ml_pipeline.tune import TuningPipeline

# Initialize tuning pipeline
tuner = TuningPipeline(
    base_model_dir="models",
    drift_threshold=0.10  # 10% performance drop triggers retrain
)

# Detect drifting models
drifting = tuner.detect_drifting_models(features_df, days_back=7)

for drift in drifting:
    print(f"{drift['model_name']}: {drift['drift_percent']:.1f}% drift")

# Auto-tune: detect + retrain
results = tuner.auto_tune(features_df, auto_retrain=True)
```

### Making Predictions

```python
from models.common.predictor import load_model, EnsemblePredictor

# Load single model
direction_model = load_model("direction_predictor", "models")
prediction = direction_model.predict(features)
print(prediction)  # {"direction": "UP", "confidence": 0.75}

# Ensemble prediction
ensemble = EnsemblePredictor(ensemble_type="weighted_avg")
ensemble.add_model(load_model("direction_predictor", "models"), weight=0.6)
ensemble.add_model(load_model("momentum_classifier", "models"), weight=0.4)

ensemble_pred = ensemble.predict(features)
```

## Model Persistence

All models are saved with:
- `model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `config.json` - Features used, hyperparameters
- `metadata.json` - Performance metrics, creation timestamp, feature importance

Structure:
```
models/
├── model_direction_predictor/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── config.json
│   └── metadata.json
├── model_risk_scorer/
│   └── ...
└── ...
```

## Performance Tracking

The tuning pipeline maintains performance history:
- `ml_pipeline/performance_history.json` - Historical metrics per model
- `ml_pipeline/retrain_log.json` - Retraining events

## Testing

Run the comprehensive test suite:

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate
python -m pytest ml_pipeline/tests/test_train.py -v
```

Or run directly:
```bash
python ml_pipeline/tests/test_train.py
```

Tests cover:
- ✓ Model training
- ✓ Model save/load
- ✓ Parallel training
- ✓ Validation metrics
- ✓ Ensemble predictions
- ✓ Drift detection
- ✓ Retraining

## Integration with Other Agents

### Agent #1 (Features)
```python
# Features agent provides get_features()
from features import get_features

features_df = get_features(symbol="BTC/USDT", period="1d", limit=1000)

# Use for training
pipeline.train_all_models(features_df)
```

### Agent #2 (Backtester)
```python
# Use predictions in backtesting
from ml_pipeline.train import TrainingPipeline

direction_pred = direction_model.predict(current_features)
if direction_pred["direction"] == "UP" and direction_pred["confidence"] > 0.7:
    # Signal to backtester
    generate_buy_signal()
```

## Requirements

```
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
```

## Performance Expectations

- **Training time**: ~60 min for all 5 models (parallel, 1000 samples)
- **Inference**: <100ms per prediction
- **Accuracy targets**:
  - Direction Predictor: >55% (better than random)
  - Risk Scorer: MAE <10 points
  - Price Predictor: MAPE <5%
  - Momentum Classifier: >50%
  - Volatility Regressor: R² >0.6

## Notes

- Models are trained with 80/20 train/validation split
- Features are normalized using StandardScaler
- PyTorch uses GPU if available (CUDA)
- Parallel training uses ThreadPoolExecutor
- All models implement the BaseModel interface

## Troubleshooting

**Model not found:**
```python
# Ensure model is trained first
pipeline.train_all_models(features_df)
```

**Feature mismatch:**
```python
# Check required features for each model
model = DirectionPredictor()
print(model.feature_names)
```

**Drift false positives:**
```python
# Adjust drift threshold
tuner = TuningPipeline(drift_threshold=0.15)  # 15% instead of 10%
```

## Roadmap

- [ ] Add LSTM for time series
- [ ] Implement stacking ensemble
- [ ] Add hyperparameter tuning (Optuna)
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Real-time prediction API

---

Built for Blofin Stack by Agent #3 🤖
