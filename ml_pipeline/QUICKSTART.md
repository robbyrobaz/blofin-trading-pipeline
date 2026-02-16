# ML Pipeline - Quick Reference

## 🚀 One-Command Demo

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate

# Install dependencies (first time only, ~5-10 min)
pip install scikit-learn xgboost torch numpy pandas joblib

# Run complete demo
python ml_pipeline/train.py
```

## 📝 Common Usage Patterns

### Train All Models
```python
from ml_pipeline.train import TrainingPipeline

pipeline = TrainingPipeline()
features_df = pipeline.generate_synthetic_data(n_samples=5000)
results = pipeline.train_all_models(features_df, max_workers=5)
print(f"Trained {results['successful']}/5 models")
```

### Validate Models
```python
from ml_pipeline.validate import ValidationPipeline

validator = ValidationPipeline()
results = validator.validate_all_models(features_df, days_back=7)

# Compare by accuracy
print(validator.compare_models("accuracy"))
```

### Detect Drift & Retrain
```python
from ml_pipeline.tune import TuningPipeline

tuner = TuningPipeline(drift_threshold=0.10)
results = tuner.auto_tune(features_df, auto_retrain=True)
print(f"{len(results['drifting_models'])} models retrained")
```

### Make Predictions
```python
from models.common.predictor import load_model

model = load_model("direction_predictor", "models")
prediction = model.predict(current_features)
# {"direction": "UP", "confidence": 0.75}
```

### Ensemble Prediction
```python
from models.common.predictor import EnsemblePredictor, load_model

ensemble = EnsemblePredictor(ensemble_type="weighted_avg")
ensemble.add_model(load_model("direction_predictor", "models"), weight=0.6)
ensemble.add_model(load_model("momentum_classifier", "models"), weight=0.4)
prediction = ensemble.predict(features)
```

## 🧪 Testing

```bash
# Run all tests
python ml_pipeline/tests/test_train.py

# Run with pytest (more detail)
pip install pytest
pytest ml_pipeline/tests/test_train.py -v -s
```

## 📦 Models

| Model | Type | Input Features | Output |
|-------|------|----------------|--------|
| direction_predictor | XGBoost | close, returns, rsi, macd, volume | `{"direction": "UP", "confidence": 0.75}` |
| risk_scorer | Random Forest | volatility, ATR, drawdown, Sharpe | `35` (0-100) |
| price_predictor | Neural Net | momentum, trend, volume | `{"predicted_price": 45240, "confidence": 0.68}` |
| momentum_classifier | SVM | momentum, ROC, trend, stochastic | `{"state": "accelerating", "confidence": 0.74}` |
| volatility_regressor | Gradient Boosting | volatility, ATR, volume | `{"predicted_vol": 0.0145, "confidence": 0.71}` |

## 🔧 Configuration

### Training Pipeline
```python
pipeline = TrainingPipeline(
    base_model_dir="models"  # Where to save models
)

results = pipeline.train_all_models(
    features_df,
    max_workers=5  # Parallel threads (adjust for CPU cores)
)
```

### Validation Pipeline
```python
validator = ValidationPipeline(base_model_dir="models")

results = validator.validate_all_models(
    features_df,
    symbols=["BTC/USDT", "ETH/USDT"],  # Optional
    days_back=7  # Holdout period
)
```

### Tuning Pipeline
```python
tuner = TuningPipeline(
    base_model_dir="models",
    drift_threshold=0.10,  # 10% performance drop
    history_file="ml_pipeline/performance_history.json"
)
```

## 📊 Performance Metrics

**Classification Models** (direction, momentum):
- accuracy, precision, recall, f1_score

**Regression Models** (risk, price, volatility):
- mae, mse, rmse, r2_score, mape

## 🗂️ File Locations

```
models/
├── model_direction_predictor/
│   ├── model.pkl          # Trained model
│   ├── scaler.pkl         # Feature scaler
│   ├── config.json        # Configuration
│   └── metadata.json      # Performance metrics
├── model_risk_scorer/
└── ... (5 models total)

ml_pipeline/
├── performance_history.json  # Metrics over time
└── retrain_log.json         # Retraining events
```

## 🔗 Integration Example

```python
# Complete workflow
from ml_pipeline.train import TrainingPipeline
from ml_pipeline.validate import ValidationPipeline
from ml_pipeline.tune import TuningPipeline
from models.common.predictor import load_model

# 1. Get features (from Agent #1)
features_df = get_features(symbol="BTC/USDT", limit=5000)

# 2. Train models
trainer = TrainingPipeline()
train_results = trainer.train_all_models(features_df)

# 3. Validate
validator = ValidationPipeline()
val_results = validator.validate_all_models(features_df)

# 4. Check drift and retrain if needed
tuner = TuningPipeline()
tune_results = tuner.auto_tune(features_df, auto_retrain=True)

# 5. Make predictions
model = load_model("direction_predictor", "models")
current_prediction = model.predict(latest_features)

# 6. Use in trading (with Agent #2)
if current_prediction["direction"] == "UP" and current_prediction["confidence"] > 0.7:
    # Send to backtester/executor
    execute_trade()
```

## 🐛 Common Issues

**ModuleNotFoundError**: Install dependencies
```bash
pip install scikit-learn xgboost torch numpy pandas joblib
```

**CUDA/GPU errors**: Use CPU-only PyTorch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Out of memory**: Reduce sample size or batch size
```python
features_df = pipeline.generate_synthetic_data(n_samples=1000)  # Instead of 5000
```

**Test failures**: Run individual test classes
```bash
python -m unittest ml_pipeline.tests.test_train.TestModelTraining -v
```

## 📚 Documentation

- Full docs: `ml_pipeline/README.md`
- Installation: `ml_pipeline/INSTALL.md`
- Build summary: `ML_PIPELINE_SUMMARY.md`

---

**Quick help**: `python ml_pipeline/train.py --help`
