# ML Pipeline - Installation Guide

## Quick Start

### 1. Install Dependencies

```bash
cd /home/rob/.openclaw/workspace/blofin-stack
source .venv/bin/activate

# Install ML libraries (this may take 5-10 minutes)
pip install scikit-learn xgboost torch numpy pandas joblib

# Verify installation
python -c "import sklearn, xgboost, torch; print('✓ All dependencies installed')"
```

### 2. Test the Pipeline

```bash
# Run comprehensive tests
python ml_pipeline/tests/test_train.py

# Or run with pytest for more detail
pip install pytest
pytest ml_pipeline/tests/test_train.py -v
```

### 3. Train Models (Synthetic Data)

```bash
# Quick training demo with synthetic data
python ml_pipeline/train.py

# This will:
# - Generate 5000 synthetic samples
# - Train all 5 models in parallel
# - Save models to models/model_XXX/ folders
# - Display performance metrics
```

### 4. Validate Models

```bash
python ml_pipeline/validate.py

# This will:
# - Load trained models
# - Backtest on holdout data
# - Calculate accuracy/precision/F1/MAE/RMSE
# - Compare model performance
```

### 5. Check for Drift

```bash
python ml_pipeline/tune.py

# This will:
# - Detect drifting models
# - Auto-retrain if needed
# - Log retraining events
```

## Integration with Real Data

Once Agent #1 (features) is ready:

```python
from features import get_features
from ml_pipeline.train import TrainingPipeline

# Get real features
features_df = get_features(symbol="BTC/USDT", period="1h", limit=5000)

# Ensure target columns exist:
# - target_direction (0/1)
# - target_risk (0-100)
# - target_price (float)
# - target_momentum (0/1/2)
# - target_volatility (float)

# Train on real data
pipeline = TrainingPipeline()
results = pipeline.train_all_models(features_df)
```

## Directory Structure After Training

```
models/
├── model_direction_predictor/
│   ├── model.pkl           (XGBoost model)
│   ├── scaler.pkl          (Feature scaler)
│   ├── config.json         (Features & hyperparams)
│   └── metadata.json       (Performance & timestamps)
├── model_risk_scorer/
│   └── ...
├── model_price_predictor/
│   └── ...
├── model_momentum_classifier/
│   └── ...
└── model_volatility_regressor/
    └── ...

ml_pipeline/
├── performance_history.json  (Created after first validation)
└── retrain_log.json         (Created after first retrain)
```

## Troubleshooting

### PyTorch Installation Issues

If torch fails to install:

```bash
# CPU-only version (smaller, faster install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA/GPU Support

For GPU acceleration (optional):

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If needed, install CUDA-enabled PyTorch
# Visit: https://pytorch.org/get-started/locally/
```

### Memory Issues

If training fails with OOM:

```python
# Reduce sample size
features_df = pipeline.generate_synthetic_data(n_samples=1000)  # Instead of 5000

# Or reduce batch size for neural net
# Edit ml_pipeline/models/price_predictor.py:
# "batch_size": 16  # Instead of 32
```

### Test Failures

If tests fail:

```bash
# Run individual test classes
python -m unittest ml_pipeline.tests.test_train.TestModelTraining -v
python -m unittest ml_pipeline.tests.test_train.TestValidation -v
python -m unittest ml_pipeline.tests.test_train.TestEnsemble -v
python -m unittest ml_pipeline.tests.test_train.TestTuning -v
```

## Performance Optimization

### Parallel Training

Adjust worker count based on CPU cores:

```python
# More cores = faster training
results = pipeline.train_all_models(features_df, max_workers=8)

# Reduce if system is constrained
results = pipeline.train_all_models(features_df, max_workers=2)
```

### Neural Network Speed

For faster neural net training:

```python
# Reduce epochs in ml_pipeline/models/price_predictor.py
"epochs": 20  # Instead of 50

# Or increase batch size
"batch_size": 64  # Instead of 32
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Run tests to verify everything works
3. ⏳ Wait for Agent #1 (features) to complete
4. ⏳ Integrate real feature data
5. ⏳ Train on historical data
6. ⏳ Deploy for production predictions

---

Questions? Check `ml_pipeline/README.md` for full documentation.
