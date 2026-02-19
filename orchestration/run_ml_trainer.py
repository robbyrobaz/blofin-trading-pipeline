#!/usr/bin/env python3
"""
ML trainer runner - retrains all models with latest trade outcomes.
Scheduled every 4h via blofin-stack-ml-trainer.timer.
"""
import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('run_ml_trainer')


def run_ml_trainer(workspace_dir: Path) -> dict:
    import numpy as np
    import pandas as pd
    from ml_pipeline.train import TrainingPipeline
    from ml_pipeline.db_connector import MLDatabaseConnector
    from features.feature_manager import FeatureManager

    db_path = workspace_dir / 'data' / 'blofin_monitor.db'
    logger = logging.getLogger('run_ml_trainer')

    pipeline = TrainingPipeline(base_model_dir=str(workspace_dir / 'models'))
    db_connector = MLDatabaseConnector(str(db_path))
    feature_manager = FeatureManager(str(db_path))

    logger.info("Fetching feature data for ML training...")
    features_df = feature_manager.get_features(
        symbol='BTC-USDT',
        timeframe='1m',
        lookback_bars=2000,
        fill_nan=True
    )

    # Generate targets
    features_df['target_direction'] = (
        features_df['close'].shift(-5) > features_df['close']
    ).astype(int)
    features_df['target_risk'] = (
        features_df['close'].rolling(20).std().fillna(0) * 100
    )
    features_df['target_price'] = features_df['close'].shift(-5)
    price_change = features_df['close'].pct_change().fillna(0)
    features_df['target_momentum'] = pd.cut(
        price_change,
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    features_df['target_volatility'] = (
        features_df['close'].rolling(10).std().fillna(0) / 1000
    )

    initial_count = len(features_df)
    features_df = features_df.dropna()
    dropped = initial_count - len(features_df)
    logger.info("Loaded %d samples (%d dropped for forward-looking targets)", len(features_df), dropped)

    nan_count = features_df.isna().sum().sum()
    inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
    if nan_count > 0:
        logger.warning("Feature data contains %d NaN values after filling", nan_count)
    if inf_count > 0:
        logger.warning("Feature data contains %d Inf values", inf_count)

    if len(features_df) < 100:
        raise ValueError(f"Insufficient training data: only {len(features_df)} samples after cleaning")

    logger.info("Training %d ML models...", len(pipeline.models))
    training_results = pipeline.train_all_models(features_df, max_workers=5)

    logger.info("Saving training results to database...")
    row_ids = db_connector.save_all_results(training_results)

    return {
        'models_trained': training_results.get('successful', 0),
        'models_failed': training_results.get('failed', 0),
        'db_rows_saved': len(row_ids),
        'training_time': training_results.get('total_time', 0),
        'samples_used': len(features_df)
    }


def main():
    workspace_dir = Path(os.environ.get(
        'BLOFIN_WORKSPACE',
        os.path.expanduser('~/.openclaw/workspace/blofin-stack')
    ))
    log_path = workspace_dir / 'data' / 'ml_trainer.log'
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("ML TRAINER START  %s", datetime.utcnow().isoformat() + 'Z')
    logger.info("=" * 60)

    start = datetime.utcnow()
    try:
        result = run_ml_trainer(workspace_dir)
        duration = (datetime.utcnow() - start).total_seconds()
        logger.info("ML TRAINER DONE in %.1fs: %s", duration, json.dumps(result, default=str))
        sys.exit(0)
    except Exception as e:
        duration = (datetime.utcnow() - start).total_seconds()
        logger.error("ML TRAINER FAILED after %.1fs: %s", duration, e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
