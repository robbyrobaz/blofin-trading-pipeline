"""
Tests for ML Pipeline - Training, Validation, and Models.
"""
import unittest
import os
import sys
import shutil
import tempfile
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml_pipeline.train import TrainingPipeline
from ml_pipeline.validate import ValidationPipeline
from ml_pipeline.tune import TuningPipeline
from ml_pipeline.models.direction_predictor import DirectionPredictor
from ml_pipeline.models.risk_scorer import RiskScorer
from ml_pipeline.models.price_predictor import PricePredictor
from ml_pipeline.models.momentum_classifier import MomentumClassifier
from ml_pipeline.models.volatility_regressor import VolatilityRegressor
from models.common.predictor import load_model, EnsemblePredictor


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.test_dir, "models")
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Generate test data
        cls.pipeline = TrainingPipeline(base_model_dir=cls.model_dir)
        cls.features_df = cls.pipeline.generate_synthetic_data(n_samples=1000)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_direction_predictor_train(self):
        """Test direction predictor training."""
        model = DirectionPredictor()
        X = self.features_df.drop(columns=["target_direction"])
        y = self.features_df["target_direction"]
        
        metrics = model.train(X, y)
        
        self.assertIsNotNone(metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("val_accuracy", metrics)
        self.assertGreater(metrics["val_accuracy"], 0.4)  # Better than random
    
    def test_risk_scorer_train(self):
        """Test risk scorer training."""
        model = RiskScorer()
        X = self.features_df.drop(columns=["target_risk"])
        y = self.features_df["target_risk"]
        
        metrics = model.train(X, y)
        
        self.assertIsNotNone(metrics)
        self.assertIn("train_r2", metrics)
        self.assertIn("val_mae", metrics)
    
    def test_price_predictor_train(self):
        """Test price predictor training."""
        model = PricePredictor()
        X = self.features_df.drop(columns=["target_price"])
        y = self.features_df["target_price"]
        
        metrics = model.train(X, y)
        
        self.assertIsNotNone(metrics)
        self.assertIn("train_mae", metrics)
        self.assertIn("val_rmse", metrics)
    
    def test_momentum_classifier_train(self):
        """Test momentum classifier training."""
        model = MomentumClassifier()
        X = self.features_df.drop(columns=["target_momentum"])
        y = self.features_df["target_momentum"]
        
        metrics = model.train(X, y)
        
        self.assertIsNotNone(metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("val_accuracy", metrics)
    
    def test_volatility_regressor_train(self):
        """Test volatility regressor training."""
        model = VolatilityRegressor()
        X = self.features_df.drop(columns=["target_volatility"])
        y = self.features_df["target_volatility"]
        
        metrics = model.train(X, y)
        
        self.assertIsNotNone(metrics)
        self.assertIn("train_r2", metrics)
        self.assertIn("val_mae", metrics)
    
    def test_model_save_load(self):
        """Test model save and load."""
        # Train model
        model = DirectionPredictor()
        X = self.features_df.drop(columns=["target_direction"])
        y = self.features_df["target_direction"]
        model.train(X, y)
        
        # Save
        model_path = os.path.join(self.model_dir, "test_model")
        model.save(model_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(model_path, "model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "metadata.json")))
        
        # Load
        new_model = DirectionPredictor()
        new_model.load(model_path)
        
        self.assertIsNotNone(new_model.model)
        self.assertIsNotNone(new_model.scaler)
    
    def test_parallel_training(self):
        """Test parallel training of all models."""
        results = self.pipeline.train_all_models(self.features_df, max_workers=5)
        
        self.assertIsNotNone(results)
        self.assertIn("successful", results)
        self.assertIn("results", results)
        
        # At least some models should train successfully
        self.assertGreater(results["successful"], 0)


class TestValidation(unittest.TestCase):
    """Test validation pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.test_dir, "models")
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Train models
        cls.train_pipeline = TrainingPipeline(base_model_dir=cls.model_dir)
        cls.features_df = cls.train_pipeline.generate_synthetic_data(n_samples=1000)
        cls.train_pipeline.train_all_models(cls.features_df, max_workers=5)
        
        cls.val_pipeline = ValidationPipeline(base_model_dir=cls.model_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_backtest_model(self):
        """Test backtesting a single model."""
        result = self.val_pipeline.backtest_model(
            model_name="direction_predictor",
            features_df=self.features_df,
            target_col="target_direction",
            days_back=7
        )
        
        self.assertIn("success", result)
        if result["success"]:
            self.assertIn("metrics", result)
            self.assertIn("accuracy", result["metrics"])
    
    def test_validate_all_models(self):
        """Test validation of all models."""
        results = self.val_pipeline.validate_all_models(self.features_df, days_back=7)
        
        self.assertIsNotNone(results)
        self.assertIn("results", results)
        
        # Check that at least some models were validated
        self.assertGreater(len(results["results"]), 0)
    
    def test_model_comparison(self):
        """Test model comparison."""
        self.val_pipeline.validate_all_models(self.features_df, days_back=7)
        
        # Try accuracy comparison
        comp_df = self.val_pipeline.compare_models("accuracy")
        
        # May be empty if no classification models, which is fine
        self.assertIsInstance(comp_df, pd.DataFrame)


class TestEnsemble(unittest.TestCase):
    """Test ensemble prediction."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.test_dir, "models")
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Train a couple of models
        cls.train_pipeline = TrainingPipeline(base_model_dir=cls.model_dir)
        cls.features_df = cls.train_pipeline.generate_synthetic_data(n_samples=1000)
        
        # Train just direction predictor
        model = DirectionPredictor()
        X = cls.features_df.drop(columns=["target_direction"])
        y = cls.features_df["target_direction"]
        model.train(X, y)
        model.save(os.path.join(cls.model_dir, "model_direction_predictor"))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_ensemble_weighted_avg(self):
        """Test weighted average ensemble."""
        ensemble = EnsemblePredictor(ensemble_type="weighted_avg")
        
        # Load model
        model1 = load_model("direction_predictor", self.model_dir)
        
        # Add to ensemble
        ensemble.add_model(model1, weight=1.0)
        
        # Make prediction
        X = self.features_df.drop(columns=["target_direction"]).head(1)
        prediction = ensemble.predict(X)
        
        self.assertIsNotNone(prediction)
    
    def test_ensemble_save_load(self):
        """Test ensemble save and load."""
        ensemble = EnsemblePredictor(ensemble_type="weighted_avg")
        
        model1 = load_model("direction_predictor", self.model_dir)
        ensemble.add_model(model1, weight=1.0)
        
        # Save config
        config_path = os.path.join(self.test_dir, "ensemble_config.json")
        ensemble.save_config(config_path)
        
        self.assertTrue(os.path.exists(config_path))


class TestTuning(unittest.TestCase):
    """Test tuning pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.model_dir = os.path.join(cls.test_dir, "models")
        cls.history_file = os.path.join(cls.test_dir, "history.json")
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Train models
        cls.train_pipeline = TrainingPipeline(base_model_dir=cls.model_dir)
        cls.features_df = cls.train_pipeline.generate_synthetic_data(n_samples=1000)
        cls.train_pipeline.train_all_models(cls.features_df, max_workers=5)
        
        cls.tune_pipeline = TuningPipeline(
            base_model_dir=cls.model_dir,
            drift_threshold=0.10,
            history_file=cls.history_file
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_detect_drift(self):
        """Test drift detection."""
        drifting = self.tune_pipeline.detect_drifting_models(self.features_df, days_back=7)
        
        # First run should establish baseline, no drift
        self.assertIsInstance(drifting, list)
    
    def test_retrain_model(self):
        """Test model retraining."""
        # First establish baseline
        self.tune_pipeline.detect_drifting_models(self.features_df, days_back=7)
        
        # Retrain direction predictor
        result = self.tune_pipeline.retrain_model(
            "direction_predictor",
            self.features_df
        )
        
        self.assertIn("success", result)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsemble))
    suite.addTests(loader.loadTestsFromTestCase(TestTuning))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
