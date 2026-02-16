#!/usr/bin/env python3
"""
ML Pipeline Verification Script
Checks installation and runs quick tests.
"""
import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required = [
        "sklearn",
        "xgboost", 
        "torch",
        "numpy",
        "pandas",
        "joblib"
    ]
    
    missing = []
    for module in required:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install scikit-learn xgboost torch numpy pandas joblib")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def check_syntax():
    """Check if all Python files compile."""
    print("\nChecking Python syntax...")
    
    files = [
        "ml_pipeline/train.py",
        "ml_pipeline/validate.py",
        "ml_pipeline/tune.py",
        "ml_pipeline/models/direction_predictor.py",
        "ml_pipeline/models/risk_scorer.py",
        "ml_pipeline/models/price_predictor.py",
        "ml_pipeline/models/momentum_classifier.py",
        "ml_pipeline/models/volatility_regressor.py",
        "models/common/base_model.py",
        "models/common/predictor.py",
    ]
    
    errors = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                compile(f.read(), filepath, 'exec')
            print(f"  ✓ {filepath}")
        except SyntaxError as e:
            print(f"  ✗ {filepath} - SYNTAX ERROR")
            errors.append((filepath, str(e)))
    
    if errors:
        print("\n⚠ Syntax errors found:")
        for filepath, error in errors:
            print(f"  {filepath}: {error}")
        return False
    
    print("\n✓ All files compile successfully")
    return True


def check_file_structure():
    """Check if all required files exist."""
    print("\nChecking file structure...")
    
    required_files = [
        "ml_pipeline/__init__.py",
        "ml_pipeline/train.py",
        "ml_pipeline/validate.py",
        "ml_pipeline/tune.py",
        "ml_pipeline/README.md",
        "ml_pipeline/INSTALL.md",
        "ml_pipeline/QUICKSTART.md",
        "ml_pipeline/models/__init__.py",
        "ml_pipeline/models/direction_predictor.py",
        "ml_pipeline/models/risk_scorer.py",
        "ml_pipeline/models/price_predictor.py",
        "ml_pipeline/models/momentum_classifier.py",
        "ml_pipeline/models/volatility_regressor.py",
        "ml_pipeline/tests/__init__.py",
        "ml_pipeline/tests/test_train.py",
        "models/__init__.py",
        "models/common/__init__.py",
        "models/common/base_model.py",
        "models/common/predictor.py",
        "ML_PIPELINE_SUMMARY.md",
    ]
    
    missing = []
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - MISSING")
            missing.append(filepath)
    
    if missing:
        print(f"\n⚠ Missing files: {len(missing)}")
        return False
    
    print(f"\n✓ All {len(required_files)} files present")
    return True


def run_quick_test():
    """Run a quick import test."""
    print("\nRunning quick import test...")
    
    try:
        # Test imports
        from ml_pipeline.train import TrainingPipeline
        from ml_pipeline.validate import ValidationPipeline
        from ml_pipeline.tune import TuningPipeline
        from models.common.base_model import BaseModel
        from models.common.predictor import load_model, EnsemblePredictor
        from ml_pipeline.models.direction_predictor import DirectionPredictor
        from ml_pipeline.models.risk_scorer import RiskScorer
        from ml_pipeline.models.price_predictor import PricePredictor
        from ml_pipeline.models.momentum_classifier import MomentumClassifier
        from ml_pipeline.models.volatility_regressor import VolatilityRegressor
        
        print("  ✓ All modules import successfully")
        
        # Test instantiation
        pipeline = TrainingPipeline()
        print("  ✓ TrainingPipeline instantiates")
        
        validator = ValidationPipeline()
        print("  ✓ ValidationPipeline instantiates")
        
        tuner = TuningPipeline()
        print("  ✓ TuningPipeline instantiates")
        
        ensemble = EnsemblePredictor()
        print("  ✓ EnsemblePredictor instantiates")
        
        print("\n✓ Quick test passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Quick test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification routine."""
    print("="*60)
    print("ML Pipeline Verification")
    print("="*60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    checks = [
        ("File Structure", check_file_structure),
        ("Python Syntax", check_syntax),
        ("Dependencies", check_dependencies),
        ("Quick Test", run_quick_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nML Pipeline is ready to use!")
        print("\nNext steps:")
        print("  1. Run demo: python ml_pipeline/train.py")
        print("  2. Run tests: python ml_pipeline/tests/test_train.py")
        print("  3. Read docs: cat ml_pipeline/README.md")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nReview the errors above and fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
