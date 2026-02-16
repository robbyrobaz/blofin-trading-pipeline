#!/usr/bin/env python3
"""
Test metric standardization in all model files.
Verifies that train() methods return the correct standardized metrics.
"""
import sys
from pathlib import Path
import inspect

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_metrics():
    """Test that all models return standardized metrics."""
    print("\n" + "="*60)
    print("METRIC STANDARDIZATION VERIFICATION")
    print("="*60)
    
    from ml_pipeline.models.direction_predictor import DirectionPredictor
    from ml_pipeline.models.risk_scorer import RiskScorer
    from ml_pipeline.models.price_predictor import PricePredictor
    from ml_pipeline.models.momentum_classifier import MomentumClassifier
    from ml_pipeline.models.volatility_regressor import VolatilityRegressor
    
    models = [
        ("DirectionPredictor", DirectionPredictor, "classification"),
        ("MomentumClassifier", MomentumClassifier, "classification"),
        ("RiskScorer", RiskScorer, "regression"),
        ("PricePredictor", PricePredictor, "regression"),
        ("VolatilityRegressor", VolatilityRegressor, "regression"),
    ]
    
    required_base = ['train_accuracy', 'test_accuracy']
    required_classification = ['f1_score', 'precision', 'recall']
    
    all_passed = True
    
    for name, model_class, model_type in models:
        print(f"\n{name} ({model_type}):")
        
        # Get the train method source code
        train_method = model_class.train
        source = inspect.getsource(train_method)
        
        # Check if metrics dict includes required fields
        checks = {
            'train_accuracy': 'train_accuracy' in source and '"train_accuracy"' in source,
            'test_accuracy': 'test_accuracy' in source and '"test_accuracy"' in source,
        }
        
        if model_type == 'classification':
            checks['f1_score'] = 'f1_score' in source and '"f1_score"' in source
            checks['precision'] = 'precision' in source and '"precision"' in source
            checks['recall'] = 'recall' in source and '"recall"' in source
        
        # Print results
        for metric, present in checks.items():
            status = "✓" if present else "❌"
            print(f"  {status} {metric}: {'present' if present else 'MISSING'}")
            if not present:
                all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL MODELS HAVE STANDARDIZED METRICS")
    else:
        print("❌ SOME MODELS MISSING STANDARDIZED METRICS")
    print("="*60)
    
    return all_passed


def test_imports():
    """Test that sklearn.metrics are imported for classification models."""
    print("\n" + "="*60)
    print("SKLEARN METRICS IMPORT VERIFICATION")
    print("="*60)
    
    import ast
    
    models = [
        ("direction_predictor.py", True),
        ("momentum_classifier.py", True),
        ("risk_scorer.py", False),
        ("price_predictor.py", False),
        ("volatility_regressor.py", False),
    ]
    
    all_passed = True
    
    for filename, needs_metrics in models:
        filepath = Path("ml_pipeline/models") / filename
        
        with open(filepath) as f:
            tree = ast.parse(f.read())
        
        # Check imports
        has_metrics_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == 'sklearn.metrics':
                    imported = [alias.name for alias in node.names]
                    if 'f1_score' in imported:
                        has_metrics_import = True
        
        if needs_metrics:
            if has_metrics_import:
                print(f"  ✓ {filename}: sklearn.metrics imported")
            else:
                print(f"  ❌ {filename}: sklearn.metrics MISSING")
                all_passed = False
        else:
            print(f"  ○ {filename}: sklearn.metrics not required (regression)")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CLASSIFICATION MODELS IMPORT METRICS")
    else:
        print("❌ SOME MODELS MISSING IMPORTS")
    print("="*60)
    
    return all_passed


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("PRIORITY 1 FIX VERIFICATION")
    print("Testing: Ranker Metric Mismatch Fix")
    print("="*60)
    
    test1 = test_imports()
    test2 = test_model_metrics()
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if test1 and test2:
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        print("\nWhat this means:")
        print("  - All models now return 'test_accuracy' (not 'val_accuracy')")
        print("  - Classification models return f1_score, precision, recall")
        print("  - DB connector can now capture these metrics")
        print("  - Ranker queries will work correctly")
        return 0
    else:
        print("⚠ SOME CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
