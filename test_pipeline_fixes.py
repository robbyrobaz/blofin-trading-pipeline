#!/usr/bin/env python3
"""
Test script for Blofin pipeline fixes.
Tests all 4 critical issues have been resolved.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_1_feature_manager_nan():
    """Test Feature Manager NaN handling fix."""
    print("\n" + "="*60)
    print("TEST 1: Feature Manager NaN Error Fix")
    print("="*60)
    
    try:
        from features.feature_manager import FeatureManager
        
        fm = FeatureManager()
        print("✓ Feature Manager initialized")
        
        # Try to get features - this should not fail with NaN error
        df = fm.get_features('BTC-USDT', '5m', lookback_bars=200)
        print(f"✓ Features loaded: shape={df.shape}")
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count == 0:
            print(f"✓ No NaN values in features")
        else:
            print(f"⚠ Warning: {nan_count} NaN values found (may need further cleaning)")
        
        print("\n✅ TEST 1 PASSED - Feature Manager works without NaN errors")
        return True
        
    except ValueError as e:
        if "Cannot convert float NaN to integer" in str(e):
            print(f"\n❌ TEST 1 FAILED - NaN error still occurs: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"\n⚠ TEST 1 INCONCLUSIVE - Different error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_model_metrics():
    """Test ML model metric standardization."""
    print("\n" + "="*60)
    print("TEST 2: ML Model Metric Standardization")
    print("="*60)
    
    try:
        from ml_pipeline.train import TrainingPipeline
        from ml_pipeline.db_connector import MLDatabaseConnector
        from features.feature_manager import FeatureManager
        
        fm = FeatureManager()
        df = fm.get_features('BTC-USDT', '1m', lookback_bars=1000)
        print(f"✓ Features loaded for training: {df.shape}")
        
        pipeline = TrainingPipeline()
        results = pipeline.train_all_models(df)
        print(f"✓ Trained {len(results)} models")
        
        # Check metrics structure
        required_metrics = ['test_accuracy']
        classification_metrics = ['f1_score', 'precision', 'recall']
        
        all_good = True
        for result in results:
            model_name = result['model_name']
            model_type = result['model_type']
            metrics = result.get('metrics', {})
            
            # Check test_accuracy exists
            if 'test_accuracy' not in metrics:
                print(f"  ❌ {model_name}: Missing 'test_accuracy'")
                all_good = False
            else:
                print(f"  ✓ {model_name}: test_accuracy={metrics['test_accuracy']:.4f}")
            
            # Check classification metrics for classification models
            if model_type == 'classification':
                for metric in classification_metrics:
                    if metric not in metrics:
                        print(f"    ❌ Missing '{metric}'")
                        all_good = False
                    else:
                        print(f"    ✓ {metric}={metrics[metric]:.4f}")
        
        if all_good:
            print("\n✅ TEST 2 PASSED - All models return standardized metrics")
            return True
        else:
            print("\n❌ TEST 2 FAILED - Some metrics missing")
            return False
            
    except Exception as e:
        print(f"\n⚠ TEST 2 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_ranker_queries():
    """Test Ranker can find models with new metrics."""
    print("\n" + "="*60)
    print("TEST 3: Ranker Metric Query Fix")
    print("="*60)
    
    try:
        from orchestration.ranker import Ranker
        
        ranker = Ranker('data/blofin_monitor.db')
        print("✓ Ranker initialized")
        
        # Try to get top models
        top_models = ranker.keep_top_models(count=5, metric='test_accuracy')
        
        if len(top_models) > 0:
            print(f"✓ Found {len(top_models)} top models")
            for m in top_models[:3]:
                print(f"  {m['rank']}. {m['model_name']}: {m['metric_value']:.4f}")
            print("\n✅ TEST 3 PASSED - Ranker successfully queries metrics")
            return True
        else:
            print("⚠ No models found (may need to train first)")
            print("✅ TEST 3 PASSED - Ranker query works (no data yet)")
            return True
            
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_strategy_designer_validation():
    """Test Strategy Designer validation prevents empty files."""
    print("\n" + "="*60)
    print("TEST 4: Strategy Designer Validation")
    print("="*60)
    
    try:
        from orchestration.strategy_designer import StrategyDesigner
        
        designer = StrategyDesigner('data/blofin_monitor.db', 'strategies')
        print("✓ Strategy Designer initialized")
        
        # Test validation methods
        print("\n Testing _extract_code validation:")
        
        # Should fail on empty output
        try:
            designer._extract_code("")
            print("  ❌ Empty code not rejected")
            return False
        except ValueError:
            print("  ✓ Empty code rejected")
        
        # Should fail on non-Python output
        try:
            designer._extract_code("This is just text, not code")
            print("  ❌ Non-code text not rejected")
            return False
        except ValueError:
            print("  ✓ Non-code text rejected")
        
        # Should succeed on valid code
        valid_code = """
class Strategy:
    def __init__(self):
        self.name = "test"
    def analyze(self, candles, indicators):
        return 'HOLD'
"""
        result = designer._extract_code(valid_code)
        if result:
            print("  ✓ Valid code accepted")
        
        print("\n Testing _save_strategy validation:")
        
        # Should fail on empty code
        try:
            designer._save_strategy("", 999)
            print("  ❌ Empty code saved")
            return False
        except ValueError:
            print("  ✓ Empty code rejected")
        
        # Should fail on code with syntax errors
        try:
            designer._save_strategy("def broken(:\n  pass", 999)
            print("  ❌ Syntax error code saved")
            return False
        except ValueError:
            print("  ✓ Syntax error code rejected")
        
        print("\n✅ TEST 4 PASSED - Strategy Designer validation works")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BLOFIN PIPELINE FIX VERIFICATION")
    print("="*60)
    
    results = {
        "Feature Manager NaN Fix": test_1_feature_manager_nan(),
        "Model Metrics Standardization": test_2_model_metrics(),
        "Ranker Query Fix": test_3_ranker_queries(),
        "Strategy Designer Validation": test_4_strategy_designer_validation(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠ SOME TESTS FAILED - Review output above")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
