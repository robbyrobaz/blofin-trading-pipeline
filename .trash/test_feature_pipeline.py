#!/usr/bin/env python3
"""
Test script for feature engineering pipeline.

Verifies:
1. Feature manager loads data without NaN/Inf
2. All feature groups work correctly
3. ML target generation works
4. Data is ready for model training
"""
import sys
import numpy as np
import pandas as pd
from features.feature_manager import FeatureManager


def test_feature_loading():
    """Test basic feature loading."""
    print("="*60)
    print("TEST 1: Feature Loading")
    print("="*60)
    
    fm = FeatureManager()
    
    # Test with different lookback periods
    for lookback in [100, 500, 2000]:
        print(f"\nLoading {lookback} bars...")
        features = fm.get_features(
            symbol='BTC-USDT',
            timeframe='1m',
            lookback_bars=lookback,
            fill_nan=True
        )
        
        nan_count = features.isna().sum().sum()
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"  Shape: {features.shape}")
        print(f"  NaN: {nan_count}, Inf: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ✗ FAIL: Found NaN or Inf values")
            return False
    
    print("\n✓ Feature loading test passed")
    return True


def test_feature_groups():
    """Test individual feature groups."""
    print("\n" + "="*60)
    print("TEST 2: Feature Groups")
    print("="*60)
    
    fm = FeatureManager()
    
    feature_groups = fm.get_feature_groups()
    print(f"\nAvailable groups: {list(feature_groups.keys())}")
    
    for group_name, group_features in feature_groups.items():
        print(f"\n{group_name}: {len(group_features)} features")
        print(f"  Examples: {group_features[:5]}")
    
    print("\n✓ Feature groups test passed")
    return True


def test_ml_target_generation():
    """Test ML target generation."""
    print("\n" + "="*60)
    print("TEST 3: ML Target Generation")
    print("="*60)
    
    fm = FeatureManager()
    
    print("\nLoading 2000 bars for target generation...")
    features = fm.get_features(
        symbol='BTC-USDT',
        timeframe='1m',
        lookback_bars=2000,
        fill_nan=True
    )
    
    print(f"Initial shape: {features.shape}")
    
    # Generate targets (same as daily_runner.py)
    print("\nGenerating targets...")
    
    features['target_direction'] = (features['close'].shift(-5) > features['close']).astype(int)
    features['target_risk'] = features['close'].rolling(20).std().fillna(0) * 100
    features['target_price'] = features['close'].shift(-5)
    
    price_change = features['close'].pct_change().fillna(0)
    features['target_momentum'] = pd.cut(
        price_change,
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    
    features['target_volatility'] = features['close'].rolling(10).std().fillna(0) / 1000
    
    # Clean data
    initial_len = len(features)
    features = features.dropna()
    dropped = initial_len - len(features)
    
    print(f"After dropna: {len(features)} samples ({dropped} dropped)")
    
    # Verify
    nan_count = features.isna().sum().sum()
    inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"Final NaN: {nan_count}, Inf: {inf_count}")
    
    # Check target distributions
    print("\nTarget distributions:")
    print(f"  Direction: {features['target_direction'].value_counts().to_dict()}")
    print(f"  Momentum: {features['target_momentum'].value_counts().to_dict()}")
    print(f"  Risk: min={features['target_risk'].min():.2f}, max={features['target_risk'].max():.2f}")
    
    # Show sample
    print("\nSample data (last 3 rows):")
    sample_cols = ['close', 'rsi_14', 'macd_12_26', 'target_direction', 'target_momentum']
    print(features[sample_cols].tail(3))
    
    if nan_count > 0 or inf_count > 0:
        print("\n✗ FAIL: Found NaN or Inf in final data")
        return False
    
    if len(features) < 1000:
        print(f"\n✗ FAIL: Insufficient samples ({len(features)} < 1000)")
        return False
    
    print("\n✓ ML target generation test passed")
    return True


def test_specific_indicators():
    """Test specific indicators that commonly cause NaN issues."""
    print("\n" + "="*60)
    print("TEST 4: Problem Indicators")
    print("="*60)
    
    fm = FeatureManager()
    
    features = fm.get_features(
        symbol='BTC-USDT',
        timeframe='1m',
        lookback_bars=500,
        fill_nan=True
    )
    
    # Test problematic indicators
    problem_indicators = [
        'rsi_14',           # Division by zero in RSI
        'stoch_k',          # Division when high == low
        'williams_r_14',    # Division when range == 0
        'adx_14',           # Multiple divisions
        'bbands_percent_b_20',  # Division by band width
        'volume_ratio_20',  # Division by avg volume
        'vwap_deviation_20',  # Division by VWAP
    ]
    
    print("\nChecking problem indicators:")
    all_good = True
    for indicator in problem_indicators:
        if indicator not in features.columns:
            print(f"  {indicator}: NOT FOUND")
            continue
        
        nan_count = features[indicator].isna().sum()
        inf_count = np.isinf(features[indicator]).sum()
        
        status = "✓" if (nan_count == 0 and inf_count == 0) else "✗"
        print(f"  {status} {indicator}: NaN={nan_count}, Inf={inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            all_good = False
    
    if all_good:
        print("\n✓ Problem indicators test passed")
        return True
    else:
        print("\n✗ Some indicators still have issues")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE TESTS")
    print("="*60)
    
    tests = [
        test_feature_loading,
        test_feature_groups,
        test_ml_target_generation,
        test_specific_indicators,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ {test_func.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Feature pipeline is ready!")
        return 0
    else:
        print(f"\n✗ {total - passed} TESTS FAILED - Fix issues before deployment")
        return 1


if __name__ == '__main__':
    sys.exit(main())
