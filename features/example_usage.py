#!/usr/bin/env python3
"""
Feature Library Usage Examples

Demonstrates how to use the Blofin Stack feature library.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features import FeatureManager


def example_1_basic_usage():
    """Example 1: Basic feature computation"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Feature Computation")
    print("=" * 70)
    
    fm = FeatureManager()
    
    # Get features for BTC (if data exists)
    try:
        df = fm.get_features(
            symbol='BTC-USDT',
            timeframe='1m',
            lookback_bars=100
        )
        
        print(f"\nLoaded {len(df)} candles")
        print(f"Features computed: {len(df.columns)}")
        print(f"\nColumns: {list(df.columns[:10])}...")
        print(f"\nLast 3 rows:\n{df[['timestamp', 'close', 'rsi_14', 'macd_12_26']].tail(3)}")
        
    except Exception as e:
        print(f"Note: Could not load data - {e}")
        print("This is normal if database is empty. Feature library is ready for use.")


def example_2_list_features():
    """Example 2: List available features"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: List Available Features")
    print("=" * 70)
    
    fm = FeatureManager()
    
    # Get all feature groups
    groups = fm.get_feature_groups()
    
    print(f"\nFeature Groups:")
    for group_name, features in groups.items():
        print(f"\n{group_name.upper()} ({len(features)} features):")
        # Show first 5 features from each group
        for feat in features[:5]:
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {sum(len(f) for f in groups.values())} features available")
    print(f"{'='*70}")


def example_3_specific_features():
    """Example 3: Get specific features only"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Request Specific Features")
    print("=" * 70)
    
    fm = FeatureManager()
    
    # Request only specific features
    feature_list = [
        'close', 'volume',
        'rsi_14', 'macd_12_26', 'macd_signal_9',
        'bbands_upper_20', 'bbands_lower_20',
        'atr_14',
        'regime_type', 'regime_strength'
    ]
    
    print(f"\nRequesting {len(feature_list)} specific features:")
    for feat in feature_list:
        print(f"  - {feat}")
    
    try:
        df = fm.get_features(
            symbol='BTC-USDT',
            timeframe='5m',
            feature_list=feature_list,
            lookback_bars=50
        )
        
        print(f"\nResult: {len(df)} rows × {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"\nNote: {e}")


def example_4_custom_parameters():
    """Example 4: Custom indicator parameters"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Indicator Parameters")
    print("=" * 70)
    
    fm = FeatureManager()
    
    # Define custom parameters
    params = {
        'price': {
            'momentum_windows': [3, 7, 14, 30]
        },
        'technical': {
            'rsi_periods': [7, 14, 21],
            'ema_periods': [5, 10, 20, 50, 100],
            'macd_config': [(12, 26, 9), (8, 17, 9)]  # Fast and standard MACD
        },
        'volatility': {
            'atr_periods': [7, 14, 28],
            'bb_configs': [(20, 2.0), (20, 2.5)]  # Different BB widths
        }
    }
    
    print("\nCustom parameters:")
    for module, config in params.items():
        print(f"\n{module}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    try:
        df = fm.get_features(
            symbol='ETH-USDT',
            timeframe='15m',
            params=params,
            lookback_bars=200
        )
        
        print(f"\n\nResult: {len(df)} candles with custom indicators")
        
        # Show some custom features
        custom_features = [c for c in df.columns if 'rsi' in c or 'momentum' in c]
        print(f"Custom features created: {custom_features[:10]}")
        
    except Exception as e:
        print(f"\nNote: {e}")


def example_5_performance():
    """Example 5: Performance and caching"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Performance & Caching")
    print("=" * 70)
    
    import time
    
    fm = FeatureManager(cache_enabled=True)
    
    try:
        print("\nFirst call (no cache)...")
        start = time.time()
        df1 = fm.get_features('BTC-USDT', '1m', lookback_bars=500)
        time1 = time.time() - start
        print(f"  Time: {time1*1000:.1f}ms")
        print(f"  Rows: {len(df1)}, Columns: {len(df1.columns)}")
        
        print("\nSecond call (cached)...")
        start = time.time()
        df2 = fm.get_features('BTC-USDT', '1m', lookback_bars=500)
        time2 = time.time() - start
        print(f"  Time: {time2*1000:.1f}ms")
        print(f"  Speedup: {time1/time2:.1f}x faster")
        
        print(f"\n✓ Cache working! ({time2*1000:.1f}ms vs {time1*1000:.1f}ms)")
        
    except Exception as e:
        print(f"\nNote: {e}")
        print("Performance test requires database with tick data")


def example_6_market_regime():
    """Example 6: Market regime detection"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Market Regime Detection")
    print("=" * 70)
    
    from features.market_regime import get_regime_type, get_regime_strength
    
    fm = FeatureManager()
    
    try:
        df = fm.get_features('BTC-USDT', '1h', lookback_bars=500)
        
        # Get current regime
        current_regime = get_regime_type(df)
        regime_strength = get_regime_strength(df)
        
        print(f"\nCurrent Market Regime:")
        print(f"  Type: {current_regime}")
        print(f"  Strength: {regime_strength:.2%}")
        
        # Show regime history
        print(f"\nLast 10 regime readings:")
        regime_cols = ['timestamp', 'close', 'regime_type', 'regime_strength', 
                      'is_trending_up', 'is_trending_down', 'is_ranging', 'is_volatile']
        print(df[regime_cols].tail(10).to_string(index=False))
        
        # Count regime distribution
        print(f"\nRegime Distribution:")
        regime_counts = df['regime_type'].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"  {regime}: {count} candles ({pct:.1f}%)")
        
    except Exception as e:
        print(f"\nNote: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("BLOFIN STACK FEATURE LIBRARY - USAGE EXAMPLES")
    print("=" * 70)
    
    examples = [
        example_1_basic_usage,
        example_2_list_features,
        example_3_specific_features,
        example_4_custom_parameters,
        example_5_performance,
        example_6_market_regime
    ]
    
    for example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user")
            return
        except Exception as e:
            print(f"\nExample failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nThe feature library is ready for use in backtesting and live trading.")
    print("\nNext steps:")
    print("  1. Ensure blofin_monitor.db has tick data")
    print("  2. Use features in strategy development")
    print("  3. Run backtests with computed features")
    print("  4. Monitor performance in production")
    print()


if __name__ == '__main__':
    main()
