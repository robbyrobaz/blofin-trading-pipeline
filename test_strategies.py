#!/usr/bin/env python3
"""
Quick smoke test for the strategy plugin system.
"""

import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def test_import_strategies():
    """Test that all strategies can be imported."""
    print("Testing strategy imports...")
    try:
        from strategies import get_all_strategies
        strategies = get_all_strategies()
        print(f"✓ Successfully loaded {len(strategies)} strategies:")
        for s in strategies:
            print(f"  - {s.name} (v{s.version}): {s.description}")
        return True
    except Exception as e:
        print(f"✗ Failed to import strategies: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_manager():
    """Test StrategyManager initialization."""
    print("\nTesting StrategyManager...")
    try:
        from strategy_manager import StrategyManager
        manager = StrategyManager()
        print(f"✓ StrategyManager initialized with {len(manager.strategies)} strategies")
        print(f"  Enabled: {sum(1 for e in manager.enabled.values() if e)}")
        print(f"  Disabled: {sum(1 for e in manager.enabled.values() if not e)}")
        return True
    except Exception as e:
        print(f"✗ StrategyManager failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_base():
    """Test knowledge base functions."""
    print("\nTesting knowledge_base...")
    try:
        import knowledge_base
        from db import connect, init_db
        
        # Use in-memory database for testing
        con = connect(':memory:')
        init_db(con)
        
        # Test adding an entry
        entry_id = knowledge_base.add_knowledge_entry(
            con,
            category='test',
            content='Test entry',
            source='test'
        )
        print(f"✓ Added knowledge entry: {entry_id}")
        
        # Test getting summary
        summary = knowledge_base.get_knowledge_summary(con)
        print(f"✓ Generated knowledge summary ({len(summary)} chars)")
        
        con.close()
        return True
    except Exception as e:
        print(f"✗ knowledge_base failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_detection():
    """Test signal detection with mock data."""
    print("\nTesting signal detection...")
    try:
        from strategy_manager import StrategyManager
        
        manager = StrategyManager()
        
        # Mock price data (upward trend)
        base_price = 50000.0
        prices = [(i * 1000, base_price + (i * 50)) for i in range(100)]
        volumes = [(i * 1000, 1000.0) for i in range(100)]
        
        signals = manager.detect_all(
            symbol='BTC-USDT',
            price=prices[-1][1],
            volume=volumes[-1][1],
            ts_ms=prices[-1][0],
            prices=prices,
            volumes=volumes
        )
        
        print(f"✓ Generated {len(signals)} signals from mock data")
        for sig in signals:
            print(f"  - {sig.strategy}: {sig.signal} (confidence: {sig.confidence:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Signal detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== Blofin Strategy Plugin System Tests ===\n")
    
    tests = [
        test_import_strategies,
        test_strategy_manager,
        test_knowledge_base,
        test_signal_detection,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
