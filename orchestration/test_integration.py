#!/usr/bin/env python3
"""
Integration test suite for orchestration layer.
Tests all components without requiring AI model calls.
"""
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from orchestration import Ranker, DailyReporter, StrategyDesigner, StrategyTuner, DailyRunner
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_database_schema():
    """Test that database has required tables."""
    print("\nTesting database schema...")
    import sqlite3
    
    db_path = Path(__file__).parent.parent / 'data' / 'blofin_monitor.db'
    if not db_path.exists():
        print(f"✗ Database not found: {db_path}")
        return False
    
    con = sqlite3.connect(str(db_path))
    cursor = con.cursor()
    
    required_tables = [
        'strategy_backtest_results',
        'ml_model_results',
        'ml_ensembles',
        'daily_reports',
        'ranking_history'
    ]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = [row[0] for row in cursor.fetchall()]
    
    missing = [t for t in required_tables if t not in existing_tables]
    if missing:
        print(f"✗ Missing tables: {missing}")
        con.close()
        return False
    
    print(f"✓ All required tables exist: {required_tables}")
    con.close()
    return True


def test_ranker():
    """Test ranker component."""
    print("\nTesting ranker...")
    try:
        from orchestration import Ranker
        
        db_path = Path(__file__).parent.parent / 'data' / 'blofin_monitor.db'
        ranker = Ranker(str(db_path))
        
        # Test ranking
        top_strategies = ranker.keep_top_strategies(count=20)
        top_models = ranker.keep_top_models(count=5)
        top_ensembles = ranker.keep_top_ensembles(count=3)
        
        print(f"  - Top strategies: {len(top_strategies)}")
        print(f"  - Top models: {len(top_models)}")
        print(f"  - Top ensembles: {len(top_ensembles)}")
        print("✓ Ranker works")
        return True
    except Exception as e:
        print(f"✗ Ranker failed: {e}")
        return False


def test_reporter():
    """Test reporter component."""
    print("\nTesting reporter...")
    try:
        from orchestration import DailyReporter
        
        db_path = Path(__file__).parent.parent / 'data' / 'blofin_monitor.db'
        reports_dir = Path(__file__).parent.parent / 'data' / 'reports'
        
        reporter = DailyReporter(str(db_path), str(reports_dir))
        report = reporter.generate_report()
        
        print(f"  - Report date: {report['date']}")
        print(f"  - Strategies designed: {report['activity']['strategies_designed']}")
        print(f"  - Portfolio health: {report['portfolio_health']['health_status']}")
        print("✓ Reporter works")
        return True
    except Exception as e:
        print(f"✗ Reporter failed: {e}")
        return False


def test_designer():
    """Test strategy designer (dry run)."""
    print("\nTesting strategy designer...")
    try:
        from orchestration import StrategyDesigner
        
        db_path = Path(__file__).parent.parent / 'data' / 'blofin_monitor.db'
        strategies_dir = Path(__file__).parent.parent / 'strategies'
        
        designer = StrategyDesigner(str(db_path), str(strategies_dir))
        
        # Test analysis methods (don't actually call Opus)
        con = designer._connect()
        top = designer._get_top_performers(con, 5)
        bottom = designer._get_bottom_performers(con, 5)
        regime = designer._analyze_market_regime(con)
        gaps = designer._identify_gaps(con)
        con.close()
        
        print(f"  - Top performers: {len(top)}")
        print(f"  - Bottom performers: {len(bottom)}")
        print(f"  - Market regime: {regime['regime']}")
        print(f"  - Gaps found: {len(gaps)}")
        print("✓ Designer analysis works (Opus call not tested)")
        return True
    except Exception as e:
        print(f"✗ Designer failed: {e}")
        return False


def test_tuner():
    """Test strategy tuner (dry run)."""
    print("\nTesting strategy tuner...")
    try:
        from orchestration import StrategyTuner
        
        db_path = Path(__file__).parent.parent / 'data' / 'blofin_monitor.db'
        strategies_dir = Path(__file__).parent.parent / 'strategies'
        
        tuner = StrategyTuner(str(db_path), str(strategies_dir))
        
        # Test analysis methods (don't actually call Sonnet)
        con = tuner._connect()
        underperformers = tuner._get_underperformers(con, limit=3)
        con.close()
        
        print(f"  - Underperformers found: {len(underperformers)}")
        for u in underperformers[:3]:
            print(f"    - {u['strategy']}: score={u['score']:.3f}")
        print("✓ Tuner analysis works (Sonnet call not tested)")
        return True
    except Exception as e:
        print(f"✗ Tuner failed: {e}")
        return False


def test_daily_runner():
    """Test daily runner initialization."""
    print("\nTesting daily runner...")
    try:
        from orchestration import DailyRunner
        
        workspace_dir = Path(__file__).parent.parent
        runner = DailyRunner(str(workspace_dir))
        
        print(f"  - Workspace: {runner.workspace_dir}")
        print(f"  - DB path: {runner.db_path}")
        print(f"  - Log path: {runner.log_path}")
        print("✓ Daily runner initialized (full pipeline not tested)")
        return True
    except Exception as e:
        print(f"✗ Daily runner failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ORCHESTRATION INTEGRATION TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_database_schema,
        test_ranker,
        test_reporter,
        test_designer,
        test_tuner,
        test_daily_runner
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Orchestration layer is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
