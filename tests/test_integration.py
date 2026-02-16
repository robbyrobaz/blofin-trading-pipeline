#!/usr/bin/env python3
"""
Integration tests for the full pipeline.
"""
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    try:
        import db
        import features
        # TODO: Add more imports as modules are ready
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_database_init():
    """Test database initialization."""
    import db
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.db') as f:
        con = db.connect(f.name)
        db.init_db(con)
        
        # Verify tables exist
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        
        assert 'strategy_backtest_results' in table_names
        assert 'ml_model_results' in table_names
        assert 'ml_ensembles' in table_names
        assert 'daily_reports' in table_names
        
        con.close()


def test_feature_computation():
    """Test feature computation pipeline."""
    # TODO: Implement when features module is ready
    pass


def test_backtest_execution():
    """Test strategy backtest execution."""
    # TODO: Implement when backtester is ready
    pass


def test_ml_training():
    """Test ML model training."""
    # TODO: Implement when ml_pipeline is ready
    pass


def test_end_to_end_pipeline():
    """Test complete end-to-end workflow."""
    # TODO: Implement full pipeline test
    pass
