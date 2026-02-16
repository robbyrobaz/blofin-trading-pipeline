#!/usr/bin/env python3
"""
Comprehensive tests for backtester module.
Tests aggregation, metrics, and backtest engine.
"""

import unittest
import numpy as np
import tempfile
import sqlite3
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backtester.aggregator import (
    aggregate_ohlcv,
    aggregate_ticks_to_1m_ohlcv,
    fast_aggregate_numpy,
    timeframe_to_minutes
)
from backtester.metrics import (
    calculate_win_rate,
    calculate_avg_pnl_pct,
    calculate_total_pnl_pct,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_score,
    calculate_all_metrics
)
from backtester.backtest_engine import BacktestEngine


class TestAggregator(unittest.TestCase):
    """Test OHLCV aggregation functions."""
    
    def test_timeframe_to_minutes(self):
        """Test timeframe string conversion."""
        self.assertEqual(timeframe_to_minutes('1m'), 1)
        self.assertEqual(timeframe_to_minutes('5m'), 5)
        self.assertEqual(timeframe_to_minutes('15m'), 15)
        self.assertEqual(timeframe_to_minutes('1h'), 60)
        self.assertEqual(timeframe_to_minutes('4h'), 240)
        self.assertEqual(timeframe_to_minutes('1d'), 1440)
    
    def test_aggregate_ohlcv_basic(self):
        """Test basic 1m -> 5m aggregation."""
        # Create 5 one-minute candles
        base_ts = int(datetime(2024, 1, 1, 12, 0).timestamp() * 1000)
        candles_1m = []
        
        for i in range(5):
            candles_1m.append({
                'ts_ms': base_ts + i * 60 * 1000,
                'open': 100 + i,
                'high': 105 + i,
                'low': 95 + i,
                'close': 102 + i,
                'volume': 10 + i
            })
        
        # Aggregate to 5m
        result = aggregate_ohlcv(candles_1m, 5)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['open'], 100)  # First open
        self.assertEqual(result[0]['high'], 109)  # Max high (105 + 4)
        self.assertEqual(result[0]['low'], 95)    # Min low
        self.assertEqual(result[0]['close'], 106)  # Last close (102 + 4)
        self.assertEqual(result[0]['volume'], 60)  # Sum (10+11+12+13+14)
    
    def test_aggregate_ticks_to_1m(self):
        """Test tick -> 1m OHLCV aggregation."""
        base_ts = int(datetime(2024, 1, 1, 12, 0).timestamp() * 1000)
        ticks = [
            {'ts_ms': base_ts + 1000, 'price': 100},
            {'ts_ms': base_ts + 15000, 'price': 102},
            {'ts_ms': base_ts + 30000, 'price': 99},
            {'ts_ms': base_ts + 45000, 'price': 101},
        ]
        
        result = aggregate_ticks_to_1m_ohlcv(ticks)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['open'], 100)
        self.assertEqual(result[0]['high'], 102)
        self.assertEqual(result[0]['low'], 99)
        self.assertEqual(result[0]['close'], 101)
    
    def test_fast_aggregate_numpy(self):
        """Test numpy-based fast aggregation."""
        # Create 10 candles (2 groups of 5)
        base_ts = int(datetime(2024, 1, 1, 12, 0).timestamp() * 1000)
        candles = []
        
        for i in range(10):
            candles.append([
                base_ts + i * 60 * 1000,  # ts_ms
                100 + i,  # open
                105 + i,  # high
                95 + i,   # low
                102 + i,  # close
                10 + i    # volume
            ])
        
        candles_array = np.array(candles)
        result = fast_aggregate_numpy(candles_array, 5)
        
        self.assertEqual(len(result), 2)
        # First group
        self.assertEqual(result[0][1], 100)  # First open
        self.assertEqual(result[0][4], 106)  # Last close of group
        # Second group
        self.assertEqual(result[1][1], 105)  # First open of second group


class TestMetrics(unittest.TestCase):
    """Test performance metrics calculations."""
    
    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        trades = [
            {'pnl_pct': 2.5},
            {'pnl_pct': -1.0},
            {'pnl_pct': 3.0},
            {'pnl_pct': -0.5},
        ]
        
        win_rate = calculate_win_rate(trades)
        self.assertEqual(win_rate, 0.5)  # 2 wins, 2 losses
    
    def test_calculate_avg_pnl_pct(self):
        """Test average P&L calculation."""
        trades = [
            {'pnl_pct': 2.0},
            {'pnl_pct': -1.0},
            {'pnl_pct': 3.0},
            {'pnl_pct': -1.0},
        ]
        
        avg = calculate_avg_pnl_pct(trades)
        self.assertEqual(avg, 0.75)  # (2 - 1 + 3 - 1) / 4
    
    def test_calculate_total_pnl_pct(self):
        """Test total P&L calculation."""
        trades = [
            {'pnl_pct': 2.0},
            {'pnl_pct': -1.0},
            {'pnl_pct': 3.0},
        ]
        
        total = calculate_total_pnl_pct(trades)
        self.assertEqual(total, 4.0)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Volatile returns (positive expected value)
        returns = [0.05, -0.04, 0.03, -0.02, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        # Should be calculable (non-zero volatility)
        self.assertIsInstance(sharpe, float)
        
        # Consistent positive returns (zero volatility edge case)
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        # Should handle zero volatility gracefully
        self.assertEqual(sharpe, 0.0)
    
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        # Equity curve with clear drawdown
        equity = [100, 110, 105, 95, 100, 115]
        
        max_dd = calculate_max_drawdown(equity)
        # Max was 110, min after was 95, so drawdown = (110-95)/110 * 100 = 13.64%
        self.assertAlmostEqual(max_dd, 13.636363636363637, places=2)
    
    def test_calculate_score(self):
        """Test composite score calculation."""
        metrics = {
            'win_rate': 0.6,           # 40 * 0.6 = 24
            'avg_pnl_pct': 1.5,        # 30 * 1.5 = 45
            'sharpe_ratio': 1.0,       # 20 * 1.0 = 20
            'max_drawdown_pct': 5.0    # 10 * 5.0 = 50 (penalty)
        }
        
        score = calculate_score(metrics)
        expected = 24 + 45 + 20 - 50
        self.assertEqual(score, max(0, expected))  # Clamped to 0
    
    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        trades = [
            {'pnl_pct': 2.0},
            {'pnl_pct': -1.0},
            {'pnl_pct': 3.0},
        ]
        
        equity = [100, 102, 101, 104]
        
        metrics = calculate_all_metrics(trades, equity)
        
        self.assertIn('win_rate', metrics)
        self.assertIn('avg_pnl_pct', metrics)
        self.assertIn('total_pnl_pct', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_pct', metrics)
        self.assertIn('score', metrics)
        self.assertIn('num_trades', metrics)
        
        self.assertEqual(metrics['num_trades'], 3)


class TestBacktestEngine(unittest.TestCase):
    """Test backtest engine with mock data."""
    
    def setUp(self):
        """Create temporary database with test data."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        
        # Create database and insert test data
        con = sqlite3.connect(self.db_path)
        con.execute('''
            CREATE TABLE ticks (
                ts_ms INTEGER,
                symbol TEXT,
                price REAL
            )
        ''')
        
        # Insert 7 days of mock tick data (1 tick per minute)
        base_ts = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        symbol = 'BTC-USDT'
        
        for i in range(7 * 24 * 60):  # 7 days of minutes
            price = 50000 + (i % 100) * 10  # Oscillating price
            con.execute(
                'INSERT INTO ticks (ts_ms, symbol, price) VALUES (?, ?, ?)',
                (base_ts + i * 60 * 1000, symbol, price)
            )
        
        con.commit()
        con.close()
    
    def tearDown(self):
        """Clean up temporary database."""
        import os
        os.unlink(self.db_path)
    
    def test_engine_initialization(self):
        """Test BacktestEngine initializes correctly."""
        engine = BacktestEngine(
            symbol='BTC-USDT',
            days_back=7,
            db_path=self.db_path
        )
        
        self.assertEqual(engine.symbol, 'BTC-USDT')
        self.assertGreater(len(engine.ticks), 0)
        self.assertGreater(len(engine.ohlcv_1m), 0)
    
    def test_get_ohlcv_different_timeframes(self):
        """Test OHLCV retrieval for different timeframes."""
        engine = BacktestEngine(
            symbol='BTC-USDT',
            days_back=7,
            db_path=self.db_path
        )
        
        ohlcv_1m = engine.get_ohlcv('1m')
        ohlcv_5m = engine.get_ohlcv('5m')
        ohlcv_60m = engine.get_ohlcv('1h')
        
        # 5m should have ~1/5 candles of 1m
        self.assertGreater(len(ohlcv_1m), len(ohlcv_5m))
        self.assertGreater(len(ohlcv_5m), len(ohlcv_60m))
    
    def test_run_strategy_basic(self):
        """Test strategy backtesting with mock strategy."""
        
        class MockStrategy:
            """Simple mock strategy that buys every 10 candles."""
            def __init__(self):
                self.call_count = 0
            
            def detect(self, candles, symbol):
                self.call_count += 1
                # Buy every 10th candle, sell every 20th
                if self.call_count % 20 == 10:
                    return {'signal': 'BUY'}
                elif self.call_count % 20 == 0:
                    return {'signal': 'SELL'}
                return None
        
        engine = BacktestEngine(
            symbol='BTC-USDT',
            days_back=7,
            db_path=self.db_path
        )
        
        strategy = MockStrategy()
        results = engine.run_strategy(strategy, timeframe='5m')
        
        self.assertIn('trades', results)
        self.assertIn('equity_curve', results)
        self.assertIn('metrics', results)
        self.assertGreater(len(results['equity_curve']), 0)
    
    def test_run_model_basic(self):
        """Test ML model backtesting with mock model."""
        
        class MockModel:
            """Simple mock model that predicts based on price."""
            def predict(self, candles, symbol):
                # Predict 1 if last candle's close > open, else 0
                if candles[-1]['close'] > candles[-1]['open']:
                    return 0.8
                return 0.2
        
        engine = BacktestEngine(
            symbol='BTC-USDT',
            days_back=7,
            db_path=self.db_path
        )
        
        model = MockModel()
        results = engine.run_model(model, timeframe='1m')
        
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
        self.assertGreater(results['num_predictions'], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_trades_metrics(self):
        """Test metrics calculation with no trades."""
        trades = []
        equity = [100]
        
        metrics = calculate_all_metrics(trades, equity)
        
        self.assertEqual(metrics['win_rate'], 0.0)
        self.assertEqual(metrics['num_trades'], 0)
    
    def test_zero_volatility_sharpe(self):
        """Test Sharpe ratio with zero volatility."""
        returns = [0.01, 0.01, 0.01, 0.01]  # Identical returns
        
        sharpe = calculate_sharpe_ratio(returns)
        # Should handle division by zero gracefully
        self.assertEqual(sharpe, 0.0)
    
    def test_single_trade_drawdown(self):
        """Test drawdown calculation with minimal data."""
        equity = [100, 95]
        
        max_dd = calculate_max_drawdown(equity)
        self.assertAlmostEqual(max_dd, 5.0, places=2)


if __name__ == '__main__':
    unittest.main()
