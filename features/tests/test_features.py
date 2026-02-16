"""
Comprehensive tests for the feature library.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.feature_manager import FeatureManager
from features import price_features, volume_features, technical_indicators
from features import volatility_features, market_regime


class TestPriceFeatures(unittest.TestCase):
    """Test price-based features."""
    
    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        
        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0, 0.01, n)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
    
    def test_basic_price_features(self):
        """Test basic price level features."""
        result = price_features.compute_price_features(self.df)
        
        # Check that basic features exist
        self.assertIn('close', result.columns)
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('hl2', result.columns)
        self.assertIn('hlc3', result.columns)
        
        # Verify calculations
        self.assertTrue(np.allclose(result['hl2'], (self.df['high'] + self.df['low']) / 2))
        self.assertTrue(np.allclose(
            result['hlc3'], 
            (self.df['high'] + self.df['low'] + self.df['close']) / 3
        ))
    
    def test_returns(self):
        """Test return calculations."""
        result = price_features.compute_price_features(self.df)
        
        self.assertIn('returns', result.columns)
        self.assertIn('log_returns', result.columns)
        
        # First value should be NaN
        self.assertTrue(pd.isna(result['returns'].iloc[0]))
        
        # Returns should be percentage change
        expected_return = (self.df['close'].iloc[10] - self.df['close'].iloc[9]) / self.df['close'].iloc[9]
        self.assertAlmostEqual(result['returns'].iloc[10], expected_return, places=6)
    
    def test_momentum(self):
        """Test momentum features."""
        result = price_features.compute_price_features(self.df)
        
        # Check momentum features exist
        self.assertIn('momentum_1', result.columns)
        self.assertIn('momentum_5', result.columns)
        self.assertIn('roc_1', result.columns)
        
        # Verify momentum calculation
        expected_momentum = self.df['close'].iloc[50] - self.df['close'].iloc[45]
        self.assertAlmostEqual(result['momentum_5'].iloc[50], expected_momentum, places=2)


class TestVolumeFeatures(unittest.TestCase):
    """Test volume-based features."""
    
    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0, 0.01, n)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
    
    def test_volume_moving_averages(self):
        """Test volume SMA and EMA."""
        result = volume_features.compute_volume_features(self.df)
        
        self.assertIn('volume_sma_20', result.columns)
        self.assertIn('volume_ema_20', result.columns)
        
        # Verify SMA calculation at index 30
        expected_sma = self.df['volume'].iloc[11:31].mean()
        self.assertAlmostEqual(result['volume_sma_20'].iloc[30], expected_sma, places=2)
    
    def test_vwap(self):
        """Test VWAP calculation."""
        result = volume_features.compute_volume_features(self.df)
        
        self.assertIn('vwap', result.columns)
        self.assertIn('vwap_20', result.columns)
        self.assertIn('vwap_deviation_20', result.columns)
        
        # VWAP should be close to typical price
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Check that VWAP is in reasonable range
        self.assertTrue(result['vwap'].iloc[-1] > self.df['low'].min())
        self.assertTrue(result['vwap'].iloc[-1] < self.df['high'].max())
    
    def test_obv(self):
        """Test On-Balance Volume."""
        result = volume_features.compute_volume_features(self.df)
        
        self.assertIn('obv', result.columns)
        
        # OBV should start at 0
        self.assertEqual(result['obv'].iloc[0], 0)
        
        # OBV should be cumulative
        self.assertTrue(abs(result['obv'].iloc[-1]) > 0)


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators."""
    
    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200  # Need more data for technical indicators
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0, 0.01, n)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
    
    def test_rsi(self):
        """Test RSI calculation."""
        result = technical_indicators.compute_technical_indicators(self.df)
        
        self.assertIn('rsi_14', result.columns)
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi_14'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())
    
    def test_macd(self):
        """Test MACD calculation."""
        result = technical_indicators.compute_technical_indicators(self.df)
        
        self.assertIn('macd_12_26', result.columns)
        self.assertIn('macd_signal_9', result.columns)
        self.assertIn('macd_histogram', result.columns)
        
        # Verify histogram = MACD - Signal
        hist_calc = result['macd_12_26'] - result['macd_signal_9']
        self.assertTrue(np.allclose(
            result['macd_histogram'].dropna(), 
            hist_calc.dropna(), 
            rtol=1e-5
        ))
    
    def test_stochastic(self):
        """Test Stochastic Oscillator."""
        result = technical_indicators.compute_technical_indicators(self.df)
        
        self.assertIn('stoch_k', result.columns)
        self.assertIn('stoch_d', result.columns)
        
        # Stochastic should be between 0 and 100
        stoch_k = result['stoch_k'].dropna()
        self.assertTrue((stoch_k >= 0).all())
        self.assertTrue((stoch_k <= 100).all())
    
    def test_moving_averages(self):
        """Test SMA and EMA."""
        result = technical_indicators.compute_technical_indicators(self.df)
        
        self.assertIn('sma_50', result.columns)
        self.assertIn('ema_50', result.columns)
        
        # Verify SMA calculation
        expected_sma = self.df['close'].iloc[100:150].mean()
        self.assertAlmostEqual(result['sma_50'].iloc[149], expected_sma, places=2)
    
    def test_adx(self):
        """Test ADX calculation."""
        result = technical_indicators.compute_technical_indicators(self.df)
        
        self.assertIn('adx_14', result.columns)
        self.assertIn('plus_di_14', result.columns)
        self.assertIn('minus_di_14', result.columns)
        
        # ADX should be between 0 and 100
        adx_values = result['adx_14'].dropna()
        self.assertTrue((adx_values >= 0).all())
        self.assertTrue((adx_values <= 100).all())


class TestVolatilityFeatures(unittest.TestCase):
    """Test volatility features."""
    
    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0, 0.01, n)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
    
    def test_atr(self):
        """Test ATR calculation."""
        result = volatility_features.compute_volatility_features(self.df)
        
        self.assertIn('atr_14', result.columns)
        self.assertIn('atr_14_pct', result.columns)
        
        # ATR should be positive
        atr_values = result['atr_14'].dropna()
        self.assertTrue((atr_values > 0).all())
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands."""
        result = volatility_features.compute_volatility_features(self.df)
        
        self.assertIn('bbands_upper_20', result.columns)
        self.assertIn('bbands_middle_20', result.columns)
        self.assertIn('bbands_lower_20', result.columns)
        self.assertIn('bbands_width_20', result.columns)
        
        # Upper should be > Middle > Lower
        valid_idx = result['bbands_upper_20'].notna()
        self.assertTrue((
            result.loc[valid_idx, 'bbands_upper_20'] > 
            result.loc[valid_idx, 'bbands_middle_20']
        ).all())
        self.assertTrue((
            result.loc[valid_idx, 'bbands_middle_20'] > 
            result.loc[valid_idx, 'bbands_lower_20']
        ).all())
    
    def test_keltner_channels(self):
        """Test Keltner Channels."""
        result = volatility_features.compute_volatility_features(self.df)
        
        self.assertIn('keltner_upper_20', result.columns)
        self.assertIn('keltner_middle_20', result.columns)
        self.assertIn('keltner_lower_20', result.columns)
        
        # Upper should be > Lower
        valid_idx = result['keltner_upper_20'].notna()
        self.assertTrue((
            result.loc[valid_idx, 'keltner_upper_20'] > 
            result.loc[valid_idx, 'keltner_lower_20']
        ).all())


class TestMarketRegime(unittest.TestCase):
    """Test market regime detection."""
    
    def test_trending_up_detection(self):
        """Test uptrend detection."""
        # Create clear uptrend
        n = 100
        prices = np.linspace(50000, 60000, n) + np.random.normal(0, 100, n)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
        
        result = market_regime.compute_regime_features(df)
        
        # Last value should detect uptrend
        self.assertTrue(result['is_trending_up'].iloc[-1])
        self.assertEqual(result['regime_type'].iloc[-1], 'trending_up')
    
    def test_ranging_detection(self):
        """Test ranging market detection."""
        # Create ranging market with tighter range
        n = 100
        prices = 50000 + np.random.normal(0, 100, n)  # Very tight range (0.2%)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * 0.999,
            'high': prices * 1.0005,
            'low': prices * 0.9995,
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
        
        # Use custom params for tighter range detection
        params = {'ranging_threshold_pct': 5.0}  # More lenient threshold
        result = market_regime.compute_regime_features(df, params)
        
        # Should detect ranging or at least not trending strongly
        self.assertTrue(result['is_ranging'].iloc[-1] or 
                       (not result['is_trending_up'].iloc[-1] and 
                        not result['is_trending_down'].iloc[-1]))
    
    def test_volatile_detection(self):
        """Test volatile market detection."""
        # Create volatile market
        n = 100
        prices = 50000 + np.random.normal(0, 2000, n)  # High volatility
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices * 0.999,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
        
        result = market_regime.compute_regime_features(df)
        
        # Should detect high volatility
        self.assertTrue(result['is_volatile'].iloc[-1])


class TestFeatureManager(unittest.TestCase):
    """Test FeatureManager integration."""
    
    def setUp(self):
        """Set up test database with sample data."""
        # Note: This test will skip if database doesn't exist
        db_path = Path(__file__).parent.parent.parent / "data" / "blofin_monitor.db"
        
        if not db_path.exists():
            self.skipTest("Database not found - skipping integration tests")
        
        self.fm = FeatureManager(str(db_path))
    
    def test_list_features(self):
        """Test listing available features."""
        features = self.fm.list_available_features()
        
        # Should have features from all groups
        self.assertGreater(len(features), 50)
        
        # Check specific features exist
        self.assertIn('rsi_14', features)
        self.assertIn('macd_12_26', features)
        self.assertIn('bbands_upper_20', features)
    
    def test_get_feature_groups(self):
        """Test getting feature groups."""
        groups = self.fm.get_feature_groups()
        
        # Should have all groups
        self.assertIn('price', groups)
        self.assertIn('volume', groups)
        self.assertIn('technical', groups)
        self.assertIn('volatility', groups)
        self.assertIn('regime', groups)
        
        # Each group should have features
        for group_name, features in groups.items():
            self.assertGreater(len(features), 0, f"Group {group_name} has no features")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_short_data(self):
        """Test with insufficient data."""
        # Only 10 bars
        n = 10
        prices = np.linspace(50000, 51000, n)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.ones(n) * 100
        })
        
        # Should not crash, but many features will be NaN
        result = price_features.compute_price_features(df)
        self.assertEqual(len(result), n)
    
    def test_missing_data(self):
        """Test handling of missing data."""
        n = 50
        prices = np.linspace(50000, 51000, n)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.ones(n) * 100
        })
        
        # Introduce some NaN values
        df.loc[20:25, 'close'] = np.nan
        
        # Should handle NaN gracefully
        result = price_features.compute_price_features(df)
        self.assertEqual(len(result), n)
    
    def test_zero_volume(self):
        """Test with zero volume periods."""
        n = 50
        prices = np.linspace(50000, 51000, n)
        volumes = np.ones(n) * 100
        volumes[20:25] = 0  # Zero volume period
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
        
        # Should not crash
        result = volume_features.compute_volume_features(df)
        self.assertEqual(len(result), n)


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPriceFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestVolumeFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestTechnicalIndicators))
    suite.addTests(loader.loadTestsFromTestCase(TestVolatilityFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketRegime))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureManager))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
