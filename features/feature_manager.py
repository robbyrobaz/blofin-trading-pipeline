"""
Central Feature Manager API

Provides unified interface for feature computation with caching.
"""
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from . import price_features
from . import volume_features
from . import technical_indicators
from . import volatility_features
from . import market_regime


class FeatureManager:
    """
    Central API for feature computation and management.
    
    Features are grouped into categories:
    - price: Basic price features (close, returns, momentum, etc.)
    - volume: Volume-based features (VWAP, OBV, etc.)
    - technical: Technical indicators (RSI, MACD, etc.)
    - volatility: Volatility measures (ATR, Bollinger Bands, etc.)
    - regime: Market regime detection
    """
    
    def __init__(self, db_path: str = None, cache_enabled: bool = True):
        """
        Initialize FeatureManager.
        
        Args:
            db_path: Path to blofin_monitor.db (defaults to ../data/blofin_monitor.db)
            cache_enabled: Enable feature caching for performance
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "blofin_monitor.db")
        
        self.db_path = db_path
        self.cache_enabled = cache_enabled
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 60  # Cache TTL in seconds
        
        # Feature group mapping
        self._feature_groups = {
            'price': price_features,
            'volume': volume_features,
            'technical': technical_indicators,
            'volatility': volatility_features,
            'regime': market_regime
        }
    
    def _load_ohlcv_from_ticks(self, symbol: str, timeframe: str = '1m', 
                                lookback_bars: int = 500) -> pd.DataFrame:
        """
        Load and aggregate tick data into OHLCV candles.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', etc.)
            lookback_bars: Number of candles to load
            
        Returns:
            DataFrame with OHLCV data
        """
        # Parse timeframe
        timeframe_seconds = self._parse_timeframe(timeframe)
        
        # Calculate time range
        end_time_ms = int(time.time() * 1000)
        start_time_ms = end_time_ms - (lookback_bars * timeframe_seconds * 1000)
        
        # Load ticks from database
        con = sqlite3.connect(self.db_path)
        query = """
            SELECT ts_ms, price, 
                   COALESCE(
                       CAST(json_extract(raw_json, '$.data[0].vol24h') AS REAL),
                       CAST(json_extract(raw_json, '$.vol24h') AS REAL),
                       0
                   ) as volume
            FROM ticks 
            WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ?
            ORDER BY ts_ms ASC
        """
        df = pd.read_sql_query(query, con, params=(symbol, start_time_ms, end_time_ms))
        con.close()
        
        if df.empty:
            raise ValueError(f"No tick data found for {symbol} in the specified time range")
        
        # Clean data BEFORE any conversions
        # 1. Convert types explicitly
        df['ts_ms'] = pd.to_numeric(df['ts_ms'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # 2. Drop rows with NaN in critical columns
        df = df.dropna(subset=['ts_ms', 'price'])
        
        # 3. Ensure ts_ms is integer type (required for pd.to_datetime with unit='ms')
        df['ts_ms'] = df['ts_ms'].astype('int64')
        
        # NOW safe to convert timestamp
        df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Aggregate into OHLCV candles
        ohlcv = df.resample(f'{timeframe_seconds}s').agg({
            'price': ['first', 'max', 'min', 'last', 'count'],
            'volume': 'sum'
        })
        
        # Flatten column names
        ohlcv.columns = ['open', 'high', 'low', 'close', 'tick_count', 'volume']
        
        # Forward fill missing candles (if no ticks in that period)
        ohlcv['close'] = ohlcv['close'].ffill()
        ohlcv['open'] = ohlcv['open'].fillna(ohlcv['close'])
        ohlcv['high'] = ohlcv['high'].fillna(ohlcv['close'])
        ohlcv['low'] = ohlcv['low'].fillna(ohlcv['close'])
        ohlcv['volume'] = ohlcv['volume'].fillna(0)
        
        # Reset index to make timestamp a column
        ohlcv.reset_index(inplace=True)
        
        # Keep only the last N bars
        ohlcv = ohlcv.tail(lookback_bars)
        
        return ohlcv
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        if timeframe in timeframe_map:
            return timeframe_map[timeframe]
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def _get_cache_key(self, symbol: str, timeframe: str, lookback_bars: int) -> str:
        """Generate cache key for feature data."""
        return f"{symbol}_{timeframe}_{lookback_bars}"
    
    def _check_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Check if cached features exist and are still valid."""
        if not self.cache_enabled:
            return None
        
        if cache_key in self._feature_cache:
            age = time.time() - self._cache_timestamps[cache_key]
            if age < self.cache_ttl:
                return self._feature_cache[cache_key].copy()
            else:
                # Cache expired, remove it
                del self._feature_cache[cache_key]
                del self._cache_timestamps[cache_key]
        
        return None
    
    def _update_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Update feature cache."""
        if self.cache_enabled:
            self._feature_cache[cache_key] = df.copy()
            self._cache_timestamps[cache_key] = time.time()
    
    def get_features(self, symbol: str, timeframe: str = '1m', 
                    feature_list: List[str] = None, lookback_bars: int = 500,
                    params: Dict[str, Any] = None, limit_rows: int = None) -> pd.DataFrame:
        """
        Compute features for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', etc.)
            feature_list: List of specific features to compute (None = all features)
            lookback_bars: Number of candles to load
            params: Optional parameters for feature computation
            limit_rows: (Smoke test) Limit raw data to N rows from database
            
        Returns:
            DataFrame with OHLCV data and computed features
        """
        cache_key = self._get_cache_key(symbol, timeframe, lookback_bars)
        
        # Check cache
        cached = self._check_cache(cache_key)
        if cached is not None and feature_list is None:
            return cached
        
        # Load OHLCV data
        ohlcv = self._load_ohlcv_from_ticks(symbol, timeframe, lookback_bars)
        
        # Start with OHLCV as base
        result = ohlcv.copy()
        
        # Determine which feature groups to compute
        if feature_list is None:
            # Compute all features
            groups_to_compute = list(self._feature_groups.keys())
        else:
            # Determine which groups are needed based on requested features
            groups_to_compute = set()
            for feature in feature_list:
                group = self._get_feature_group(feature)
                if group:
                    groups_to_compute.add(group)
            groups_to_compute = list(groups_to_compute)
        
        # Compute features by group
        params = params or {}
        
        if 'price' in groups_to_compute:
            price_df = price_features.compute_price_features(ohlcv, params.get('price', {}))
            result = self._merge_features(result, price_df)
        
        if 'volume' in groups_to_compute:
            volume_df = volume_features.compute_volume_features(ohlcv, params.get('volume', {}))
            result = self._merge_features(result, volume_df)
        
        if 'technical' in groups_to_compute:
            tech_df = technical_indicators.compute_technical_indicators(ohlcv, params.get('technical', {}))
            result = self._merge_features(result, tech_df)
        
        if 'volatility' in groups_to_compute:
            vol_df = volatility_features.compute_volatility_features(ohlcv, params.get('volatility', {}))
            result = self._merge_features(result, vol_df)
        
        if 'regime' in groups_to_compute:
            regime_df = market_regime.compute_regime_features(ohlcv, params.get('regime', {}))
            result = self._merge_features(result, regime_df)
        
        # Filter to requested features if specified
        if feature_list is not None:
            # Keep OHLCV columns + requested features
            base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            cols_to_keep = base_cols + [f for f in feature_list if f in result.columns]
            result = result[cols_to_keep]
        
        # Update cache
        self._update_cache(cache_key, result)
        
        return result
    
    def _merge_features(self, base_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Merge feature dataframe with base dataframe."""
        # Get columns that aren't already in base_df
        new_cols = [col for col in feature_df.columns if col not in base_df.columns]
        
        # Merge only new columns
        for col in new_cols:
            base_df[col] = feature_df[col]
        
        return base_df
    
    def _get_feature_group(self, feature_name: str) -> Optional[str]:
        """Determine which group a feature belongs to."""
        for group_name, module in self._feature_groups.items():
            if hasattr(module, 'get_available_features'):
                if feature_name in module.get_available_features():
                    return group_name
        return None
    
    def list_available_features(self, group: str = None) -> List[str]:
        """
        List all available features.
        
        Args:
            group: Optional group name to filter by ('price', 'volume', 'technical', 'volatility', 'regime')
            
        Returns:
            List of feature names
        """
        if group:
            if group not in self._feature_groups:
                raise ValueError(f"Unknown feature group: {group}")
            module = self._feature_groups[group]
            return module.get_available_features()
        else:
            # Return all features from all groups
            all_features = []
            for module in self._feature_groups.values():
                all_features.extend(module.get_available_features())
            return all_features
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get all feature groups and their features.
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        return {
            group_name: module.get_available_features()
            for group_name, module in self._feature_groups.items()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached features."""
        self._feature_cache.clear()
        self._cache_timestamps.clear()
