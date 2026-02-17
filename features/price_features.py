"""
Price-based features for technical analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_price_features(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute basic price features from OHLCV dataframe.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Optional parameters dict (e.g., momentum windows)
        
    Returns:
        DataFrame with computed features
    """
    params = params or {}
    result = df.copy()
    
    # Basic price levels
    result['close'] = df['close']
    result['open'] = df['open']
    result['high'] = df['high']
    result['low'] = df['low']
    
    # Derived price levels
    result['hl2'] = (df['high'] + df['low']) / 2
    result['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    result['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Returns with NaN protection
    result['returns'] = df['close'].pct_change()
    result['returns'] = result['returns'].replace([np.inf, -np.inf], np.nan)
    
    # Log returns with protection
    price_ratio = df['close'] / df['close'].shift(1)
    # Protect against log(0) or log(negative)
    price_ratio = price_ratio.replace(0, np.nan)
    price_ratio = price_ratio.where(price_ratio > 0, np.nan)
    result['log_returns'] = np.log(price_ratio)
    result['log_returns'] = result['log_returns'].replace([np.inf, -np.inf], np.nan)
    
    # Momentum features (price change over N bars)
    momentum_windows = params.get('momentum_windows', [1, 5, 10, 20, 50])
    for n in momentum_windows:
        result[f'momentum_{n}'] = df['close'] - df['close'].shift(n)
        
        # ROC with division-by-zero protection
        shifted_price = df['close'].shift(n).replace(0, np.nan)
        result[f'roc_{n}'] = ((df['close'] - df['close'].shift(n)) / shifted_price) * 100
        result[f'roc_{n}'] = result[f'roc_{n}'].replace([np.inf, -np.inf], np.nan)
    
    # Price range features
    result['range'] = df['high'] - df['low']
    
    # Range percentage with protection
    close_safe = df['close'].replace(0, np.nan)
    result['range_pct'] = ((df['high'] - df['low']) / close_safe) * 100
    result['range_pct'] = result['range_pct'].replace([np.inf, -np.inf], np.nan)
    
    # Gap detection
    result['gap_up'] = df['open'] > df['high'].shift(1)
    result['gap_down'] = df['open'] < df['low'].shift(1)
    result['gap_size'] = df['open'] - df['close'].shift(1)
    
    # Gap size percentage with protection
    shifted_close = df['close'].shift(1).replace(0, np.nan)
    result['gap_size_pct'] = ((df['open'] - df['close'].shift(1)) / shifted_close) * 100
    result['gap_size_pct'] = result['gap_size_pct'].replace([np.inf, -np.inf], np.nan)
    
    return result


def get_available_features() -> list:
    """Return list of all available price features."""
    return [
        'close', 'open', 'high', 'low',
        'hl2', 'hlc3', 'ohlc4',
        'returns', 'log_returns',
        'momentum_1', 'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
        'roc_1', 'roc_5', 'roc_10', 'roc_20', 'roc_50',
        'range', 'range_pct',
        'gap_up', 'gap_down', 'gap_size', 'gap_size_pct'
    ]
