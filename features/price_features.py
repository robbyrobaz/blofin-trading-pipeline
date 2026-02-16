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
    
    # Returns
    result['returns'] = df['close'].pct_change()
    result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Momentum features (price change over N bars)
    momentum_windows = params.get('momentum_windows', [1, 5, 10, 20, 50])
    for n in momentum_windows:
        result[f'momentum_{n}'] = df['close'] - df['close'].shift(n)
        result[f'roc_{n}'] = ((df['close'] - df['close'].shift(n)) / df['close'].shift(n)) * 100
    
    # Price range features
    result['range'] = df['high'] - df['low']
    result['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
    
    # Gap detection
    result['gap_up'] = df['open'] > df['high'].shift(1)
    result['gap_down'] = df['open'] < df['low'].shift(1)
    result['gap_size'] = df['open'] - df['close'].shift(1)
    result['gap_size_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
    
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
