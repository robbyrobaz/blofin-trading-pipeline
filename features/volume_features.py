"""
Volume-based features and indicators.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_volume_features(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute volume-based features from OHLCV dataframe.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Optional parameters dict (e.g., SMA/EMA windows)
        
    Returns:
        DataFrame with computed features
    """
    params = params or {}
    result = df.copy()
    
    # Raw volume
    result['volume'] = df['volume']
    
    # Volume moving averages
    volume_windows = params.get('volume_windows', [5, 10, 20, 50])
    for n in volume_windows:
        result[f'volume_sma_{n}'] = df['volume'].rolling(window=n).mean()
        result[f'volume_ema_{n}'] = df['volume'].ewm(span=n, adjust=False).mean()
    
    # Volume ratio (current volume / average volume)
    result['volume_ratio_20'] = df['volume'] / df['volume'].rolling(window=20).mean()
    result['volume_ratio_50'] = df['volume'] / df['volume'].rolling(window=50).mean()
    
    # Volume surge detection (volume > 2x average)
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    result['volume_surge'] = (df['volume'] > 2 * avg_volume_20).astype(int)
    result['volume_surge_ratio'] = df['volume'] / avg_volume_20
    
    # VWAP (Volume Weighted Average Price)
    result['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Rolling VWAP (more practical for trading)
    vwap_window = params.get('vwap_window', 20)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    result[f'vwap_{vwap_window}'] = (
        (typical_price * df['volume']).rolling(window=vwap_window).sum() /
        df['volume'].rolling(window=vwap_window).sum()
    )
    
    # VWAP deviation
    result[f'vwap_deviation_{vwap_window}'] = (
        (df['close'] - result[f'vwap_{vwap_window}']) / result[f'vwap_{vwap_window}']
    ) * 100
    
    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    result['obv'] = obv
    
    # OBV moving averages
    result['obv_sma_20'] = result['obv'].rolling(window=20).mean()
    result['obv_ema_20'] = result['obv'].ewm(span=20, adjust=False).mean()
    
    # Volume price trend
    result['volume_price_trend'] = (
        result['obv'].diff() * df['close'].pct_change()
    )
    
    return result


def get_available_features() -> list:
    """Return list of all available volume features."""
    return [
        'volume',
        'volume_sma_5', 'volume_sma_10', 'volume_sma_20', 'volume_sma_50',
        'volume_ema_5', 'volume_ema_10', 'volume_ema_20', 'volume_ema_50',
        'volume_ratio_20', 'volume_ratio_50',
        'volume_surge', 'volume_surge_ratio',
        'vwap', 'vwap_20',
        'vwap_deviation_20',
        'obv', 'obv_sma_20', 'obv_ema_20',
        'volume_price_trend'
    ]
