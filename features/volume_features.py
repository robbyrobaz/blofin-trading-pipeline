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
    
    # Volume ratio (current volume / average volume) with protection
    avg_volume_20 = df['volume'].rolling(window=20).mean().replace(0, np.nan)
    result['volume_ratio_20'] = df['volume'] / avg_volume_20
    result['volume_ratio_20'] = result['volume_ratio_20'].replace([np.inf, -np.inf], np.nan)
    
    avg_volume_50 = df['volume'].rolling(window=50).mean().replace(0, np.nan)
    result['volume_ratio_50'] = df['volume'] / avg_volume_50
    result['volume_ratio_50'] = result['volume_ratio_50'].replace([np.inf, -np.inf], np.nan)
    
    # Volume surge detection (volume > 2x average) - handle NaN in avg_volume
    avg_volume_20_filled = df['volume'].rolling(window=20).mean().fillna(1)  # Use 1 to avoid div by 0
    result['volume_surge'] = (df['volume'] > 2 * avg_volume_20_filled).astype(int)
    
    # Volume surge ratio with protection
    avg_volume_20_safe = avg_volume_20_filled.replace(0, np.nan)
    result['volume_surge_ratio'] = df['volume'] / avg_volume_20_safe
    result['volume_surge_ratio'] = result['volume_surge_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # VWAP (Volume Weighted Average Price) with protection
    # Cumulative VWAP - protect against division by zero
    cumsum_volume = df['volume'].cumsum()
    cumsum_volume_safe = cumsum_volume.replace(0, np.nan)
    result['vwap'] = (df['close'] * df['volume']).cumsum() / cumsum_volume_safe
    result['vwap'] = result['vwap'].replace([np.inf, -np.inf], np.nan)
    
    # Rolling VWAP (more practical for trading)
    vwap_window = params.get('vwap_window', 20)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    rolling_volume_sum = df['volume'].rolling(window=vwap_window).sum().replace(0, np.nan)
    result[f'vwap_{vwap_window}'] = (
        (typical_price * df['volume']).rolling(window=vwap_window).sum() / rolling_volume_sum
    )
    result[f'vwap_{vwap_window}'] = result[f'vwap_{vwap_window}'].replace([np.inf, -np.inf], np.nan)
    
    # VWAP deviation with protection
    vwap_safe = result[f'vwap_{vwap_window}'].replace(0, np.nan)
    result[f'vwap_deviation_{vwap_window}'] = (
        (df['close'] - result[f'vwap_{vwap_window}']) / vwap_safe
    ) * 100
    result[f'vwap_deviation_{vwap_window}'] = result[f'vwap_deviation_{vwap_window}'].replace([np.inf, -np.inf], np.nan)
    
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
    
    # Volume price trend - protect against NaN in obv.diff() and pct_change()
    obv_diff = result['obv'].diff()
    price_pct = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
    result['volume_price_trend'] = obv_diff * price_pct
    result['volume_price_trend'] = result['volume_price_trend'].replace([np.inf, -np.inf], np.nan)
    
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
