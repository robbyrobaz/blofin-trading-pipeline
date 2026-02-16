#!/usr/bin/env python3
"""
Timeframe aggregation module.
Converts 1-minute OHLCV data to higher timeframes (5m, 15m, 60m, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def aggregate_ohlcv(candles_1m: List[Dict[str, Any]], target_minutes: int) -> List[Dict[str, Any]]:
    """
    Aggregate 1-minute candles into higher timeframes.
    
    Args:
        candles_1m: List of 1-minute OHLCV candles with keys: ts_ms, open, high, low, close, volume
        target_minutes: Target timeframe in minutes (e.g., 5, 15, 60)
    
    Returns:
        List of aggregated OHLCV candles
    
    Example:
        >>> candles_1m = [
        ...     {'ts_ms': 1000000, 'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 10},
        ...     {'ts_ms': 1060000, 'open': 101, 'high': 103, 'low': 100, 'close': 102, 'volume': 15},
        ... ]
        >>> aggregate_ohlcv(candles_1m, 5)
    """
    if not candles_1m:
        return []
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(candles_1m)
    
    # Ensure ts_ms is sorted
    df = df.sort_values('ts_ms')
    
    # Convert ts_ms to datetime for grouping
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
    
    # Group by target timeframe
    df.set_index('timestamp', inplace=True)
    
    # Resample to target timeframe
    freq = f'{target_minutes}min'
    
    aggregated = df.resample(freq).agg({
        'ts_ms': 'first',       # First timestamp of the period
        'open': 'first',        # First open price
        'high': 'max',          # Highest high
        'low': 'min',           # Lowest low
        'close': 'last',        # Last close price
        'volume': 'sum'         # Sum of volumes
    })
    
    # Remove rows with NaN (incomplete periods)
    aggregated = aggregated.dropna()
    
    # Convert back to list of dicts
    result = []
    for idx, row in aggregated.iterrows():
        result.append({
            'ts_ms': int(row['ts_ms']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
    
    return result


def aggregate_ticks_to_1m_ohlcv(ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate raw ticks into 1-minute OHLCV candles.
    
    Args:
        ticks: List of tick data with keys: ts_ms, price, volume (optional)
    
    Returns:
        List of 1-minute OHLCV candles
    """
    if not ticks:
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(ticks)
    
    # Ensure ts_ms is sorted
    df = df.sort_values('ts_ms')
    
    # Convert ts_ms to datetime
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
    
    # Add volume column if not present (default to 1)
    if 'volume' not in df.columns:
        df['volume'] = 1.0
    
    # Group by 1-minute intervals
    df.set_index('timestamp', inplace=True)
    
    aggregated = df.resample('1min').agg({
        'ts_ms': 'first',
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    
    # Flatten column names
    aggregated.columns = ['ts_ms', 'open', 'high', 'low', 'close', 'volume']
    
    # Remove rows with NaN
    aggregated = aggregated.dropna()
    
    # Convert to list of dicts
    result = []
    for idx, row in aggregated.iterrows():
        result.append({
            'ts_ms': int(row['ts_ms']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })
    
    return result


def fast_aggregate_numpy(candles_1m: np.ndarray, target_minutes: int) -> np.ndarray:
    """
    Fast OHLCV aggregation using pure numpy (no pandas overhead).
    
    Args:
        candles_1m: Numpy array with shape (N, 6) where columns are [ts_ms, open, high, low, close, volume]
        target_minutes: Target timeframe in minutes
    
    Returns:
        Numpy array with aggregated candles
    """
    if len(candles_1m) == 0:
        return np.array([])
    
    # Sort by timestamp
    candles_1m = candles_1m[candles_1m[:, 0].argsort()]
    
    # Calculate period boundaries (milliseconds)
    period_ms = target_minutes * 60 * 1000
    
    # Find first period start (round down to nearest period)
    first_ts = candles_1m[0, 0]
    first_period_start = (first_ts // period_ms) * period_ms
    
    # Group candles by period
    periods = ((candles_1m[:, 0] - first_period_start) // period_ms).astype(int)
    
    # Find unique periods
    unique_periods = np.unique(periods)
    
    result = []
    for period_idx in unique_periods:
        mask = periods == period_idx
        period_candles = candles_1m[mask]
        
        if len(period_candles) == 0:
            continue
        
        # Aggregate OHLCV
        ts_ms = period_candles[0, 0]  # First timestamp
        open_price = period_candles[0, 1]  # First open
        high_price = period_candles[:, 2].max()  # Max high
        low_price = period_candles[:, 3].min()  # Min low
        close_price = period_candles[-1, 4]  # Last close
        volume = period_candles[:, 5].sum()  # Sum volume
        
        result.append([ts_ms, open_price, high_price, low_price, close_price, volume])
    
    return np.array(result)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes.
    
    Args:
        timeframe: String like '1m', '5m', '15m', '1h', '4h', '1d'
    
    Returns:
        Number of minutes
    
    Examples:
        >>> timeframe_to_minutes('5m')
        5
        >>> timeframe_to_minutes('1h')
        60
        >>> timeframe_to_minutes('1d')
        1440
    """
    timeframe = timeframe.lower()
    
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 24
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")
