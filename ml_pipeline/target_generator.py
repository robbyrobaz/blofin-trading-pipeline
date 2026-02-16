"""Generate training targets for ML models from price data."""

import pandas as pd
import numpy as np


def add_targets(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Add target columns for ML training.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Candles to look ahead for target
        
    Returns:
        DataFrame with added target columns
    """
    df = df.copy()
    
    # Target 1: Price direction (UP=1, DOWN=0) for next 5 candles
    future_price = df['close'].shift(-lookback)
    df['target_direction'] = (future_price > df['close']).astype(int)
    
    # Target 2: Price prediction (actual price)
    df['target_price'] = future_price
    
    # Target 3: Momentum direction (accelerating=1, decelerating=0)
    df['momentum'] = df['close'].diff()
    future_momentum = df['momentum'].shift(-lookback)
    df['target_momentum'] = (future_momentum > df['momentum']).astype(int)
    
    # Target 4: Risk level (based on next N candles volatility)
    future_returns = df['close'].pct_change().rolling(lookback).std().shift(-lookback)
    df['target_volatility'] = future_returns
    
    # Remove NaN rows (edge cases)
    df = df.dropna()
    
    return df


def prepare_features_and_targets(df: pd.DataFrame, lookback: int = 5):
    """
    Prepare X (features) and y (targets) for ML models.
    
    Args:
        df: DataFrame with features and targets
        lookback: Used for target generation
        
    Returns:
        Tuple of (X, y_dict) where y_dict has all targets
    """
    # Add targets if not present
    if 'target_direction' not in df.columns:
        df = add_targets(df, lookback=lookback)
    
    # Define feature columns (exclude OHLCV and targets)
    exclude_cols = {
        'timestamp', 'open', 'high', 'low', 'close', 'tick_count', 'volume',
        'target_direction', 'target_price', 'target_momentum', 'target_volatility',
        'momentum'
    }
    
    X_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[X_cols]
    
    # Targets dict
    y_dict = {
        'direction': df['target_direction'],
        'price': df['target_price'],
        'momentum': df['target_momentum'],
        'volatility': df['target_volatility']
    }
    
    return X, y_dict, X_cols
