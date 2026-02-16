"""
Volatility-based features and indicators.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR (Average True Range)."""
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = sma + (std * std_mult)
    lower_band = sma - (std * std_mult)
    width = ((upper_band - lower_band) / sma) * 100
    
    # %B indicator (position within bands)
    percent_b = (series - lower_band) / (upper_band - lower_band)
    
    return {
        'bbands_upper': upper_band,
        'bbands_middle': sma,
        'bbands_lower': lower_band,
        'bbands_width': width,
        'bbands_percent_b': percent_b
    }


def compute_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Keltner Channels."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    basis = typical_price.ewm(span=period, adjust=False).mean()
    atr = compute_atr(df, period)
    
    upper_channel = basis + (atr * atr_mult)
    lower_channel = basis - (atr * atr_mult)
    width = ((upper_channel - lower_channel) / basis) * 100
    
    return {
        'keltner_upper': upper_channel,
        'keltner_middle': basis,
        'keltner_lower': lower_channel,
        'keltner_width': width
    }


def compute_historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate historical volatility (annualized)."""
    log_returns = np.log(series / series.shift(1))
    volatility = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized (252 trading days)
    return volatility * 100  # Convert to percentage


def compute_volatility_features(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Compute volatility-based features from OHLCV dataframe.
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Optional parameters dict
        
    Returns:
        DataFrame with computed features
    """
    params = params or {}
    result = df.copy()
    
    # ATR (Average True Range)
    atr_periods = params.get('atr_periods', [14])
    for period in atr_periods:
        result[f'atr_{period}'] = compute_atr(df, period)
        # ATR as percentage of close price
        result[f'atr_{period}_pct'] = (result[f'atr_{period}'] / df['close']) * 100
    
    # Standard Deviation
    std_periods = params.get('std_periods', [20])
    for period in std_periods:
        result[f'std_dev_{period}'] = df['close'].rolling(window=period).std()
        result[f'std_dev_{period}_pct'] = (result[f'std_dev_{period}'] / df['close']) * 100
    
    # Bollinger Bands
    bb_configs = params.get('bb_configs', [(20, 2.0)])
    for period, std_mult in bb_configs:
        bb_result = compute_bollinger_bands(df['close'], period, std_mult)
        result[f'bbands_upper_{period}'] = bb_result['bbands_upper']
        result[f'bbands_middle_{period}'] = bb_result['bbands_middle']
        result[f'bbands_lower_{period}'] = bb_result['bbands_lower']
        result[f'bbands_width_{period}'] = bb_result['bbands_width']
        result[f'bbands_percent_b_{period}'] = bb_result['bbands_percent_b']
    
    # Keltner Channels
    keltner_configs = params.get('keltner_configs', [(20, 2.0)])
    for period, atr_mult in keltner_configs:
        keltner_result = compute_keltner_channels(df, period, atr_mult)
        result[f'keltner_upper_{period}'] = keltner_result['keltner_upper']
        result[f'keltner_middle_{period}'] = keltner_result['keltner_middle']
        result[f'keltner_lower_{period}'] = keltner_result['keltner_lower']
        result[f'keltner_width_{period}'] = keltner_result['keltner_width']
    
    # Historical Volatility
    hv_periods = params.get('hv_periods', [20, 50])
    for period in hv_periods:
        result[f'historical_volatility_{period}'] = compute_historical_volatility(df['close'], period)
    
    # Volatility ratio (short-term vs long-term)
    result['volatility_ratio_20_50'] = (
        df['close'].rolling(window=20).std() / df['close'].rolling(window=50).std()
    )
    
    # Price position within Bollinger Bands (squeeze detection)
    bb_20 = compute_bollinger_bands(df['close'], 20, 2.0)
    result['bb_squeeze'] = (bb_20['bbands_width'] < bb_20['bbands_width'].rolling(window=50).quantile(0.25)).astype(int)
    
    # Parkinson's volatility (high-low range based)
    result['parkinson_volatility_20'] = (
        np.sqrt(1 / (4 * np.log(2)) * np.log(df['high'] / df['low']) ** 2)
        .rolling(window=20).mean() * np.sqrt(252) * 100
    )
    
    return result


def get_available_features() -> list:
    """Return list of all available volatility features."""
    return [
        'atr_14', 'atr_14_pct',
        'std_dev_20', 'std_dev_20_pct',
        'bbands_upper_20', 'bbands_middle_20', 'bbands_lower_20', 
        'bbands_width_20', 'bbands_percent_b_20',
        'keltner_upper_20', 'keltner_middle_20', 'keltner_lower_20', 
        'keltner_width_20',
        'historical_volatility_20', 'historical_volatility_50',
        'volatility_ratio_20_50',
        'bb_squeeze',
        'parkinson_volatility_20'
    ]
