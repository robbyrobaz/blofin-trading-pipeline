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
    """Calculate Bollinger Bands with NaN protection."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = sma + (std * std_mult)
    lower_band = sma - (std * std_mult)
    
    # Width calculation with division-by-zero protection
    sma_safe = sma.replace(0, np.nan)
    width = ((upper_band - lower_band) / sma_safe) * 100
    width = width.replace([np.inf, -np.inf], np.nan)
    
    # %B indicator (position within bands) with protection
    band_range = (upper_band - lower_band).replace(0, np.nan)
    percent_b = (series - lower_band) / band_range
    percent_b = percent_b.replace([np.inf, -np.inf], np.nan)
    
    return {
        'bbands_upper': upper_band,
        'bbands_middle': sma,
        'bbands_lower': lower_band,
        'bbands_width': width,
        'bbands_percent_b': percent_b
    }


def compute_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Keltner Channels with NaN protection."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    basis = typical_price.ewm(span=period, adjust=False).mean()
    atr = compute_atr(df, period)
    
    upper_channel = basis + (atr * atr_mult)
    lower_channel = basis - (atr * atr_mult)
    
    # Width calculation with division-by-zero protection
    basis_safe = basis.replace(0, np.nan)
    width = ((upper_channel - lower_channel) / basis_safe) * 100
    width = width.replace([np.inf, -np.inf], np.nan)
    
    return {
        'keltner_upper': upper_channel,
        'keltner_middle': basis,
        'keltner_lower': lower_channel,
        'keltner_width': width
    }


def compute_historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate historical volatility (annualized) with NaN protection."""
    # Protect against log(0) and log(negative)
    price_ratio = series / series.shift(1)
    price_ratio = price_ratio.replace(0, np.nan)
    price_ratio = price_ratio.where(price_ratio > 0, np.nan)
    
    log_returns = np.log(price_ratio)
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    
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
        
        # ATR as percentage of close price with protection
        close_safe = df['close'].replace(0, np.nan)
        result[f'atr_{period}_pct'] = (result[f'atr_{period}'] / close_safe) * 100
        result[f'atr_{period}_pct'] = result[f'atr_{period}_pct'].replace([np.inf, -np.inf], np.nan)
    
    # Standard Deviation
    std_periods = params.get('std_periods', [20])
    for period in std_periods:
        result[f'std_dev_{period}'] = df['close'].rolling(window=period).std()
        
        # Std dev as percentage with protection
        close_safe = df['close'].replace(0, np.nan)
        result[f'std_dev_{period}_pct'] = (result[f'std_dev_{period}'] / close_safe) * 100
        result[f'std_dev_{period}_pct'] = result[f'std_dev_{period}_pct'].replace([np.inf, -np.inf], np.nan)
    
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
    
    # Volatility ratio (short-term vs long-term) with protection
    std_20 = df['close'].rolling(window=20).std()
    std_50 = df['close'].rolling(window=50).std().replace(0, np.nan)
    result['volatility_ratio_20_50'] = std_20 / std_50
    result['volatility_ratio_20_50'] = result['volatility_ratio_20_50'].replace([np.inf, -np.inf], np.nan)
    
    # Price position within Bollinger Bands (squeeze detection)
    bb_20 = compute_bollinger_bands(df['close'], 20, 2.0)
    bb_width = bb_20['bbands_width'].fillna(0)  # Fill NaN for quantile calculation
    threshold = bb_width.rolling(window=50).quantile(0.25)
    result['bb_squeeze'] = (bb_width < threshold).astype(int)
    
    # Parkinson's volatility (high-low range based) with protection
    high_low_ratio = (df['high'] / df['low']).replace(0, np.nan)
    high_low_ratio = high_low_ratio.where(high_low_ratio > 0, np.nan)
    
    log_hl = np.log(high_low_ratio)
    log_hl = log_hl.replace([np.inf, -np.inf], np.nan)
    
    parkinson_base = np.sqrt(1 / (4 * np.log(2)) * log_hl ** 2)
    result['parkinson_volatility_20'] = (
        parkinson_base.rolling(window=20).mean() * np.sqrt(252) * 100
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
